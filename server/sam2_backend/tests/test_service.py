import hashlib
import io
import sys
import types

import numpy as np
import pytest
from PIL import Image

from server.sam2_backend.service import Sam2Service
from server.sam2_backend.service import Sam2ServiceError
from server.sam2_backend.service import mask_to_bbox


class FakePredictor:
    def __init__(self, mask):
        self.mask = mask
        self.images = []

    def set_image(self, image):
        self.images.append(image)

    def predict(self, point_coords, point_labels, multimask_output=True):
        return np.asarray([self.mask]), np.asarray([0.75]), None


def _image_bytes(width=8, height=6):
    output = io.BytesIO()
    Image.new("RGB", (width, height), color=(10, 20, 30)).save(output, format="PNG")
    return output.getvalue()


def test_register_image_prepares_and_returns_hash_dimensions():
    image_bytes = _image_bytes(width=9, height=7)
    service = Sam2Service(predictor=FakePredictor(np.zeros((7, 9), dtype=bool)))

    response = service.register_image(image_bytes, client_frame_key="frame-1")

    assert response == {
        "image_id": hashlib.sha256(image_bytes).hexdigest(),
        "width": 9,
        "height": 7,
        "prepared": True,
    }
    assert len(service._predictor.images) == 1


def test_point_prompt_returns_bbox_from_best_mask():
    mask = np.zeros((6, 8), dtype=bool)
    mask[2:5, 3:7] = True
    service = Sam2Service(predictor=FakePredictor(mask))
    registered = service.register_image(_image_bytes(width=8, height=6))

    response = service.point_prompt(
        image_id=registered["image_id"],
        x=4,
        y=3,
        label=1,
    )

    assert response["bbox"] == [3.0, 2.0, 7.0, 5.0]
    assert response["score"] == 0.75
    assert response["model"] == "sam2.1"


def test_point_prompt_rejects_out_of_bounds_point():
    service = Sam2Service(predictor=FakePredictor(np.zeros((6, 8), dtype=bool)))
    registered = service.register_image(_image_bytes(width=8, height=6))

    with pytest.raises(Sam2ServiceError) as exc_info:
        service.point_prompt(registered["image_id"], x=9, y=3)

    assert exc_info.value.status_code == 400
    assert exc_info.value.detail == "Point prompt is outside the image."


def test_mask_to_bbox_rejects_empty_mask():
    with pytest.raises(Sam2ServiceError) as exc_info:
        mask_to_bbox(np.zeros((4, 4), dtype=bool))

    assert exc_info.value.status_code == 422
    assert exc_info.value.detail == "SAM2 returned an empty mask."


def test_register_image_enforces_upload_and_decoded_pixel_limits():
    upload_limited = Sam2Service(
        predictor=FakePredictor(np.zeros((1, 1), dtype=bool)),
        max_upload_bytes=2,
    )
    with pytest.raises(Sam2ServiceError) as upload_error:
        upload_limited.register_image(b"123")
    assert upload_error.value.status_code == 413

    pixel_limited = Sam2Service(
        predictor=FakePredictor(np.zeros((3, 3), dtype=bool)),
        max_pixels=8,
    )
    with pytest.raises(Sam2ServiceError) as pixel_error:
        pixel_limited.register_image(_image_bytes(width=3, height=3))
    assert pixel_error.value.status_code == 413


def test_image_cache_evicts_least_recently_used_frame():
    service = Sam2Service(
        predictor=FakePredictor(np.zeros((6, 8), dtype=bool)),
        max_cached_images=2,
    )
    first = service.register_image(_image_bytes(width=8, height=6))
    second = service.register_image(_image_bytes(width=9, height=6))
    service.register_image(_image_bytes(width=10, height=6))

    with pytest.raises(Sam2ServiceError, match="Unknown image_id") as exc_info:
        service.point_prompt(first["image_id"], 1, 1)

    assert exc_info.value.status_code == 404
    assert second["image_id"] in service._images


@pytest.mark.parametrize(
    "masks,scores,detail",
    [
        (np.empty((0, 3, 3)), np.empty(0), "invalid shape"),
        (np.zeros((1, 3, 3)), np.empty(0), "counts do not match"),
        (np.zeros((2, 3, 3)), np.array([0.5]), "counts do not match"),
        (np.zeros((1, 3, 3)), np.array([np.nan]), "non-finite"),
        (np.array([[["bad"]]]), np.array([0.5]), "non-numeric"),
    ],
)
def test_select_mask_rejects_malformed_model_outputs(masks, scores, detail):
    service = Sam2Service(predictor=FakePredictor(np.zeros((3, 3))))

    with pytest.raises(Sam2ServiceError, match=detail):
        service._select_mask(masks, scores)


def test_readiness_distinguishes_loaded_predictor_from_missing_configuration():
    ready = Sam2Service(predictor=FakePredictor(np.zeros((2, 2), dtype=bool)))
    not_ready = Sam2Service()

    assert ready.readiness()["ready"] is True
    assert not_ready.health()["status"] == "ok"
    assert not_ready.readiness()["ready"] is False


def test_readiness_surfaces_predictor_initialization_failure(monkeypatch):
    service = Sam2Service()

    def fail_to_initialize():
        raise Sam2ServiceError(503, "checkpoint is incompatible")

    monkeypatch.setattr(service, "_ensure_predictor", fail_to_initialize)

    readiness = service.readiness()

    assert readiness["ready"] is False
    assert readiness["detail"] == "checkpoint is incompatible"


def test_predictor_accepts_installed_package_hydra_config_name(tmp_path, monkeypatch):
    checkpoint = tmp_path / "model.pt"
    checkpoint.write_bytes(b"checkpoint")
    calls = []

    def build_sam2(model_cfg, checkpoint_path, device):
        calls.append((model_cfg, checkpoint_path, device))
        return object()

    class FakeImagePredictor:
        def __init__(self, model):
            self.model = model

    sam2_package = types.ModuleType("sam2")
    sam2_package.__path__ = []
    build_module = types.ModuleType("sam2.build_sam")
    build_module.build_sam2 = build_sam2
    predictor_module = types.ModuleType("sam2.sam2_image_predictor")
    predictor_module.SAM2ImagePredictor = FakeImagePredictor
    monkeypatch.setitem(sys.modules, "torch", types.ModuleType("torch"))
    monkeypatch.setitem(sys.modules, "sam2", sam2_package)
    monkeypatch.setitem(sys.modules, "sam2.build_sam", build_module)
    monkeypatch.setitem(sys.modules, "sam2.sam2_image_predictor", predictor_module)
    config_name = "configs/sam2.1/sam2.1_hiera_l.yaml"
    service = Sam2Service(
        model_cfg=config_name,
        checkpoint=str(checkpoint),
        device="cpu",
    )

    service._ensure_predictor()

    assert calls == [(config_name, str(checkpoint), "cpu")]
    assert isinstance(service._predictor, FakeImagePredictor)


def test_point_prompt_rejects_boolean_label_and_invalid_image_id():
    service = Sam2Service(predictor=FakePredictor(np.ones((6, 8), dtype=bool)))
    registered = service.register_image(_image_bytes(width=8, height=6))

    with pytest.raises(Sam2ServiceError, match="label"):
        service.point_prompt(registered["image_id"], 1, 1, label=True)
    with pytest.raises(Sam2ServiceError, match="image_id"):
        service.point_prompt([], 1, 1)
