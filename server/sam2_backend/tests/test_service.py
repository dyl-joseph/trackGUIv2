import hashlib
import io

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

    assert response["bbox"] == [3.0, 2.0, 6.0, 4.0]
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
