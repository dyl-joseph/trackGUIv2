import contextlib
import hashlib
import io
import os
import threading
from dataclasses import dataclass

import numpy as np
from PIL import Image


class Sam2ServiceError(RuntimeError):
    def __init__(self, status_code, detail):
        super().__init__(detail)
        self.status_code = status_code
        self.detail = detail


@dataclass
class ImageRecord:
    image_id: str
    image: np.ndarray
    width: int
    height: int
    client_frame_key: str = None


def decode_image(image_bytes):
    try:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception as exc:
        raise Sam2ServiceError(400, "Cannot decode uploaded image.") from exc
    return np.asarray(image)


def mask_to_bbox(mask):
    mask = np.asarray(mask).astype(bool)
    if mask.ndim != 2:
        raise Sam2ServiceError(422, "SAM2 returned an invalid mask.")
    if not mask.any():
        raise Sam2ServiceError(422, "SAM2 returned an empty mask.")
    ys, xs = np.where(mask)
    return [
        float(xs.min()),
        float(ys.min()),
        float(xs.max()),
        float(ys.max()),
    ]


class Sam2Service:
    def __init__(
        self,
        predictor=None,
        model_name="sam2.1",
        model_cfg=None,
        checkpoint=None,
        device=None,
    ):
        self._predictor = predictor
        self._model_cfg = model_cfg
        self._checkpoint = checkpoint
        self._device = device or "cuda"
        self._torch = None
        self._lock = threading.Lock()
        self._images = {}
        self._current_image_id = None
        self.model_name = model_name

    @classmethod
    def from_env(cls):
        return cls(
            model_name=os.environ.get("SAM2_MODEL_NAME", "sam2.1"),
            model_cfg=os.environ.get("SAM2_MODEL_CFG"),
            checkpoint=os.environ.get("SAM2_CHECKPOINT"),
            device=os.environ.get("SAM2_DEVICE", "cuda"),
        )

    def health(self):
        return {
            "ready": self._predictor is not None,
            "model": self.model_name,
            "cached_images": len(self._images),
        }

    def register_image(self, image_bytes, client_frame_key=None):
        if not image_bytes:
            raise Sam2ServiceError(400, "Uploaded image is empty.")

        image_id = hashlib.sha256(image_bytes).hexdigest()
        existing = self._images.get(image_id)
        if existing is not None:
            return {
                "image_id": existing.image_id,
                "width": existing.width,
                "height": existing.height,
                "prepared": True,
            }

        image = decode_image(image_bytes)
        height, width = image.shape[:2]
        record = ImageRecord(
            image_id=image_id,
            image=image,
            width=width,
            height=height,
            client_frame_key=client_frame_key,
        )

        with self._lock:
            self._ensure_predictor()
            with self._inference_context():
                self._predictor.set_image(record.image)
            self._current_image_id = image_id
            self._images[image_id] = record

        return {
            "image_id": image_id,
            "width": width,
            "height": height,
            "prepared": True,
        }

    def point_prompt(self, image_id, x, y, label=1):
        record = self._images.get(image_id)
        if record is None:
            raise Sam2ServiceError(404, "Unknown image_id.")
        x = float(x)
        y = float(y)
        if not (0 <= x <= record.width - 1 and 0 <= y <= record.height - 1):
            raise Sam2ServiceError(400, "Point prompt is outside the image.")
        if label not in [0, 1]:
            raise Sam2ServiceError(400, "Point label must be 0 or 1.")

        with self._lock:
            self._ensure_predictor()
            with self._inference_context():
                if self._current_image_id != image_id:
                    self._predictor.set_image(record.image)
                    self._current_image_id = image_id
                masks, scores, _ = self._predictor.predict(
                    point_coords=np.array([[x, y]], dtype=np.float32),
                    point_labels=np.array([label], dtype=np.int64),
                    multimask_output=True,
                )

        mask = self._select_mask(masks, scores)
        return {
            "bbox": mask_to_bbox(mask),
            "score": self._select_score(scores),
            "model": self.model_name,
        }

    def _select_mask(self, masks, scores):
        masks = np.asarray(masks)
        if masks.ndim == 2:
            return masks
        if masks.ndim != 3:
            raise Sam2ServiceError(422, "SAM2 returned masks with invalid shape.")
        scores = np.asarray(scores).reshape(-1)
        index = int(scores.argmax()) if scores.size else 0
        return masks[index]

    def _select_score(self, scores):
        scores = np.asarray(scores).reshape(-1)
        if not scores.size:
            return None
        return float(scores.max())

    def _ensure_predictor(self):
        if self._predictor is not None:
            return
        if not self._model_cfg or not self._checkpoint:
            raise Sam2ServiceError(
                503,
                "SAM2_MODEL_CFG and SAM2_CHECKPOINT must be set before inference.",
            )
        try:
            import torch
            from sam2.build_sam import build_sam2
            from sam2.sam2_image_predictor import SAM2ImagePredictor
        except Exception as exc:
            raise Sam2ServiceError(
                503,
                "SAM2 is not installed in this backend environment.",
            ) from exc

        model = build_sam2(self._model_cfg, self._checkpoint, device=self._device)
        self._predictor = SAM2ImagePredictor(model)
        self._torch = torch

    @contextlib.contextmanager
    def _inference_context(self):
        if self._torch is None:
            yield
            return
        with self._torch.inference_mode():
            if str(self._device).startswith("cuda"):
                with self._torch.autocast("cuda", dtype=self._torch.bfloat16):
                    yield
            else:
                yield
