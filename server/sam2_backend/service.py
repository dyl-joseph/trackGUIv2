import contextlib
import hashlib
import io
import math
import os
import threading
from collections import OrderedDict
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


def decode_image(image_bytes, max_pixels=40_000_000):
    try:
        source = Image.open(io.BytesIO(image_bytes))
        width, height = source.size
        if width <= 0 or height <= 0 or width * height > max_pixels:
            raise Sam2ServiceError(
                413,
                "Decoded image exceeds the {} pixel limit.".format(max_pixels),
            )
        image = source.convert("RGB")
    except Sam2ServiceError:
        raise
    except Exception as exc:
        raise Sam2ServiceError(400, "Cannot decode uploaded image.") from exc
    return np.asarray(image)


def mask_to_bbox(mask):
    try:
        mask = np.asarray(mask)
        if mask.dtype != bool:
            numeric_mask = np.asarray(mask, dtype=float)
            if not np.isfinite(numeric_mask).all():
                raise ValueError
            mask = numeric_mask.astype(bool)
    except (TypeError, ValueError) as exc:
        raise Sam2ServiceError(422, "SAM2 returned an invalid mask.") from exc
    if mask.ndim != 2:
        raise Sam2ServiceError(422, "SAM2 returned an invalid mask.")
    if not mask.any():
        raise Sam2ServiceError(422, "SAM2 returned an empty mask.")
    ys, xs = np.where(mask)
    return [
        float(xs.min()),
        float(ys.min()),
        float(xs.max() + 1),
        float(ys.max() + 1),
    ]


class Sam2Service:
    def __init__(
        self,
        predictor=None,
        model_name="sam2.1",
        model_cfg=None,
        checkpoint=None,
        device=None,
        max_upload_bytes=25 * 1024 * 1024,
        max_pixels=40_000_000,
        max_cached_images=8,
    ):
        self._predictor = predictor
        self._model_cfg = model_cfg
        self._checkpoint = checkpoint
        self._device = device or "cuda"
        self._torch = None
        self._lock = threading.Lock()
        self._images = OrderedDict()
        self._current_image_id = None
        self.model_name = model_name
        self.max_upload_bytes = int(max_upload_bytes)
        self.max_pixels = int(max_pixels)
        self.max_cached_images = int(max_cached_images)
        if min(self.max_upload_bytes, self.max_pixels, self.max_cached_images) <= 0:
            raise ValueError("SAM2 resource limits must be positive")

    @classmethod
    def from_env(cls):
        return cls(
            model_name=os.environ.get("SAM2_MODEL_NAME", "sam2.1"),
            model_cfg=os.environ.get("SAM2_MODEL_CFG"),
            checkpoint=os.environ.get("SAM2_CHECKPOINT"),
            device=os.environ.get("SAM2_DEVICE", "cuda"),
            max_upload_bytes=os.environ.get("SAM2_MAX_UPLOAD_BYTES", 25 * 1024 * 1024),
            max_pixels=os.environ.get("SAM2_MAX_PIXELS", 40_000_000),
            max_cached_images=os.environ.get("SAM2_CACHE_FRAMES", 8),
        )

    def health(self):
        return {
            "status": "ok",
            "model": self.model_name,
            "cached_images": len(self._images),
        }

    def readiness(self):
        try:
            with self._lock:
                self._ensure_predictor()
        except Sam2ServiceError as exc:
            return {
                "ready": False,
                "model": self.model_name,
                "detail": exc.detail,
            }
        return {
            "ready": True,
            "model": self.model_name,
            "detail": None,
        }

    def register_image(self, image_bytes, client_frame_key=None):
        if not image_bytes:
            raise Sam2ServiceError(400, "Uploaded image is empty.")
        if len(image_bytes) > self.max_upload_bytes:
            raise Sam2ServiceError(
                413,
                "Upload exceeds the {} byte limit.".format(self.max_upload_bytes),
            )

        image_id = hashlib.sha256(image_bytes).hexdigest()
        image = decode_image(image_bytes, max_pixels=self.max_pixels)
        height, width = image.shape[:2]
        record = ImageRecord(
            image_id=image_id,
            image=image,
            width=width,
            height=height,
            client_frame_key=client_frame_key,
        )

        with self._lock:
            existing = self._images.get(image_id)
            if existing is not None:
                self._images.move_to_end(image_id)
                return {
                    "image_id": existing.image_id,
                    "width": existing.width,
                    "height": existing.height,
                    "prepared": self._current_image_id == image_id,
                }
            self._ensure_predictor()
            with self._inference_context():
                self._predictor.set_image(record.image)
            self._current_image_id = image_id
            self._images[image_id] = record
            self._images.move_to_end(image_id)
            while len(self._images) > self.max_cached_images:
                evicted_id, _ = self._images.popitem(last=False)
                if evicted_id == self._current_image_id:
                    self._current_image_id = None

        return {
            "image_id": image_id,
            "width": width,
            "height": height,
            "prepared": True,
        }

    def point_prompt(self, image_id, x, y, label=1):
        if not isinstance(image_id, str) or not image_id.strip():
            raise Sam2ServiceError(400, "image_id must be a non-empty string.")
        if isinstance(x, bool) or isinstance(y, bool):
            raise Sam2ServiceError(400, "Point coordinates must be finite numbers.")
        try:
            x = float(x)
            y = float(y)
        except (TypeError, ValueError) as exc:
            raise Sam2ServiceError(
                400, "Point coordinates must be finite numbers."
            ) from exc
        if not math.isfinite(x) or not math.isfinite(y):
            raise Sam2ServiceError(400, "Point coordinates must be finite numbers.")
        if isinstance(label, bool) or label not in [0, 1]:
            raise Sam2ServiceError(400, "Point label must be 0 or 1.")

        with self._lock:
            record = self._images.get(image_id)
            if record is None:
                raise Sam2ServiceError(404, "Unknown image_id.")
            self._images.move_to_end(image_id)
            if not (0 <= x <= record.width - 1 and 0 <= y <= record.height - 1):
                raise Sam2ServiceError(400, "Point prompt is outside the image.")
            self._ensure_predictor()
            with self._inference_context():
                if self._current_image_id != image_id:
                    self._predictor.set_image(record.image)
                    self._current_image_id = image_id
                result = self._predictor.predict(
                    point_coords=np.array([[x, y]], dtype=np.float32),
                    point_labels=np.array([label], dtype=np.int64),
                    multimask_output=True,
                )
            if not isinstance(result, (tuple, list)) or len(result) < 2:
                raise Sam2ServiceError(422, "SAM2 returned an invalid result tuple.")
            masks, scores = result[:2]

        mask = self._select_mask(
            masks,
            scores,
            expected_shape=(record.height, record.width),
        )
        return {
            "bbox": mask_to_bbox(mask),
            "score": self._select_score(scores),
            "model": self.model_name,
        }

    def _select_mask(self, masks, scores, expected_shape=None):
        try:
            masks = np.asarray(masks)
        except (TypeError, ValueError) as exc:
            raise Sam2ServiceError(
                422, "SAM2 returned masks with invalid shape."
            ) from exc
        if masks.ndim == 2:
            masks = masks[None, ...]
        if masks.ndim != 3 or masks.shape[0] == 0:
            raise Sam2ServiceError(422, "SAM2 returned masks with invalid shape.")
        try:
            if masks.dtype != bool:
                numeric_masks = np.asarray(masks, dtype=float)
                if not np.isfinite(numeric_masks).all():
                    raise ValueError
                masks = numeric_masks.astype(bool)
            scores = np.asarray(scores, dtype=float).reshape(-1)
        except (TypeError, ValueError) as exc:
            raise Sam2ServiceError(
                422, "SAM2 returned non-numeric masks or scores."
            ) from exc
        if scores.size != masks.shape[0]:
            raise Sam2ServiceError(422, "SAM2 mask and score counts do not match.")
        if scores.size and not np.isfinite(scores).all():
            raise Sam2ServiceError(422, "SAM2 returned non-finite scores.")
        if expected_shape is not None and tuple(masks.shape[1:]) != tuple(
            expected_shape
        ):
            raise Sam2ServiceError(422, "SAM2 mask dimensions do not match the image.")
        index = int(scores.argmax())
        return masks[index]

    def _select_score(self, scores):
        try:
            scores = np.asarray(scores, dtype=float).reshape(-1)
        except (TypeError, ValueError) as exc:
            raise Sam2ServiceError(422, "SAM2 returned invalid scores.") from exc
        if not scores.size:
            return None
        if not np.isfinite(scores).all():
            raise Sam2ServiceError(422, "SAM2 returned non-finite scores.")
        return float(scores.max())

    def _ensure_predictor(self):
        if self._predictor is not None:
            return
        if not self._model_cfg or not self._checkpoint:
            raise Sam2ServiceError(
                503,
                "SAM2_MODEL_CFG and SAM2_CHECKPOINT must be set before inference.",
            )
        if not os.path.isfile(self._checkpoint):
            raise Sam2ServiceError(
                503,
                "SAM2_CHECKPOINT must reference an existing file.",
            )
        try:
            import torch
            from sam2.build_sam import build_sam2
            from sam2.sam2_image_predictor import SAM2ImagePredictor

            model = build_sam2(self._model_cfg, self._checkpoint, device=self._device)
            self._predictor = SAM2ImagePredictor(model)
            self._torch = torch
        except Exception as exc:
            raise Sam2ServiceError(
                503,
                "SAM2 predictor initialization failed: {}".format(exc),
            ) from exc

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
