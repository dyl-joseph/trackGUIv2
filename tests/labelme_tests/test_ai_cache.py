import collections
import threading

import numpy as np

from labelme.ai import efficient_sam
from labelme.ai import segment_anything_model


def _wait_for_embedding(model):
    model._thread.join()
    model._thread = None


def test_efficient_sam_keeps_only_current_image_embedding():
    class EncoderSession:
        def run(self, output_names, input_feed):
            image = input_feed["batched_images"]
            return (np.array([float(image.sum())], dtype=np.float32),)

    model = efficient_sam.EfficientSam.__new__(efficient_sam.EfficientSam)
    model._encoder_session = EncoderSession()
    model._lock = threading.Lock()
    model._image_embedding_cache = collections.OrderedDict()
    model._thread = None

    for value in range(3):
        image = np.full((4, 4, 4), value, dtype=np.uint8)
        model.set_image(image)
        _wait_for_embedding(model)

        assert len(model._image_embedding_cache) == 1
        assert next(iter(model._image_embedding_cache)) == (
            efficient_sam._image_cache_key(image)
        )


def test_segment_anything_keeps_only_current_image_embedding(monkeypatch):
    def compute_image_embedding(image_size, encoder_session, image):
        return np.array([float(image.sum())], dtype=np.float32)

    monkeypatch.setattr(
        segment_anything_model,
        "_compute_image_embedding",
        compute_image_embedding,
    )

    model = segment_anything_model.SegmentAnythingModel.__new__(
        segment_anything_model.SegmentAnythingModel
    )
    model._image_size = 1024
    model._encoder_session = object()
    model._lock = threading.Lock()
    model._image_embedding_cache = collections.OrderedDict()
    model._thread = None

    for value in range(3):
        image = np.full((4, 4, 4), value, dtype=np.uint8)
        model.set_image(image)
        _wait_for_embedding(model)

        assert len(model._image_embedding_cache) == 1
        assert next(
            iter(model._image_embedding_cache)
        ) == segment_anything_model._image_cache_key(image)
