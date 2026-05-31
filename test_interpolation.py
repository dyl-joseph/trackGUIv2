import numpy as np
import pytest


def _as_bbox_array(bboxes):
    bboxes = np.asarray(bboxes, dtype=float)
    if bboxes.ndim != 2 or bboxes.shape[1] != 4:
        raise ValueError("Expected bounding boxes with shape (N, 4).")
    return bboxes


def cvt_xyxy2xywh(old_bboxes):
    old_bboxes = _as_bbox_array(old_bboxes)
    new_bboxes = np.zeros(old_bboxes.shape, dtype=float)
    new_bboxes[:, 0] = (old_bboxes[:, 0] + old_bboxes[:, 2]) / 2
    new_bboxes[:, 1] = (old_bboxes[:, 1] + old_bboxes[:, 3]) / 2
    new_bboxes[:, 2] = old_bboxes[:, 2] - old_bboxes[:, 0]
    new_bboxes[:, 3] = old_bboxes[:, 3] - old_bboxes[:, 1]
    return new_bboxes


def cvt_xywh2xyxy(old_bboxes):
    old_bboxes = _as_bbox_array(old_bboxes)
    new_bboxes = np.zeros(old_bboxes.shape, dtype=float)
    dw = old_bboxes[:, 2] / 2
    dh = old_bboxes[:, 3] / 2
    new_bboxes[:, 0] = old_bboxes[:, 0] - dw
    new_bboxes[:, 1] = old_bboxes[:, 1] - dh
    new_bboxes[:, 2] = old_bboxes[:, 0] + dw
    new_bboxes[:, 3] = old_bboxes[:, 1] + dh
    return new_bboxes


def test_bbox_conversion_roundtrip():
    xyxy = np.array([[10, 20, 30, 50], [0, 0, 8, 12]], dtype=float)
    xywh = cvt_xyxy2xywh(xyxy)

    np.testing.assert_allclose(
        xywh,
        np.array([[20, 35, 20, 30], [4, 6, 8, 12]], dtype=float),
    )
    np.testing.assert_allclose(cvt_xywh2xyxy(xywh), xyxy)


def test_bbox_conversion_rejects_flat_input():
    with pytest.raises(ValueError, match="shape"):
        cvt_xyxy2xywh([10, 20, 30, 50])
