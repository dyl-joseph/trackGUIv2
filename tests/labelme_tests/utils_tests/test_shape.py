import numpy as np
import pytest

from labelme import utils
from labelme.utils import shape as shape_module

from .util import get_img_and_data


def test_shapes_to_label():
    img, data = get_img_and_data()
    label_name_to_value = {}
    for shape in data["shapes"]:
        label_name = shape["label"]
        label_value = len(label_name_to_value)
        label_name_to_value[label_name] = label_value
    cls, _ = shape_module.shapes_to_label(
        img.shape, data["shapes"], label_name_to_value
    )
    assert cls.shape == img.shape[:2]


def test_shape_to_mask():
    img, data = get_img_and_data()
    for shape in data["shapes"]:
        points = shape["points"]
        mask = shape_module.shape_to_mask(img.shape[:2], points)
        assert mask.shape == img.shape[:2]


def test_inverted_rectangle_is_normalized_before_rasterizing():
    mask = shape_module.shape_to_mask(
        (10, 10), [[8, 7], [2, 3]], shape_type="rectangle"
    )

    assert mask[3, 2]
    assert mask[7, 8]
    assert not mask[0, 0]


def test_stored_mask_shape_is_placed_at_origin_with_clipping():
    stored_mask = np.array([[True, False], [False, True]], dtype=bool)
    shape = {
        "label": "object",
        "group_id": 1,
        "shape_type": "mask",
        "points": [[-1, 1], [1, 3]],
        "mask": utils.img_arr_to_b64(stored_mask),
    }

    labels, _ = shape_module.shapes_to_label((4, 4), [shape], {"object": 2})

    assert labels[2, 0] == 2
    assert np.count_nonzero(labels) == 1


def test_masks_to_bboxes_rejects_empty_mask_explicitly():
    with pytest.raises(ValueError, match="index 0 is empty"):
        shape_module.masks_to_bboxes(np.zeros((1, 3, 3), dtype=bool))
