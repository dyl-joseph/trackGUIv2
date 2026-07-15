import pytest

from labelme.tracking_utils import interpolation_indices
from labelme.tracking_utils import intersect_xyxy_with_image
from labelme.tracking_utils import normalized_rectangle_points
from labelme.tracking_utils import prediction_to_clamped_rectangle
from labelme.tracking_utils import upsert_tracked_rectangle


def test_interpolation_indices_use_exact_interval_and_include_endpoint():
    assert interpolation_indices(1, 10, 3, frame_count=12) == [0, 3, 6, 9]
    assert interpolation_indices(2, 10, 3, frame_count=12) == [1, 4, 7, 9]


@pytest.mark.parametrize(
    "start,end,interval",
    [(0, 2, 1), (-1, 2, 1), (2, 1, 1), (1, 13, 1), (1, 2, 0), (1, 2, -1)],
)
def test_interpolation_indices_reject_invalid_ranges(start, end, interval):
    with pytest.raises(ValueError):
        interpolation_indices(start, end, interval, frame_count=12)


def test_upsert_preserves_falsy_id_type_metadata_and_removes_duplicates():
    shapes = [
        {
            "label": "person",
            "track_id": 0,
            "group_id": 7,
            "points": [[1, 1], [2, 2]],
            "shape_type": "rectangle",
            "flags": {"occluded": True},
            "custom": "keep",
        },
        {
            "label": "person",
            "track_id": "0",
            "group_id": 8,
            "points": [[3, 3], [4, 4]],
            "shape_type": "rectangle",
        },
    ]

    updated = upsert_tracked_rectangle(shapes, "person", "0", [[10, 11], [20, 21]])

    assert updated == [
        {
            "label": "person",
            "track_id": 0,
            "group_id": 7,
            "points": [[10, 11], [20, 21]],
            "shape_type": "rectangle",
            "flags": {"occluded": True},
            "custom": "keep",
            "description": "",
            "mask": None,
        }
    ]


def test_normalized_rectangle_points_reorders_inverted_corners():
    assert normalized_rectangle_points([[8, 7], [2, 3]]) == [
        [2.0, 3.0],
        [8.0, 7.0],
    ]


def test_normalized_rectangle_points_rejects_degenerate_box():
    with pytest.raises(ValueError, match="positive width and height"):
        normalized_rectangle_points([[2, 3], [2, 8]])


@pytest.mark.parametrize(
    "points",
    [
        [[float("nan"), 0], [2, 3]],
        [[0, float("inf")], [2, 3]],
        [[False, 0], [2, 3]],
        [["not-a-number", 0], [2, 3]],
    ],
)
def test_normalized_rectangle_points_rejects_nonfinite_coordinates(points):
    with pytest.raises(ValueError, match="finite numbers"):
        normalized_rectangle_points(points)


def test_prediction_to_clamped_rectangle_rejects_negative_size_before_conversion():
    with pytest.raises(ValueError, match="non-positive"):
        prediction_to_clamped_rectangle([10, 10, -4, 6], 20, 20)


def test_prediction_to_clamped_rectangle_clamps_to_target_image():
    assert prediction_to_clamped_rectangle([1, 1, 6, 6], 5, 4) == [
        [0.0, 0.0],
        [4.0, 4.0],
    ]


@pytest.mark.parametrize(
    "prediction",
    [
        [-10, 5, 2, 2],
        [20, 5, 2, 2],
        [5, -10, 2, 2],
        [5, 20, 2, 2],
    ],
)
def test_prediction_to_clamped_rectangle_rejects_fully_offscreen_box(prediction):
    with pytest.raises(ValueError, match="empty box"):
        prediction_to_clamped_rectangle(prediction, 10, 10)


def test_rectangle_intersection_clips_partial_overlap_without_creating_border_box():
    assert intersect_xyxy_with_image([-5, 2, 3, 15], 10, 10) == [
        [0.0, 2.0],
        [3.0, 10.0],
    ]
    with pytest.raises(ValueError, match="does not intersect"):
        intersect_xyxy_with_image([11, 2, 15, 8], 10, 10)


def test_upsert_refuses_to_convert_conflicting_nonrectangle():
    shapes = [
        {
            "label": "person",
            "track_id": 2,
            "points": [[0, 0], [1, 0], [1, 1]],
            "shape_type": "polygon",
        }
    ]

    with pytest.raises(ValueError, match="non-rectangle"):
        upsert_tracked_rectangle(shapes, "person", 2, [[2, 2], [4, 4]])


def test_upsert_refuses_missing_id_without_collapsing_untracked_rectangles():
    shapes = [
        {
            "label": "person",
            "track_id": None,
            "group_id": None,
            "points": [[0, 0], [2, 2]],
            "shape_type": "rectangle",
        },
        {
            "label": "person",
            "track_id": None,
            "group_id": None,
            "points": [[3, 3], [5, 5]],
            "shape_type": "rectangle",
        },
    ]

    with pytest.raises(ValueError, match="track ID"):
        upsert_tracked_rectangle(shapes, "person", None, [[6, 6], [8, 8]])

    assert len(shapes) == 2
    assert [shape["points"] for shape in shapes] == [
        [[0, 0], [2, 2]],
        [[3, 3], [5, 5]],
    ]
