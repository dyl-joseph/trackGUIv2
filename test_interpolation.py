import pytest

from labelme.tracking_utils import interpolation_indices
from labelme.tracking_utils import normalized_rectangle_points


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


def test_normalized_rectangle_points_reorders_inverted_corners():
    assert normalized_rectangle_points([[8, 7], [2, 3]]) == [
        [2.0, 3.0],
        [8.0, 7.0],
    ]


def test_normalized_rectangle_points_rejects_degenerate_box():
    with pytest.raises(ValueError, match="positive width and height"):
        normalized_rectangle_points([[2, 3], [2, 8]])
