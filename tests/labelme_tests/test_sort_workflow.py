import copy
from types import SimpleNamespace

import numpy as np
import pytest
from qtpy import QtWidgets

import labelme.app
import labelme.config


def _rectangle(track_id, points, label="person"):
    return {
        "label": label,
        "track_id": track_id,
        "group_id": track_id,
        "points": points,
        "shape_type": "rectangle",
        "flags": {},
        "description": "",
        "mask": None,
    }


def _polygon(track_id=99):
    return {
        "label": "zone",
        "track_id": track_id,
        "group_id": track_id,
        "points": [[0, 0], [2, 0], [1, 2]],
        "shape_type": "polygon",
        "flags": {},
        "description": "",
        "mask": None,
    }


def _run_sort(
    qtbot,
    monkeypatch,
    frame_shapes,
    tracks_provider,
    *,
    option=1,
    current_index=0,
    end_frame=None,
):
    frame_paths = ["/frames/{}.jpg".format(index) for index in range(len(frame_shapes))]
    end_frame = len(frame_paths) if end_frame is None else end_frame

    class Value:
        def value(self):
            return end_frame

    class Dialog:
        def __init__(self, **_kwargs):
            self.option_value = option
            self.end_frame = Value()

        def exec_(self):
            return QtWidgets.QDialog.Accepted

    class FakeKalmanBoxTracker:
        count = 0
        instances = []

        def __init__(self, bbox, id):
            self.bbox = np.asarray(bbox, dtype=float)
            self.id = id
            type(self).instances.append(self)

    class FakeSort:
        instances = []

        def __init__(self, **_kwargs):
            self.trackers = []
            self.calls = []
            type(self).instances.append(self)

        def update(self, detections):
            detections = np.asarray(detections, dtype=float)
            self.calls.append(detections.copy())
            rows = tracks_provider(len(self.calls) - 1, detections, self)
            return np.asarray(rows, dtype=float).reshape((-1, 5))

    labels = {
        path: SimpleNamespace(shapes=copy.deepcopy(shapes))
        for path, shapes in zip(frame_paths, frame_shapes)
    }
    win = labelme.app.MainWindow(config=labelme.config.get_default_config())
    qtbot.addWidget(win)
    win._imageListCache = frame_paths
    win.filename = frame_paths[current_index]
    errors = []
    information = []
    saved_requests = []
    win.errorMessage = lambda title, message: errors.append((title, message))
    win.informationMessage = lambda title, message: information.append((title, message))
    monkeypatch.setattr(win, "_ensureSavedForWorkflow", lambda _title: True)
    monkeypatch.setattr(win, "_loadLabelForImage", labels.__getitem__)
    monkeypatch.setattr(
        win,
        "_labelSaveRequest",
        lambda image_path, shapes, _label_file: {
            "image_path": image_path,
            "shapes": copy.deepcopy(shapes),
        },
    )

    def save_batch(requests, _title):
        saved_requests.extend(requests)
        return True

    monkeypatch.setattr(win, "_saveLabelBatch", save_batch)
    monkeypatch.setattr(win, "loadFile", lambda _filename: True)
    monkeypatch.setattr(labelme.app, "TrackDialog", Dialog)
    monkeypatch.setattr(labelme.app, "SORT_main", FakeSort)
    monkeypatch.setattr(labelme.app, "KalmanBoxTracker", FakeKalmanBoxTracker)

    win.SORT()

    return SimpleNamespace(
        window=win,
        frames=frame_paths,
        tracker=FakeSort.instances[0] if FakeSort.instances else None,
        seeds=FakeKalmanBoxTracker.instances,
        errors=errors,
        information=information,
        requests=saved_requests,
    )


@pytest.mark.gui
def test_sort_from_scratch_seeds_from_first_target_json(qtbot, monkeypatch):
    result = _run_sort(
        qtbot,
        monkeypatch,
        [
            [_rectangle(None, [[1, 2], [5, 6]])],
            [_rectangle(88, [[20, 21], [30, 31]])],
        ],
        lambda _call, detections, _tracker: [list(detections[0, :4]) + [11]],
        option=1,
        current_index=1,
    )

    assert result.tracker.calls[0][0, :4].tolist() == [1.0, 2.0, 5.0, 6.0]
    assert result.requests[0]["image_path"] == result.frames[0]


@pytest.mark.gui
def test_sort_from_current_uses_numeric_ids_from_current_json(qtbot, monkeypatch):
    result = _run_sort(
        qtbot,
        monkeypatch,
        [
            [_rectangle(2, [[0, 0], [2, 2]])],
            [_rectangle("7", [[10, 11], [20, 21]])],
            [_rectangle("7", [[12, 11], [22, 21]])],
        ],
        lambda _call, detections, _tracker: [list(detections[0, :4]) + [0]],
        option=2,
        current_index=1,
    )

    assert len(result.seeds) == 1
    assert result.seeds[0].id == 0
    assert result.seeds[0].bbox.tolist() == [10.0, 11.0, 20.0, 21.0]
    assert result.tracker.calls[0][0, :4].tolist() == [10.0, 11.0, 20.0, 21.0]


@pytest.mark.gui
def test_sort_from_current_rejects_an_end_before_the_current_frame(qtbot, monkeypatch):
    result = _run_sort(
        qtbot,
        monkeypatch,
        [
            [_rectangle(2, [[0, 0], [2, 2]])],
            [_rectangle(7, [[10, 11], [20, 21]])],
        ],
        lambda *_args: pytest.fail("invalid range must not reach SORT.update"),
        option=2,
        current_index=1,
        end_frame=1,
    )

    assert result.errors == [("Track IDs", "End frame must include the start frame.")]


@pytest.mark.gui
@pytest.mark.parametrize(
    "seed_ids,expected_error",
    [
        (["7", "7"], "Every seed rectangle must have a unique track ID."),
        (["7", "not-numeric"], "Every seed rectangle must have a numeric track ID."),
    ],
)
def test_sort_from_current_rejects_ambiguous_seed_ids(
    qtbot, monkeypatch, seed_ids, expected_error
):
    result = _run_sort(
        qtbot,
        monkeypatch,
        [
            [
                _rectangle(seed_ids[0], [[0, 0], [4, 4]]),
                _rectangle(seed_ids[1], [[10, 10], [14, 14]]),
            ]
        ],
        lambda *_args: pytest.fail("invalid seeds must not reach SORT.update"),
        option=2,
    )

    assert result.errors == [("Track IDs", expected_error)]
    assert result.requests == []


@pytest.mark.gui
def test_sort_from_current_preserves_seed_id_representation_and_type(
    qtbot, monkeypatch
):
    def tracks(_call, detections, _tracker):
        return [
            list(detections[0, :4]) + [0],
            list(detections[1, :4]) + [1],
        ]

    result = _run_sort(
        qtbot,
        monkeypatch,
        [
            [
                _rectangle("007", [[0, 0], [4, 4]]),
                _rectangle(7.0, [[10, 10], [14, 14]]),
            ],
            [
                _rectangle("7", [[1, 0], [5, 4]]),
                _rectangle(7, [[11, 10], [15, 14]]),
            ],
        ],
        tracks,
        option=2,
    )

    assert [seed.id for seed in result.seeds] == [0, 1]
    assert len(result.requests) == 1
    saved_ids = [shape["track_id"] for shape in result.requests[0]["shapes"]]
    assert saved_ids == ["007", 7.0]
    assert isinstance(saved_ids[0], str)
    assert isinstance(saved_ids[1], float)


@pytest.mark.gui
def test_sort_passes_an_empty_middle_frame_to_tracker(qtbot, monkeypatch):
    def tracks(_call, detections, _tracker):
        if detections.size == 0:
            return []
        return [list(detections[0, :4]) + [31]]

    result = _run_sort(
        qtbot,
        monkeypatch,
        [
            [_rectangle(None, [[1, 1], [5, 5]])],
            [],
            [_rectangle(None, [[2, 1], [6, 5]])],
        ],
        tracks,
    )

    assert [len(call) for call in result.tracker.calls] == [1, 0, 1]
    assert [request["image_path"] for request in result.requests] == [
        result.frames[0],
        result.frames[2],
    ]


@pytest.mark.gui
def test_sort_skips_nonrectangles_and_normalizes_inverted_boxes(qtbot, monkeypatch):
    result = _run_sort(
        qtbot,
        monkeypatch,
        [[_polygon(), _rectangle(None, [[8, 7], [2, 3]])]],
        lambda _call, detections, _tracker: [list(detections[0, :4]) + [9]],
    )

    assert result.tracker.calls[0][:, :4].tolist() == [[2.0, 3.0, 8.0, 7.0]]
    saved_shapes = result.requests[0]["shapes"]
    assert saved_shapes[0]["shape_type"] == "polygon"
    assert saved_shapes[0]["track_id"] == 99
    assert saved_shapes[1]["track_id"] == "9"


@pytest.mark.gui
def test_sort_uses_hungarian_row_when_fewer_tracks_than_rectangles(qtbot, monkeypatch):
    result = _run_sort(
        qtbot,
        monkeypatch,
        [
            [
                _rectangle("1", [[0, 0], [4, 4]]),
                _rectangle("2", [[100, 100], [104, 104]]),
            ]
        ],
        lambda _call, _detections, _tracker: [[100, 100, 104, 104, 77]],
    )

    saved_shapes = result.requests[0]["shapes"]
    assert [shape["track_id"] for shape in saved_shapes] == ["1", "77"]


@pytest.mark.gui
def test_sort_uses_hungarian_column_for_reordered_tracks(qtbot, monkeypatch):
    result = _run_sort(
        qtbot,
        monkeypatch,
        [
            [
                _rectangle(None, [[0, 0], [4, 4]]),
                _rectangle(None, [[100, 100], [104, 104]]),
            ]
        ],
        lambda _call, _detections, _tracker: [
            [100, 100, 104, 104, 20],
            [0, 0, 4, 4, 10],
        ],
    )

    saved_shapes = result.requests[0]["shapes"]
    assert [shape["track_id"] for shape in saved_shapes] == ["10", "20"]
