import threading

import numpy as np
from PIL import Image

import labelme.app
import labelme.track_algo
from labelme.tracking_utils import load_oriented_cv_image


def test_csrt_worker_normalizes_boxes_and_reports_exact_stop(monkeypatch):
    class FakeTracker:
        def __init__(self):
            self.updates = iter(
                [
                    (True, (-2, -3, 20, 20)),
                    (False, (0, 0, 0, 0)),
                ]
            )

        def init(self, frame, bbox):
            return True

        def update(self, frame):
            return next(self.updates)

    class FakeTrackerFactory:
        @staticmethod
        def create():
            return FakeTracker()

    frames = {path: np.zeros((8, 9, 3), dtype=np.uint8) for path in ["0", "1", "2"]}
    monkeypatch.setattr(labelme.app, "load_oriented_cv_image", frames.get)
    monkeypatch.setattr(
        labelme.app.cv2, "TrackerCSRT", FakeTrackerFactory, raising=False
    )
    progress = []

    result = labelme.app.MainWindow._runCsrtTracking(
        tuple(frames),
        0,
        3,
        (1, 1, 3, 3),
        threading.Event(),
        lambda value, message: progress.append((value, message)),
    )

    assert result["frames"][0]["points"] == [[0, 0], [9, 8]]
    assert result["last_index"] == 1
    assert result["stop_reason"] == "tracking failed on frame 3"
    assert [value for value, _ in progress] == [1, 2, 3]


def test_csrt_worker_cancellation_produces_no_partial_result(monkeypatch):
    class FakeTracker:
        def init(self, frame, bbox):
            return True

    class FakeTrackerFactory:
        @staticmethod
        def create():
            return FakeTracker()

    monkeypatch.setattr(
        labelme.app,
        "load_oriented_cv_image",
        lambda _: np.zeros((4, 4, 3), dtype=np.uint8),
    )
    monkeypatch.setattr(
        labelme.app.cv2, "TrackerCSRT", FakeTrackerFactory, raising=False
    )
    canceled = threading.Event()
    canceled.set()

    result = labelme.app.MainWindow._runCsrtTracking(
        ("0", "1"), 0, 2, (0, 0, 2, 2), canceled, lambda *_: None
    )

    assert result["frames"] == []
    assert result["stop_reason"] == "canceled"


def test_csrt_worker_rejects_fully_offscreen_box(monkeypatch):
    class FakeTracker:
        def init(self, frame, bbox):
            return True

        def update(self, frame):
            return True, (20, 2, 5, 4)

    class FakeTrackerFactory:
        @staticmethod
        def create():
            return FakeTracker()

    monkeypatch.setattr(
        labelme.app,
        "load_oriented_cv_image",
        lambda _: np.zeros((10, 10, 3), dtype=np.uint8),
    )
    monkeypatch.setattr(
        labelme.app.cv2, "TrackerCSRT", FakeTrackerFactory, raising=False
    )

    result = labelme.app.MainWindow._runCsrtTracking(
        ("0", "1"), 0, 2, (1, 1, 3, 3), threading.Event(), lambda *_: None
    )

    assert result["frames"] == []
    assert result["stop_reason"] == "tracker returned an empty box on frame 2"


def test_botsort_worker_rejects_fully_offscreen_box(monkeypatch):
    class FakeTracker:
        def init(self, frame, initial_box):
            return True

        def update(self, frame):
            return True, [20, 2, 25, 6]

        def reset(self):
            pass

    monkeypatch.setattr(labelme.track_algo, "BoTSORTForwardTracker", FakeTracker)
    monkeypatch.setattr(
        labelme.app,
        "load_oriented_cv_image",
        lambda _: np.zeros((10, 10, 3), dtype=np.uint8),
    )

    result = labelme.app.MainWindow._runBoTSORTTracking(
        ("0", "1"),
        0,
        2,
        [1, 1, 4, 4],
        False,
        threading.Event(),
        lambda *_: None,
    )

    assert result["frames"] == []
    assert result["stop_reason"] == "tracker returned an empty box on frame 2"


def test_csrt_worker_decodes_exif_rotated_frames_in_display_orientation(
    tmp_path, monkeypatch
):
    frame_paths = [tmp_path / "0.jpg", tmp_path / "1.jpg"]
    exif = Image.Exif()
    exif[274] = 6
    for frame_path in frame_paths:
        Image.new("RGB", (20, 10), (10, 20, 30)).save(frame_path, exif=exif)

    class FakeTracker:
        initialized_shape = None

        def init(self, frame, bbox):
            type(self).initialized_shape = frame.shape
            return True

        def update(self, frame):
            assert frame.shape[:2] == (20, 10)
            return True, (1, 2, 4, 6)

    class FakeTrackerFactory:
        @staticmethod
        def create():
            return FakeTracker()

    monkeypatch.setattr(
        labelme.app.cv2, "TrackerCSRT", FakeTrackerFactory, raising=False
    )

    oriented = load_oriented_cv_image(str(frame_paths[0]))
    result = labelme.app.MainWindow._runCsrtTracking(
        tuple(str(path) for path in frame_paths),
        0,
        2,
        (1, 2, 4, 6),
        threading.Event(),
        lambda *_: None,
    )

    assert oriented.shape[:2] == (20, 10)
    assert FakeTracker.initialized_shape[:2] == (20, 10)
    assert result["frames"][0]["image_shape"][:2] == (20, 10)
