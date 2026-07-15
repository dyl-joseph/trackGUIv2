import threading

import numpy as np

import labelme.app


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
    monkeypatch.setattr(labelme.app.cv2, "imread", frames.get)
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
        labelme.app.cv2,
        "imread",
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
