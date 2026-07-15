import os
import os.path as osp
import shutil
import tempfile
from pathlib import Path

import pytest
from qtpy import QtWidgets

import labelme.app
import labelme.config
from labelme.label_file import LabelFile
from labelme.track_algo import botsort_tracker

here = osp.dirname(osp.abspath(__file__))
data_dir = osp.join(here, "data")


def _write_frame_sequence(directory, frame_count=40):
    source_image = Path(data_dir) / "raw" / "2011_000003.jpg"
    shape = dict(
        label="person",
        group_id=1,
        track_id="1",
        points=[(10, 10), (20, 20)],
        shape_type="rectangle",
        flags={},
        description=None,
        mask=None,
    )
    for index in range(frame_count):
        image_path = directory / f"frame{index:06d}.jpg"
        shutil.copy(source_image, image_path)
        LabelFile().save(
            filename=str(image_path.with_suffix(".json")),
            shapes=[shape.copy()],
            imagePath=image_path.name,
            imageData=None,
            imageHeight=10,
            imageWidth=10,
            flags={},
        )


@pytest.mark.gui
@pytest.mark.performance
def test_repeated_frame_navigation_ignores_inactive_ai_model(qtbot, tmp_path):
    class SlowInactiveAiModel:
        name = "slow-inactive"

        def __init__(self):
            self.set_image_calls = 0

        def set_image(self, image):
            self.set_image_calls += 1

    _write_frame_sequence(tmp_path)

    app = QtWidgets.QApplication.instance()
    win = labelme.app.MainWindow(config=labelme.config.get_default_config())
    qtbot.addWidget(win)
    win.importDirImages(str(tmp_path), load=False)
    win.loadFile(win.imageList[0])
    app.processEvents()

    ai_model = SlowInactiveAiModel()
    win.canvas._ai_model = ai_model
    win.canvas.createMode = "polygon"

    frame_steps = 25
    for _ in range(frame_steps):
        win.openNextImg()
        app.processEvents()

    assert ai_model.set_image_calls == 0


@pytest.mark.performance
def test_default_mplconfigdir_uses_os_temp_directory():
    expected = Path(tempfile.gettempdir()) / "labelme-matplotlib"
    assert Path(os.environ["MPLCONFIGDIR"]) == expected
    assert expected.is_dir()


def test_botsort_model_uses_ultralytics_cache_when_not_local(monkeypatch, tmp_path):
    package_root = tmp_path / "labelme"
    track_algo_dir = package_root / "track_algo"
    run_dir = tmp_path / "run"
    track_algo_dir.mkdir(parents=True)
    run_dir.mkdir()

    monkeypatch.setattr(
        botsort_tracker,
        "__file__",
        str(track_algo_dir / "botsort_tracker.py"),
    )
    monkeypatch.chdir(run_dir)

    assert botsort_tracker._resolve_model_path("yolo11n.pt") == "yolo11n.pt"
