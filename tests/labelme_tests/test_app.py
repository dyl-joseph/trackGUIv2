import json
import os.path as osp
import shutil
import tempfile

import pytest
from qtpy import QtCore
from qtpy import QtGui
from qtpy import QtWidgets

import labelme.app
import labelme.config
import labelme.testing
from labelme.label_file import LabelFile
from labelme.shape import Shape

here = osp.dirname(osp.abspath(__file__))
data_dir = osp.join(here, "data")


def _win_show_and_wait_imageData(qtbot, win):
    win.show()

    def check_imageData():
        assert hasattr(win, "imageData")
        assert win.imageData is not None

    qtbot.waitUntil(check_imageData)  # wait for loadFile


@pytest.mark.gui
def test_MainWindow_open(qtbot):
    win = labelme.app.MainWindow()
    qtbot.addWidget(win)
    win.show()
    win.close()


@pytest.mark.gui
def test_MainWindow_open_img(qtbot):
    img_file = osp.join(data_dir, "raw/2011_000003.jpg")
    win = labelme.app.MainWindow(filename=img_file)
    qtbot.addWidget(win)
    _win_show_and_wait_imageData(qtbot, win)
    win.close()


@pytest.mark.gui
def test_MainWindow_open_json(qtbot):
    json_files = [
        osp.join(data_dir, "annotated_with_data/apc2016_obj3.json"),
        osp.join(data_dir, "annotated/2011_000003.json"),
    ]
    for json_file in json_files:
        labelme.testing.assert_labelfile_sanity(json_file)

        win = labelme.app.MainWindow(filename=json_file)
        qtbot.addWidget(win)
        _win_show_and_wait_imageData(qtbot, win)
        win.close()


def create_MainWindow_with_directory(qtbot):
    directory = osp.join(data_dir, "raw")
    win = labelme.app.MainWindow(filename=directory)
    qtbot.addWidget(win)
    _win_show_and_wait_imageData(qtbot, win)
    return win


@pytest.mark.gui
def test_MainWindow_openNextImg(qtbot):
    win = create_MainWindow_with_directory(qtbot)
    win.openNextImg()


@pytest.mark.gui
def test_MainWindow_openPrevImg(qtbot):
    win = create_MainWindow_with_directory(qtbot)
    win.openNextImg()


@pytest.mark.gui
def test_MainWindow_annotate_jpg(qtbot):
    tmp_dir = tempfile.mkdtemp()
    input_file = osp.join(data_dir, "raw/2011_000003.jpg")
    out_file = osp.join(tmp_dir, "2011_000003.json")

    config = labelme.config.get_default_config()
    win = labelme.app.MainWindow(
        config=config,
        filename=input_file,
        output_file=out_file,
    )
    qtbot.addWidget(win)
    _win_show_and_wait_imageData(qtbot, win)

    label = "whole"
    points = [
        (100, 100),
        (100, 238),
        (400, 238),
        (400, 100),
    ]
    shapes = [
        dict(
            label=label,
            group_id=None,
            points=points,
            shape_type="polygon",
            mask=None,
            flags={},
            other_data={},
        )
    ]
    win.loadLabels(shapes)
    win.saveFile()

    labelme.testing.assert_labelfile_sanity(out_file)
    shutil.rmtree(tmp_dir)


@pytest.mark.gui
def test_new_shape_uses_single_prompt_and_auto_track_id(qtbot):
    class FakeLabelDialog:
        def __init__(self):
            self.edit = QtWidgets.QLineEdit()
            self.popups = 0

        def popUp(self, text=None):
            self.popups += 1
            return "person", {}, None, ""

        def addLabelHistory(self, label):
            pass

    class FailingIDDialog:
        def __init__(self):
            self.edit = QtWidgets.QLineEdit()
            self.history = []

        def popUp(self, text=None):
            raise AssertionError("newShape should not open the ID dialog")

        def addIDHistory(self, track_id):
            self.history.append(track_id)

    config = labelme.config.get_default_config()
    win = labelme.app.MainWindow(config=config)
    qtbot.addWidget(win)
    win.mode = "NORMAL"
    win.labelDialog = FakeLabelDialog()
    win.IDDialog = FailingIDDialog()

    shape = Shape(shape_type="rectangle")
    shape.addPoint(QtCore.QPointF(10, 10))
    shape.addPoint(QtCore.QPointF(20, 20))
    shape.close()
    win.canvas.shapes.append(shape)
    win.canvas.storeShapes()

    win.newShape()

    assert win.labelDialog.popups == 1
    assert shape.label == "person"
    assert shape.track_id == "1"
    assert win.IDDialog.history == ["1"]


@pytest.mark.gui
def test_track_modification_reject_does_not_change_labels(qtbot, monkeypatch):
    class RejectedDeletionDialog:
        def __init__(self, parent=None):
            self.start_frame_cell = QtWidgets.QLineEdit("1")
            self.end_frame_cell = QtWidgets.QLineEdit("2")
            self.ID_cell = QtWidgets.QLineEdit("7")
            self.label_cell = QtWidgets.QLineEdit("person")
            self.new_ID_cell = QtWidgets.QLineEdit("")
            self.new_label_cell = QtWidgets.QLineEdit("")

        @property
        def mode(self):
            return "Remove Box"

        def exec_(self):
            return QtWidgets.QDialog.Rejected

    tmp_dir = tempfile.mkdtemp()
    try:
        image_file = osp.join(tmp_dir, "000001.jpg")
        image_file_2 = osp.join(tmp_dir, "000002.jpg")
        shutil.copy(osp.join(data_dir, "raw/2011_000003.jpg"), image_file)
        shutil.copy(osp.join(data_dir, "raw/2011_000003.jpg"), image_file_2)

        shape = dict(
            label="person",
            group_id=7,
            track_id="7",
            points=[(10, 10), (20, 20)],
            shape_type="rectangle",
            flags={},
            description=None,
            mask=None,
        )
        for image_path in [image_file, image_file_2]:
            LabelFile().save(
                filename=osp.splitext(image_path)[0] + ".json",
                shapes=[shape.copy()],
                imagePath=osp.basename(image_path),
                imageData=None,
                imageHeight=10,
                imageWidth=10,
                flags={},
            )

        before = json.load(open(osp.splitext(image_file)[0] + ".json"))

        config = labelme.config.get_default_config()
        win = labelme.app.MainWindow(config=config)
        qtbot.addWidget(win)
        win._imageListCache = [image_file, image_file_2]
        win.lastOpenDir = tmp_dir
        win.image = QtGui.QImage(10, 10, QtGui.QImage.Format_RGB32)

        monkeypatch.setattr(labelme.app, "DeletionDialog", RejectedDeletionDialog)

        win.DELETION()

        after = json.load(open(osp.splitext(image_file)[0] + ".json"))
        assert after == before
    finally:
        shutil.rmtree(tmp_dir)


@pytest.mark.gui
def test_track_modification_swap_id_updates_without_deleting(qtbot, monkeypatch):
    class AcceptedSwapIDDialog:
        def __init__(self, parent=None):
            self.start_frame_cell = QtWidgets.QLineEdit("1")
            self.end_frame_cell = QtWidgets.QLineEdit("2")
            self.ID_cell = QtWidgets.QLineEdit("7")
            self.label_cell = QtWidgets.QLineEdit("person")
            self.new_ID_cell = QtWidgets.QLineEdit("9")
            self.new_label_cell = QtWidgets.QLineEdit("")

        @property
        def mode(self):
            return "Swap ID"

        def exec_(self):
            return QtWidgets.QDialog.Accepted

    tmp_dir = tempfile.mkdtemp()
    try:
        image_files = [
            osp.join(tmp_dir, "000001.jpg"),
            osp.join(tmp_dir, "000002.jpg"),
        ]
        for image_file in image_files:
            shutil.copy(osp.join(data_dir, "raw/2011_000003.jpg"), image_file)

        person = dict(
            label="person",
            group_id=7,
            track_id="7",
            points=[(10, 10), (20, 20)],
            shape_type="rectangle",
            flags={},
            description=None,
            mask=None,
        )
        other = dict(
            label="person",
            group_id=8,
            track_id="8",
            points=[(30, 30), (40, 40)],
            shape_type="rectangle",
            flags={},
            description=None,
            mask=None,
        )
        for image_file in image_files:
            LabelFile().save(
                filename=osp.splitext(image_file)[0] + ".json",
                shapes=[person.copy(), other.copy()],
                imagePath=osp.basename(image_file),
                imageData=None,
                imageHeight=10,
                imageWidth=10,
                flags={},
            )

        config = labelme.config.get_default_config()
        win = labelme.app.MainWindow(config=config)
        qtbot.addWidget(win)
        win._imageListCache = image_files
        win.lastOpenDir = tmp_dir
        win.image = QtGui.QImage(10, 10, QtGui.QImage.Format_RGB32)
        win.loadFile = lambda filename=None: None
        win.informationMessage = lambda title, message: None
        win.errorMessage = lambda title, message: pytest.fail(message)

        monkeypatch.setattr(labelme.app, "DeletionDialog", AcceptedSwapIDDialog)

        win.DELETION()

        for image_file in image_files:
            data = json.load(open(osp.splitext(image_file)[0] + ".json"))
            assert len(data["shapes"]) == 2
            assert data["shapes"][0]["label"] == "person"
            assert data["shapes"][0]["track_id"] == "9"
            assert data["shapes"][0]["group_id"] == 9
            assert data["shapes"][1]["track_id"] == "8"
            assert data["shapes"][1]["group_id"] == 8
    finally:
        shutil.rmtree(tmp_dir)


@pytest.mark.gui
def test_track_modification_remove_box_with_new_id_does_not_delete(
    qtbot, monkeypatch
):
    class MisconfiguredRemoveDialog:
        def __init__(self, parent=None):
            self.start_frame_cell = QtWidgets.QLineEdit("1")
            self.end_frame_cell = QtWidgets.QLineEdit("2")
            self.ID_cell = QtWidgets.QLineEdit("7")
            self.label_cell = QtWidgets.QLineEdit("person")
            self.new_ID_cell = QtWidgets.QLineEdit("9")
            self.new_label_cell = QtWidgets.QLineEdit("")

        @property
        def mode(self):
            return "Remove Box"

        def exec_(self):
            return QtWidgets.QDialog.Accepted

    tmp_dir = tempfile.mkdtemp()
    try:
        image_files = [
            osp.join(tmp_dir, "000001.jpg"),
            osp.join(tmp_dir, "000002.jpg"),
        ]
        for image_file in image_files:
            shutil.copy(osp.join(data_dir, "raw/2011_000003.jpg"), image_file)

        shape = dict(
            label="person",
            group_id=7,
            track_id="7",
            points=[(10, 10), (20, 20)],
            shape_type="rectangle",
            flags={},
            description=None,
            mask=None,
        )
        for image_file in image_files:
            LabelFile().save(
                filename=osp.splitext(image_file)[0] + ".json",
                shapes=[shape.copy()],
                imagePath=osp.basename(image_file),
                imageData=None,
                imageHeight=10,
                imageWidth=10,
                flags={},
            )

        first_json = osp.splitext(image_files[0])[0] + ".json"
        before = json.load(open(first_json))

        config = labelme.config.get_default_config()
        win = labelme.app.MainWindow(config=config)
        qtbot.addWidget(win)
        win._imageListCache = image_files
        win.lastOpenDir = tmp_dir
        win.image = QtGui.QImage(10, 10, QtGui.QImage.Format_RGB32)
        win.informationMessage = lambda title, message: pytest.fail(message)
        errors = []
        win.errorMessage = lambda title, message: errors.append((title, message))

        monkeypatch.setattr(labelme.app, "DeletionDialog", MisconfiguredRemoveDialog)

        win.DELETION()

        after = json.load(open(first_json))
        assert after == before
        assert errors == [
            (
                "Track Modification",
                "Remove Box deletes matching boxes. Choose Swap ID or Swap Label "
                "to apply the new value.",
            )
        ]
    finally:
        shutil.rmtree(tmp_dir)
