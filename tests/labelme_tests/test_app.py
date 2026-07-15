import json
import os.path as osp
import shutil
import tempfile

import numpy as np
import pytest
from qtpy import QtCore
from qtpy import QtGui
from qtpy import QtWidgets

import labelme.app
import labelme.config
import labelme.testing
from labelme.hosted_sam2_client import HostedSam2Error
from labelme.label_file import LabelFile
from labelme.shape import Shape
from labelme.widgets import Canvas

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


def _copy_test_image_sequence(tmp_dir):
    input_file = osp.join(data_dir, "raw/2011_000003.jpg")
    image_files = [
        osp.join(tmp_dir, "000001.jpg"),
        osp.join(tmp_dir, "000002.jpg"),
    ]
    for image_file in image_files:
        shutil.copy(input_file, image_file)
    return image_files


def _read_json(filename):
    with open(filename) as f:
        return json.load(f)


class FakeHostedSam2Client:
    def __init__(self, bbox=None, fail_prompt=False):
        self.bbox = bbox or [12.0, 14.0, 40.0, 45.0]
        self.fail_prompt = fail_prompt
        self.register_calls = []
        self.prompt_calls = []

    def is_configured(self):
        return True

    def register_image(self, image_data, client_frame_key=None):
        self.register_calls.append((image_data, client_frame_key))
        image = QtGui.QImage.fromData(image_data)
        return {
            "image_id": "fake-image-id",
            "width": image.width(),
            "height": image.height(),
            "prepared": True,
        }

    def point_prompt(self, image_id, x, y, label=1):
        self.prompt_calls.append((image_id, x, y, label))
        if self.fail_prompt:
            raise HostedSam2Error("prompt failed")
        return {"bbox": list(self.bbox), "score": 0.9, "model": "fake-sam2"}


class FakeLabelDialog:
    def __init__(self, label="person"):
        self.edit = QtWidgets.QLineEdit()
        self.label = label
        self.popups = 0
        self.history = []

    def popUp(self, text=None):
        self.popups += 1
        return self.label, {}, None, ""

    def addLabelHistory(self, label):
        self.history.append(label)


class FailingIDDialog:
    def __init__(self):
        self.edit = QtWidgets.QLineEdit()
        self.history = []

    def popUp(self, text=None):
        raise AssertionError("new SAM2 bbox should not open the ID dialog")

    def addIDHistory(self, track_id):
        self.history.append(track_id)


class FakeSlider:
    def __init__(self):
        self._value = 0

    def setValue(self, value):
        self._value = value

    def value(self):
        return self._value


class FakeBrightnessContrastDialog:
    next_brightness = 0
    next_contrast = 0
    instances = []

    def __init__(self, image, callback, parent=None):
        self.slider_brightness = FakeSlider()
        self.slider_contrast = FakeSlider()
        self._callback = callback
        self._parent = parent
        self.exec_calls = 0
        self.applied_values = []
        type(self).instances.append(self)

    def exec_(self):
        self.exec_calls += 1
        self.slider_brightness.setValue(type(self).next_brightness)
        self.slider_contrast.setValue(type(self).next_contrast)

    def onNewValue(self, _value):
        self.applied_values.append(
            (self.slider_brightness.value(), self.slider_contrast.value())
        )
        self._callback(QtGui.QImage.fromData(self._parent.imageData))


@pytest.mark.gui
def test_MainWindow_openNextImg(qtbot):
    win = create_MainWindow_with_directory(qtbot)
    win.openNextImg()


@pytest.mark.gui
def test_MainWindow_openPrevImg(qtbot):
    win = create_MainWindow_with_directory(qtbot)
    win.openNextImg()


@pytest.mark.gui
def test_brightness_contrast_persists_across_video_frames(qtbot, monkeypatch):
    tmp_dir = tempfile.mkdtemp()
    try:
        image_files = _copy_test_image_sequence(tmp_dir)
        monkeypatch.setattr(
            labelme.app,
            "BrightnessContrastDialog",
            FakeBrightnessContrastDialog,
        )
        FakeBrightnessContrastDialog.instances.clear()
        FakeBrightnessContrastDialog.next_brightness = 65
        FakeBrightnessContrastDialog.next_contrast = 80

        config = labelme.config.get_default_config()
        win = labelme.app.MainWindow(config=config, filename=tmp_dir)
        qtbot.addWidget(win)
        _win_show_and_wait_imageData(qtbot, win)

        video_key = win._brightnessContrastKey()
        assert video_key == osp.normpath(osp.abspath(tmp_dir))

        win.brightnessContrast(None)

        assert win.brightnessContrast_values == {video_key: (65, 80)}

        win.openNextImg()

        assert win.filename == image_files[1]
        assert win._brightnessContrastKey() == video_key
        assert len(win.brightnessContrast_values) == 1
        assert FakeBrightnessContrastDialog.instances[-1].applied_values == [(65, 80)]
    finally:
        shutil.rmtree(tmp_dir)


@pytest.mark.gui
def test_brightness_contrast_does_not_carry_to_new_video_by_default(qtbot, monkeypatch):
    first_dir = tempfile.mkdtemp()
    second_dir = tempfile.mkdtemp()
    try:
        _copy_test_image_sequence(first_dir)
        _copy_test_image_sequence(second_dir)
        monkeypatch.setattr(
            labelme.app,
            "BrightnessContrastDialog",
            FakeBrightnessContrastDialog,
        )
        FakeBrightnessContrastDialog.instances.clear()
        FakeBrightnessContrastDialog.next_brightness = 55
        FakeBrightnessContrastDialog.next_contrast = 75

        config = labelme.config.get_default_config()
        win = labelme.app.MainWindow(config=config, filename=first_dir)
        qtbot.addWidget(win)
        _win_show_and_wait_imageData(qtbot, win)

        first_key = win._brightnessContrastKey()
        win.brightnessContrast(None)
        dialog_count = len(FakeBrightnessContrastDialog.instances)

        win.importDirImages(second_dir)

        second_key = win._brightnessContrastKey()
        assert second_key == osp.normpath(osp.abspath(second_dir))
        assert second_key != first_key
        assert len(FakeBrightnessContrastDialog.instances) == dialog_count
        assert win.brightnessContrast_values[first_key] == (55, 75)
        assert win.brightnessContrast_values[second_key] == (None, None)
    finally:
        shutil.rmtree(first_dir)
        shutil.rmtree(second_dir)


@pytest.mark.gui
def test_brightness_contrast_keep_prev_carries_between_videos(qtbot, monkeypatch):
    first_dir = tempfile.mkdtemp()
    second_dir = tempfile.mkdtemp()
    try:
        _copy_test_image_sequence(first_dir)
        _copy_test_image_sequence(second_dir)
        monkeypatch.setattr(
            labelme.app,
            "BrightnessContrastDialog",
            FakeBrightnessContrastDialog,
        )
        FakeBrightnessContrastDialog.instances.clear()
        FakeBrightnessContrastDialog.next_brightness = 45
        FakeBrightnessContrastDialog.next_contrast = 90

        config = labelme.config.get_default_config()
        config["keep_prev_brightness"] = True
        config["keep_prev_contrast"] = True
        win = labelme.app.MainWindow(config=config, filename=first_dir)
        qtbot.addWidget(win)
        _win_show_and_wait_imageData(qtbot, win)

        first_key = win._brightnessContrastKey()
        win.brightnessContrast(None)

        win.importDirImages(second_dir)

        second_key = win._brightnessContrastKey()
        assert second_key == osp.normpath(osp.abspath(second_dir))
        assert second_key != first_key
        assert win.brightnessContrast_values[first_key] == (45, 90)
        assert win.brightnessContrast_values[second_key] == (45, 90)
        assert FakeBrightnessContrastDialog.instances[-1].applied_values == [(45, 90)]
    finally:
        shutil.rmtree(first_dir)
        shutil.rmtree(second_dir)


@pytest.mark.gui
def test_pending_autosave_new_bbox_stays_on_current_frame_when_opening_next(qtbot):
    tmp_dir = tempfile.mkdtemp()
    try:
        image_files = _copy_test_image_sequence(tmp_dir)

        config = labelme.config.get_default_config()
        win = labelme.app.MainWindow(config=config, filename=tmp_dir)
        qtbot.addWidget(win)
        _win_show_and_wait_imageData(qtbot, win)
        assert win.filename == image_files[0]

        shape = Shape(
            label="person",
            group_id=1,
            track_id="1",
            shape_type="rectangle",
            flags={},
        )
        shape.addPoint(QtCore.QPointF(10, 10))
        shape.addPoint(QtCore.QPointF(20, 20))
        shape.close()
        win.loadShapes([shape])
        win.setDirty()

        assert win._save_timer.isActive()

        win.openNextImg()

        first_json = osp.splitext(image_files[0])[0] + ".json"
        second_json = osp.splitext(image_files[1])[0] + ".json"
        data = _read_json(first_json)

        assert data["imagePath"] == "000001.jpg"
        assert data["shapes"][0]["points"] == [[10.0, 10.0], [20.0, 20.0]]
        assert not osp.exists(second_json)
    finally:
        shutil.rmtree(tmp_dir)


@pytest.mark.gui
def test_frame_navigation_releases_stale_canvas_references(qtbot):
    tmp_dir = tempfile.mkdtemp()
    try:
        _copy_test_image_sequence(tmp_dir)

        config = labelme.config.get_default_config()
        win = labelme.app.MainWindow(config=config, filename=tmp_dir)
        qtbot.addWidget(win)
        _win_show_and_wait_imageData(qtbot, win)

        shape = Shape(
            label="person",
            group_id=1,
            track_id="1",
            shape_type="rectangle",
            flags={},
        )
        shape.addPoint(QtCore.QPointF(10, 10))
        shape.addPoint(QtCore.QPointF(20, 20))
        shape.close()
        win.loadShapes([shape])
        win.canvas.selectedShapes = [shape]
        win.canvas.selectedShapesCopy = [shape.copy()]
        win.canvas.current = shape
        win.canvas.hShape = shape
        win.canvas.prevhShape = shape
        win.canvas.visible = {shape: True}

        win.openNextImg()

        assert win.canvas.shapes == []
        assert win.canvas.selectedShapes == []
        assert win.canvas.selectedShapesCopy == []
        assert win.canvas.current is None
        assert win.canvas.hShape is None
        assert win.canvas.prevhShape is None
        assert win.canvas.visible == {}
    finally:
        shutil.rmtree(tmp_dir)


@pytest.mark.gui
def test_keep_prev_shapes_survive_frame_reset(qtbot):
    tmp_dir = tempfile.mkdtemp()
    try:
        _copy_test_image_sequence(tmp_dir)

        config = labelme.config.get_default_config()
        config["keep_prev"] = True
        win = labelme.app.MainWindow(config=config, filename=tmp_dir)
        qtbot.addWidget(win)
        _win_show_and_wait_imageData(qtbot, win)

        shape = Shape(
            label="person",
            group_id=1,
            track_id="1",
            shape_type="rectangle",
            flags={},
        )
        shape.addPoint(QtCore.QPointF(10, 10))
        shape.addPoint(QtCore.QPointF(20, 20))
        shape.close()
        win.loadShapes([shape])

        win.openNextImg()

        assert len(win.canvas.shapes) == 1
        assert win.canvas.shapes[0].label == "person"
        assert [(p.x(), p.y()) for p in win.canvas.shapes[0].points] == [
            (10.0, 10.0),
            (20.0, 20.0),
        ]
    finally:
        shutil.rmtree(tmp_dir)


@pytest.mark.gui
def test_keyboard_bbox_move_autosaves_before_frame_change(qtbot):
    tmp_dir = tempfile.mkdtemp()
    try:
        image_files = _copy_test_image_sequence(tmp_dir)
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

        config = labelme.config.get_default_config()
        win = labelme.app.MainWindow(config=config, filename=tmp_dir)
        qtbot.addWidget(win)
        _win_show_and_wait_imageData(qtbot, win)
        assert win.filename == image_files[0]

        shape_obj = win.canvas.shapes[0]
        win.canvas.selectedShapes = [shape_obj]
        win.canvas.prevPoint = QtCore.QPointF(shape_obj.points[0])
        win.canvas.calculateOffsets(win.canvas.prevPoint)
        win.canvas.moveByKeyboard(QtCore.QPointF(1.0, 0.0))

        assert win._save_timer.isActive()

        win.openNextImg()
        win.openPrevImg()

        data = _read_json(osp.splitext(image_files[0])[0] + ".json")
        assert data["shapes"][0]["points"] == [[11.0, 10.0], [21.0, 20.0]]
        assert [(p.x(), p.y()) for p in win.canvas.shapes[0].points] == [
            (11.0, 10.0),
            (21.0, 20.0),
        ]
    finally:
        shutil.rmtree(tmp_dir)


@pytest.mark.gui
def test_forward_tracking_actions_disable_on_final_frame(qtbot, tmp_path):
    _copy_test_image_sequence(str(tmp_path))
    win = labelme.app.MainWindow(filename=str(tmp_path))
    qtbot.addWidget(win)
    _win_show_and_wait_imageData(qtbot, win)

    assert win.actions.trackForward.isEnabled()
    assert win.actions.trackForwardBoTSORT.isEnabled()

    win.openNextImg()

    assert not win.actions.trackForward.isEnabled()
    assert not win.actions.trackForwardBoTSORT.isEnabled()


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
def test_canvas_point_prompt_emits_image_coordinates(qtbot):
    canvas = Canvas()
    qtbot.addWidget(canvas)
    pixmap = QtGui.QPixmap(100, 80)
    pixmap.fill(QtGui.QColor("black"))
    canvas.resize(100, 80)
    canvas.loadPixmap(pixmap)
    points = []
    canvas.pointPromptRequested.connect(points.append)

    assert canvas.armPointPrompt()
    event = QtGui.QMouseEvent(
        QtCore.QEvent.MouseButtonPress,
        QtCore.QPointF(10, 15),
        QtCore.Qt.LeftButton,
        QtCore.Qt.LeftButton,
        QtCore.Qt.NoModifier,
    )
    canvas.mousePressEvent(event)

    assert len(points) == 1
    assert points[0].x() == 10
    assert points[0].y() == 15


@pytest.mark.gui
def test_promptForNewShapeMetadata_uses_tracking_values_outside_normal_mode(qtbot):
    class RaisingLabelDialog:
        def __init__(self):
            self.edit = QtWidgets.QLineEdit()

        def popUp(self, text=None):
            raise AssertionError("tracking modes should bypass the label popup")

    config = labelme.config.get_default_config()
    config["display_label_popup"] = False
    win = labelme.app.MainWindow(config=config)
    qtbot.addWidget(win)

    item = win.uniqLabelList.createItemFromLabel("selected-label")
    win.uniqLabelList.addItem(item)
    win.uniqLabelList.setCurrentItem(item)
    item.setSelected(True)

    win.mode = "TRACK INTERPOLATION"
    win.label_INPO = "tracked-label"
    win.ID_INPO = "42"
    win.labelDialog = RaisingLabelDialog()

    metadata = win._promptForNewShapeMetadata()

    assert metadata == ("tracked-label", {}, None, "", "42")


@pytest.mark.gui
def test_hosted_sam2_point_prompt_adds_bbox_with_existing_popup(qtbot):
    img_file = osp.join(data_dir, "raw/2011_000003.jpg")
    config = labelme.config.get_default_config()
    config["hosted_sam2"]["url"] = "http://sam2.example"
    win = labelme.app.MainWindow(config=config, filename=img_file)
    qtbot.addWidget(win)
    _win_show_and_wait_imageData(qtbot, win)
    fake_client = FakeHostedSam2Client(bbox=[12, 14, 40, 45])
    win._hosted_sam2_client = fake_client
    win.labelDialog = FakeLabelDialog()
    win.IDDialog = FailingIDDialog()

    win.startHostedSam2PointPrompt()

    qtbot.waitUntil(lambda: len(fake_client.register_calls) == 1)
    assert win.canvas._point_prompt_armed
    win.canvas.pointPromptRequested.emit(QtCore.QPointF(20, 25))
    qtbot.waitUntil(lambda: len(fake_client.prompt_calls) == 1)
    qtbot.waitUntil(lambda: len(win.canvas.shapes) == 1)

    shape = win.canvas.shapes[0]
    assert shape.label == "person"
    assert shape.track_id == "1"
    assert [(p.x(), p.y()) for p in shape.points] == [(12.0, 14.0), (40.0, 45.0)]
    assert win.labelDialog.popups == 1
    assert win.IDDialog.history == ["1"]
    assert win._save_timer.isActive()
    win.setClean()


@pytest.mark.gui
def test_hosted_sam2_popup_cancel_creates_no_bbox(qtbot):
    img_file = osp.join(data_dir, "raw/2011_000003.jpg")
    config = labelme.config.get_default_config()
    config["hosted_sam2"]["url"] = "http://sam2.example"
    win = labelme.app.MainWindow(config=config, filename=img_file)
    qtbot.addWidget(win)
    _win_show_and_wait_imageData(qtbot, win)
    fake_client = FakeHostedSam2Client()
    win._hosted_sam2_client = fake_client
    win.labelDialog = FakeLabelDialog(label="")
    win.IDDialog = FailingIDDialog()

    win.startHostedSam2PointPrompt()
    qtbot.waitUntil(lambda: len(fake_client.register_calls) == 1)
    win.canvas.pointPromptRequested.emit(QtCore.QPointF(20, 25))
    qtbot.waitUntil(lambda: len(fake_client.prompt_calls) == 1)
    qtbot.waitUntil(lambda: win.labelDialog.popups == 1)

    assert win.canvas.shapes == []
    assert win.labelDialog.popups == 1


@pytest.mark.gui
def test_hosted_sam2_prompt_failure_creates_no_bbox(qtbot):
    img_file = osp.join(data_dir, "raw/2011_000003.jpg")
    config = labelme.config.get_default_config()
    config["hosted_sam2"]["url"] = "http://sam2.example"
    win = labelme.app.MainWindow(config=config, filename=img_file)
    qtbot.addWidget(win)
    _win_show_and_wait_imageData(qtbot, win)
    fake_client = FakeHostedSam2Client(fail_prompt=True)
    win._hosted_sam2_client = fake_client
    errors = []
    win.errorMessage = lambda title, message: errors.append((title, message))

    win.startHostedSam2PointPrompt()
    qtbot.waitUntil(lambda: len(fake_client.register_calls) == 1)
    win.canvas.pointPromptRequested.emit(QtCore.QPointF(20, 25))
    qtbot.waitUntil(lambda: bool(errors))

    assert errors == [("Hosted SAM2", "prompt failed")]
    assert win.canvas.shapes == []


@pytest.mark.gui
def test_hosted_sam2_reregisters_once_after_backend_cache_loss(qtbot):
    class RestartingClient(FakeHostedSam2Client):
        def register_image(self, image_data, client_frame_key=None):
            response = super().register_image(image_data, client_frame_key)
            response["image_id"] = "image-{}".format(len(self.register_calls))
            return response

        def point_prompt(self, image_id, x, y, label=1):
            self.prompt_calls.append((image_id, x, y, label))
            if len(self.prompt_calls) == 1:
                raise HostedSam2Error("Unknown image_id.", status_code=404)
            return {"bbox": list(self.bbox), "score": 0.9, "model": "fake"}

    img_file = osp.join(data_dir, "raw/2011_000003.jpg")
    config = labelme.config.get_default_config()
    config["hosted_sam2"]["url"] = "http://sam2.example"
    win = labelme.app.MainWindow(config=config, filename=img_file)
    qtbot.addWidget(win)
    _win_show_and_wait_imageData(qtbot, win)
    client = RestartingClient()
    win._hosted_sam2_client = client
    win.labelDialog = FakeLabelDialog()
    win.IDDialog = FailingIDDialog()

    win.startHostedSam2PointPrompt()
    qtbot.waitUntil(lambda: len(client.register_calls) == 1)
    win.canvas.pointPromptRequested.emit(QtCore.QPointF(20, 25))
    qtbot.waitUntil(lambda: len(win.canvas.shapes) == 1)

    assert len(client.register_calls) == 2
    assert [call[0] for call in client.prompt_calls] == ["image-1", "image-2"]
    win.setClean()


@pytest.mark.gui
def test_hosted_sam2_does_not_reregister_a_stale_frame(qtbot):
    config = labelme.config.get_default_config()
    config["hosted_sam2"]["url"] = "http://sam2.example"
    win = labelme.app.MainWindow(config=config)
    qtbot.addWidget(win)
    client = FakeHostedSam2Client()
    win._hosted_sam2_client = client
    win.image = QtGui.QImage(2, 2, QtGui.QImage.Format_RGB32)
    win.imageData = b"current-frame"
    win.imagePath = "/frames/current.png"
    stale_key = "stale-frame-key"
    win._hosted_sam2_image_cache[stale_key] = {"image_id": "stale"}
    win._hosted_sam2_request_context = {
        "kind": "point_prompt",
        "frame_key": stale_key,
        "point": (1.0, 1.0),
        "retry_count": 0,
        "image_data": b"stale-frame",
    }

    win._hostedSam2RequestFailed(HostedSam2Error("Unknown image_id.", status_code=404))

    assert client.register_calls == []
    assert stale_key not in win._hosted_sam2_image_cache


@pytest.mark.gui
def test_hosted_sam2_client_cache_is_access_ordered(qtbot):
    win = labelme.app.MainWindow(config=labelme.config.get_default_config())
    qtbot.addWidget(win)
    win._hosted_sam2_max_cached_frames = 2
    win._rememberHostedSam2Image("a", {"image_id": "a"})
    win._rememberHostedSam2Image("b", {"image_id": "b"})

    assert win._hostedSam2CachedImage("a") == {"image_id": "a"}
    win._rememberHostedSam2Image("c", {"image_id": "c"})

    assert list(win._hosted_sam2_image_cache) == ["a", "c"]


def test_load_image_file_returns_raw_jpeg_when_no_orientation():
    image_file = osp.join(data_dir, "raw/2011_000003.jpg")

    with open(image_file, "rb") as f:
        raw = f.read()

    assert LabelFile.load_image_file(image_file) == raw


@pytest.mark.gui
def test_canvas_load_pixmap_does_not_refresh_inactive_ai_model(qtbot):
    class FakeAiModel:
        name = "fake"

        def __init__(self):
            self.images = []

        def set_image(self, image):
            self.images.append(image)

    win = labelme.app.MainWindow(config=labelme.config.get_default_config())
    qtbot.addWidget(win)
    ai_model = FakeAiModel()
    win.canvas._ai_model = ai_model
    win.canvas.createMode = "polygon"

    pixmap = QtGui.QPixmap(10, 10)
    pixmap.fill(QtGui.QColor("black"))
    win.canvas.loadPixmap(pixmap)

    assert ai_model.images == []


@pytest.mark.gui
def test_toggle_draw_mode_releases_ai_model_when_leaving_ai_mode(qtbot):
    class FakeAiModel:
        name = "fake"

        def __init__(self):
            self.closed = False

        def close(self):
            self.closed = True

    win = labelme.app.MainWindow(config=labelme.config.get_default_config())
    qtbot.addWidget(win)
    ai_model = FakeAiModel()
    win.canvas._ai_model = ai_model
    win.canvas.createMode = "ai_polygon"

    win.toggleDrawMode(True)

    assert ai_model.closed
    assert win.canvas._ai_model is None


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

        before = _read_json(osp.splitext(image_file)[0] + ".json")

        config = labelme.config.get_default_config()
        win = labelme.app.MainWindow(config=config)
        qtbot.addWidget(win)
        win._imageListCache = [image_file, image_file_2]
        win.lastOpenDir = tmp_dir
        win.image = QtGui.QImage(10, 10, QtGui.QImage.Format_RGB32)

        monkeypatch.setattr(labelme.app, "DeletionDialog", RejectedDeletionDialog)

        win.DELETION()

        after = _read_json(osp.splitext(image_file)[0] + ".json")
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
            data = _read_json(osp.splitext(image_file)[0] + ".json")
            assert len(data["shapes"]) == 2
            assert data["shapes"][0]["label"] == "person"
            assert data["shapes"][0]["track_id"] == "9"
            assert data["shapes"][0]["group_id"] == 9
            assert data["shapes"][1]["track_id"] == "8"
            assert data["shapes"][1]["group_id"] == 8
    finally:
        shutil.rmtree(tmp_dir)


@pytest.mark.gui
def test_track_modification_swap_id_exchanges_existing_track(qtbot, monkeypatch):
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

        old_track = dict(
            label="person",
            group_id=7,
            track_id="7",
            points=[(10, 10), (20, 20)],
            shape_type="rectangle",
            flags={},
            description=None,
            mask=None,
        )
        existing_new_track = dict(
            label="person",
            group_id=9,
            track_id="9",
            points=[(30, 30), (40, 40)],
            shape_type="rectangle",
            flags={},
            description=None,
            mask=None,
        )
        for image_file in image_files:
            LabelFile().save(
                filename=osp.splitext(image_file)[0] + ".json",
                shapes=[old_track.copy(), existing_new_track.copy()],
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
            data = _read_json(osp.splitext(image_file)[0] + ".json")
            assert len(data["shapes"]) == 2
            assert data["shapes"][0]["track_id"] == "9"
            assert data["shapes"][0]["group_id"] == 9
            assert data["shapes"][1]["track_id"] == "7"
            assert data["shapes"][1]["group_id"] == 7
    finally:
        shutil.rmtree(tmp_dir)


@pytest.mark.gui
def test_track_modification_swap_id_handles_label_case_change(qtbot, monkeypatch):
    class AcceptedSwapIDDialog:
        def __init__(self, parent=None):
            self.start_frame_cell = QtWidgets.QLineEdit("1")
            self.end_frame_cell = QtWidgets.QLineEdit("2")
            self.ID_cell = QtWidgets.QLineEdit("11")
            self.label_cell = QtWidgets.QLineEdit("Person")
            self.new_ID_cell = QtWidgets.QLineEdit("12")
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

        shapes = [
            dict(
                label="Person",
                group_id=11,
                track_id="11",
                points=[(10, 10), (20, 20)],
                shape_type="rectangle",
                flags={},
                description=None,
                mask=None,
            ),
            dict(
                label="Person",
                group_id=12,
                track_id="12",
                points=[(30, 30), (40, 40)],
                shape_type="rectangle",
                flags={},
                description=None,
                mask=None,
            ),
        ]
        for index, image_file in enumerate(image_files):
            frame_shapes = [shape.copy() for shape in shapes]
            if index == 1:
                frame_shapes[0]["label"] = "person"
            LabelFile().save(
                filename=osp.splitext(image_file)[0] + ".json",
                shapes=frame_shapes,
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

        for index, image_file in enumerate(image_files):
            data = _read_json(osp.splitext(image_file)[0] + ".json")
            source, destination = data["shapes"]
            if index == 0:
                assert source["track_id"] == "12"
                assert source["group_id"] == 12
                assert destination["track_id"] == "11"
                assert destination["group_id"] == 11
            else:
                assert source["label"] == "person"
                assert source["track_id"] == "11"
                assert source["group_id"] == 11
                assert destination["track_id"] == "12"
                assert destination["group_id"] == 12
    finally:
        shutil.rmtree(tmp_dir)


@pytest.mark.gui
def test_track_modification_swap_id_keeps_different_case_tracks_separate(
    qtbot, monkeypatch
):
    class AcceptedSwapIDDialog:
        def __init__(self, parent=None):
            self.start_frame_cell = QtWidgets.QLineEdit("1")
            self.end_frame_cell = QtWidgets.QLineEdit("2")
            self.ID_cell = QtWidgets.QLineEdit("5")
            self.label_cell = QtWidgets.QLineEdit("Person")
            self.new_ID_cell = QtWidgets.QLineEdit("6")
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

        shapes = [
            dict(
                label=label,
                group_id=track_id,
                track_id=str(track_id),
                points=points,
                shape_type="rectangle",
                flags={},
                description=None,
                mask=None,
            )
            for label, track_id, points in [
                ("Person", 5, [(10, 10), (20, 20)]),
                ("Person", 6, [(30, 30), (40, 40)]),
                ("person", 5, [(50, 50), (60, 60)]),
                ("person", 6, [(70, 70), (80, 80)]),
            ]
        ]
        for image_file in image_files:
            LabelFile().save(
                filename=osp.splitext(image_file)[0] + ".json",
                shapes=[shape.copy() for shape in shapes],
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
            data = _read_json(osp.splitext(image_file)[0] + ".json")
            upper_source, upper_destination, lower_source, lower_destination = data[
                "shapes"
            ]
            assert upper_source["track_id"] == "6"
            assert upper_source["group_id"] == 6
            assert upper_destination["track_id"] == "5"
            assert upper_destination["group_id"] == 5
            assert lower_source["track_id"] == "5"
            assert lower_source["group_id"] == 5
            assert lower_destination["track_id"] == "6"
            assert lower_destination["group_id"] == 6
    finally:
        shutil.rmtree(tmp_dir)


@pytest.mark.gui
def test_track_modification_remove_box_with_new_id_does_not_delete(qtbot, monkeypatch):
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
        before = _read_json(first_json)

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

        after = _read_json(first_json)
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


@pytest.mark.gui
def test_autosave_marks_dirty_immediately_and_failed_save_blocks_navigation(
    qtbot, tmp_path, monkeypatch
):
    image_file = tmp_path / "frame.jpg"
    shutil.copy(osp.join(data_dir, "raw/2011_000003.jpg"), image_file)
    config = labelme.config.get_default_config()
    win = labelme.app.MainWindow(config=config, filename=str(image_file))
    qtbot.addWidget(win)
    _win_show_and_wait_imageData(qtbot, win)
    shape = Shape(label="person", track_id="1", shape_type="rectangle", flags={})
    shape.points = [QtCore.QPointF(1, 1), QtCore.QPointF(5, 5)]
    shape.point_labels = [1, 1]
    win.loadShapes([shape])

    win.setDirty()

    assert win.dirty
    assert win._save_timer.isActive()
    monkeypatch.setattr(win, "saveLabels", lambda _filename: False)
    monkeypatch.setattr(
        QtWidgets.QMessageBox,
        "question",
        lambda *_args, **_kwargs: QtWidgets.QMessageBox.Save,
    )

    assert not win.mayContinue()
    assert win.dirty
    win.setClean()


@pytest.mark.gui
def test_discard_after_autosave_failure_clears_pending_retry(
    qtbot, tmp_path, monkeypatch
):
    image_file = tmp_path / "frame.jpg"
    shutil.copy(osp.join(data_dir, "raw/2011_000003.jpg"), image_file)
    win = labelme.app.MainWindow(
        config=labelme.config.get_default_config(), filename=str(image_file)
    )
    qtbot.addWidget(win)
    _win_show_and_wait_imageData(qtbot, win)
    save_attempts = []
    monkeypatch.setattr(
        win, "saveLabels", lambda filename: save_attempts.append(filename) and False
    )
    monkeypatch.setattr(
        QtWidgets.QMessageBox,
        "question",
        lambda *_args, **_kwargs: QtWidgets.QMessageBox.Discard,
    )

    win.setDirty()

    assert win.mayContinue()
    assert len(save_attempts) == 1
    assert win._pending_auto_save_target is None
    assert win._flushPendingAutoSave()
    assert len(save_attempts) == 1
    win.setClean()


@pytest.mark.gui
def test_filename_search_only_filters_rows_and_keeps_displayed_frame(qtbot, tmp_path):
    first = QtGui.QImage(10, 5, QtGui.QImage.Format_RGB32)
    second = QtGui.QImage(20, 5, QtGui.QImage.Format_RGB32)
    first.save(str(tmp_path / "a.png"))
    second.save(str(tmp_path / "b.png"))
    win = labelme.app.MainWindow(filename=str(tmp_path))
    qtbot.addWidget(win)
    _win_show_and_wait_imageData(qtbot, win)
    win.openNextImg()
    assert osp.basename(win.filename) == "b.png"
    assert win.image.width() == 20

    win.fileSearch.setText("a\\.png$")

    assert osp.basename(win.filename) == "b.png"
    assert osp.basename(win.imagePath) == "b.png"
    assert win.image.width() == 20
    assert not win.fileListWidget.item(0).isHidden()
    assert win.fileListWidget.item(1).isHidden()


@pytest.mark.gui
def test_failed_file_load_keeps_previous_frame_state(qtbot, tmp_path):
    image_file = tmp_path / "frame.jpg"
    shutil.copy(osp.join(data_dir, "raw/2011_000003.jpg"), image_file)
    invalid_label = tmp_path / "broken.json"
    invalid_label.write_text("{not valid json", encoding="utf-8")
    win = labelme.app.MainWindow(
        config=labelme.config.get_default_config(), filename=str(image_file)
    )
    qtbot.addWidget(win)
    _win_show_and_wait_imageData(qtbot, win)
    original_filename = win.filename
    original_image_path = win.imagePath
    original_width = win.image.width()
    errors = []
    win.errorMessage = lambda title, message: errors.append((title, message))

    assert not win.loadFile(str(invalid_label))

    assert errors
    assert win.filename == original_filename
    assert win.imagePath == original_image_path
    assert win.image.width() == original_width


@pytest.mark.gui
def test_output_directory_saves_duplicate_basenames_in_nested_paths(qtbot, tmp_path):
    image_root = tmp_path / "images"
    first = image_root / "camera1" / "frame.jpg"
    second = image_root / "camera2" / "frame.jpg"
    first.parent.mkdir(parents=True)
    second.parent.mkdir(parents=True)
    source = osp.join(data_dir, "raw/2011_000003.jpg")
    shutil.copy(source, first)
    shutil.copy(source, second)
    output = tmp_path / "annotations"
    config = labelme.config.get_default_config()
    config["auto_save"] = False
    win = labelme.app.MainWindow(config=config, output_dir=str(output))
    qtbot.addWidget(win)
    win.importDirImages(str(image_root), load=False)

    for image_path, label in [(first, "first"), (second, "second")]:
        win.loadFile(str(image_path))
        qtbot.waitUntil(lambda path=str(image_path): win.filename == path)
        shape = Shape(label=label, track_id=0, shape_type="rectangle", flags={})
        shape.points = [QtCore.QPointF(1, 1), QtCore.QPointF(5, 5)]
        shape.point_labels = [1, 1]
        win.loadShapes([shape])
        win.setDirty()
        assert win.saveFile()

    first_label = output / "camera1" / "frame.json"
    second_label = output / "camera2" / "frame.json"
    assert LabelFile(str(first_label)).shapes[0]["label"] == "first"
    assert LabelFile(str(second_label)).shapes[0]["label"] == "second"


@pytest.mark.gui
def test_normal_gui_save_preserves_unknown_metadata_and_track_zero(qtbot, tmp_path):
    image_file = tmp_path / "frame.jpg"
    shutil.copy(osp.join(data_dir, "raw/2011_000003.jpg"), image_file)
    label_file = tmp_path / "frame.json"
    label_file.write_text(
        json.dumps(
            {
                "version": "legacy",
                "flags": {"reviewed": True},
                "shapes": [
                    {
                        "label": "person",
                        "points": [[1, 1], [5, 5]],
                        "group_id": 7,
                        "track_id": 0,
                        "shape_type": "rectangle",
                        "flags": {"occluded": True},
                        "description": "kept",
                        "mask": None,
                        "confidence": 0.8,
                        "other_data": "user-value",
                    }
                ],
                "imagePath": image_file.name,
                "imageData": None,
                "imageHeight": 375,
                "imageWidth": 500,
                "review": {"owner": "qa"},
            }
        ),
        encoding="utf-8",
    )
    win = labelme.app.MainWindow(
        config=labelme.config.get_default_config(), filename=str(label_file)
    )
    qtbot.addWidget(win)
    _win_show_and_wait_imageData(qtbot, win)

    assert win.saveFile()

    saved = json.loads(label_file.read_text(encoding="utf-8"))
    assert saved["review"] == {"owner": "qa"}
    assert saved["flags"] == {"reviewed": True}
    assert saved["shapes"][0]["track_id"] == 0
    assert saved["shapes"][0]["confidence"] == 0.8
    assert saved["shapes"][0]["other_data"] == "user-value"


@pytest.mark.gui
def test_editing_only_a_label_does_not_replace_track_zero_with_group_id(qtbot):
    class EditLabelDialog:
        def popUp(self, **_kwargs):
            return "vehicle", {}, 7, "edited"

    win = labelme.app.MainWindow(config=labelme.config.get_default_config())
    qtbot.addWidget(win)
    shape = Shape(
        label="person",
        group_id=7,
        track_id=0,
        shape_type="rectangle",
        flags={},
    )
    shape.points = [QtCore.QPointF(1, 1), QtCore.QPointF(5, 5)]
    shape.point_labels = [1, 1]
    win.loadShapes([shape])
    win.labelDialog = EditLabelDialog()

    win.editLabel(next(iter(win.labelList)))

    assert shape.label == "vehicle"
    assert shape.group_id == 7
    assert shape.track_id == 0
    win.setClean()


@pytest.mark.gui
def test_configured_save_options_are_not_overridden_and_actions_update_config(qtbot):
    config = labelme.config.get_default_config()
    config["auto_save"] = False
    config["store_data"] = True
    win = labelme.app.MainWindow(config=config)
    qtbot.addWidget(win)

    assert not win.actions.saveAuto.isChecked()
    assert win.actions.saveWithImageData.isChecked()

    win.actions.saveAuto.trigger()
    win.actions.saveWithImageData.trigger()

    assert win._config["auto_save"] is True
    assert win._config["store_data"] is False


@pytest.mark.gui
def test_fresh_window_adds_both_annotation_docks(qtbot):
    win = labelme.app.MainWindow(config=labelme.config.get_default_config())
    qtbot.addWidget(win)

    assert win.dockWidgetArea(win.shape_dock) != QtCore.Qt.NoDockWidgetArea
    assert win.dockWidgetArea(win.id_dock) != QtCore.Qt.NoDockWidgetArea


@pytest.mark.gui
def test_save_as_becomes_authoritative_for_later_saves(qtbot, tmp_path, monkeypatch):
    image_file = tmp_path / "frame.jpg"
    shutil.copy(osp.join(data_dir, "raw/2011_000003.jpg"), image_file)
    chosen_label_file = tmp_path / "chosen-name.json"
    canonical_label_file = tmp_path / "frame.json"
    config = labelme.config.get_default_config()
    config["auto_save"] = False
    win = labelme.app.MainWindow(config=config, filename=str(image_file))
    qtbot.addWidget(win)
    _win_show_and_wait_imageData(qtbot, win)
    monkeypatch.setattr(win, "saveFileDialog", lambda: str(chosen_label_file))

    assert win.saveFileAs()
    assert win.getLabelFile() == str(chosen_label_file)
    assert win.hasLabelFile()
    assert win.saveFile()
    assert chosen_label_file.is_file()
    assert not canonical_label_file.exists()
    assert win.loadFile(str(image_file))
    assert win.getLabelFile() == str(chosen_label_file)


@pytest.mark.gui
def test_normal_save_preserves_existing_embedded_image_data(qtbot, tmp_path):
    source_image = osp.join(data_dir, "raw/2011_000003.jpg")
    with open(source_image, "rb") as handle:
        image_data = handle.read()
    label_file = tmp_path / "self-contained.json"
    LabelFile().save(
        filename=str(label_file),
        shapes=[],
        imagePath="missing-image.jpg",
        imageData=image_data,
        imageHeight=375,
        imageWidth=500,
        flags={},
    )
    config = labelme.config.get_default_config()
    config["auto_save"] = False
    config["store_data"] = False
    win = labelme.app.MainWindow(config=config, filename=str(label_file))
    qtbot.addWidget(win)
    _win_show_and_wait_imageData(qtbot, win)

    assert win.labelFile.imageDataEmbedded
    assert win.saveFile()
    assert _read_json(label_file)["imageData"] is not None


def _save_output_dir_annotation(path, image_path, label):
    LabelFile().save(
        filename=str(path),
        shapes=[
            {
                "label": label,
                "points": [[10, 10], [20, 20]],
                "group_id": 1,
                "track_id": 1,
                "shape_type": "rectangle",
                "flags": {},
                "description": "",
                "mask": None,
            }
        ],
        imagePath=osp.relpath(image_path, path.parent),
        imageData=None,
        imageHeight=375,
        imageWidth=500,
        flags={},
    )


@pytest.mark.gui
def test_changing_output_dir_loads_the_new_directory_annotation(
    qtbot, tmp_path, monkeypatch
):
    image_dir = tmp_path / "images"
    old_output = tmp_path / "old-labels"
    new_output = tmp_path / "new-labels"
    image_dir.mkdir()
    old_output.mkdir()
    new_output.mkdir()
    image_file = image_dir / "frame.jpg"
    shutil.copy(osp.join(data_dir, "raw/2011_000003.jpg"), image_file)
    _save_output_dir_annotation(old_output / "frame.json", image_file, "old")
    _save_output_dir_annotation(new_output / "frame.json", image_file, "new")
    config = labelme.config.get_default_config()
    config["auto_save"] = False
    win = labelme.app.MainWindow(
        config=config,
        filename=str(image_dir),
        output_dir=str(old_output),
    )
    qtbot.addWidget(win)
    _win_show_and_wait_imageData(qtbot, win)
    assert win.canvas.shapes[0].label == "old"
    monkeypatch.setattr(
        QtWidgets.QFileDialog,
        "getExistingDirectory",
        lambda *_args, **_kwargs: str(new_output),
    )

    win.changeOutputDirDialog()

    assert win.output_dir == str(new_output)
    assert win.canvas.shapes[0].label == "new"
    assert win.labelFile.filename == str(new_output / "frame.json")


@pytest.mark.gui
def test_changing_output_dir_rolls_back_when_new_annotation_is_invalid(
    qtbot, tmp_path, monkeypatch
):
    image_dir = tmp_path / "images"
    old_output = tmp_path / "old-labels"
    bad_output = tmp_path / "bad-labels"
    image_dir.mkdir()
    old_output.mkdir()
    bad_output.mkdir()
    image_file = image_dir / "frame.jpg"
    shutil.copy(osp.join(data_dir, "raw/2011_000003.jpg"), image_file)
    _save_output_dir_annotation(old_output / "frame.json", image_file, "old")
    (bad_output / "frame.json").write_text("{bad json", encoding="utf-8")
    config = labelme.config.get_default_config()
    config["auto_save"] = False
    win = labelme.app.MainWindow(
        config=config,
        filename=str(image_dir),
        output_dir=str(old_output),
    )
    qtbot.addWidget(win)
    _win_show_and_wait_imageData(qtbot, win)
    errors = []
    win.errorMessage = lambda title, message: errors.append((title, message))
    monkeypatch.setattr(
        QtWidgets.QFileDialog,
        "getExistingDirectory",
        lambda *_args, **_kwargs: str(bad_output),
    )

    win.changeOutputDirDialog()

    assert errors
    assert win.output_dir == str(old_output)
    assert win.canvas.shapes[0].label == "old"
    assert win.labelFile.filename == str(old_output / "frame.json")


@pytest.mark.gui
def test_canonical_save_retires_legacy_flat_annotation(qtbot, tmp_path):
    image_root = tmp_path / "images"
    image_file = image_root / "camera1" / "frame.jpg"
    output = tmp_path / "annotations"
    image_file.parent.mkdir(parents=True)
    output.mkdir()
    shutil.copy(osp.join(data_dir, "raw/2011_000003.jpg"), image_file)
    legacy = output / "frame.json"
    canonical = output / "camera1" / "frame.json"
    _save_output_dir_annotation(legacy, image_file, "legacy")
    config = labelme.config.get_default_config()
    config["auto_save"] = False
    win = labelme.app.MainWindow(
        config=config,
        filename=str(image_root),
        output_dir=str(output),
    )
    qtbot.addWidget(win)
    _win_show_and_wait_imageData(qtbot, win)

    assert win.canvas.shapes[0].label == "legacy"
    assert win.saveFile()
    assert canonical.is_file()
    assert not legacy.exists()

    _save_output_dir_annotation(legacy, image_file, "stale")
    assert win.saveFile()
    assert not legacy.exists()

    canonical.unlink()
    assert win.loadFile(str(image_file))
    assert win.canvas.shapes == []


@pytest.mark.gui
@pytest.mark.parametrize(
    "shape_override",
    [
        {"shape_type": "unsupported"},
        {"points": [["bad", 1], [2, 3]]},
        {"flags": None},
    ],
)
def test_bad_shape_data_does_not_destroy_displayed_frame(
    qtbot, tmp_path, shape_override
):
    first = tmp_path / "first.jpg"
    second = tmp_path / "second.jpg"
    source = osp.join(data_dir, "raw/2011_000003.jpg")
    shutil.copy(source, first)
    shutil.copy(source, second)
    shape = {
        "label": "person",
        "points": [[1, 1], [5, 5]],
        "group_id": 1,
        "track_id": 1,
        "shape_type": "rectangle",
        "flags": {},
        "description": "",
        "mask": None,
    }
    shape.update(shape_override)
    (tmp_path / "second.json").write_text(
        json.dumps(
            {
                "version": "test",
                "flags": {},
                "shapes": [shape],
                "imagePath": second.name,
                "imageData": None,
                "imageHeight": 375,
                "imageWidth": 500,
            }
        ),
        encoding="utf-8",
    )
    win = labelme.app.MainWindow(
        config=labelme.config.get_default_config(), filename=str(first)
    )
    qtbot.addWidget(win)
    _win_show_and_wait_imageData(qtbot, win)
    original_data = win.imageData
    errors = []
    win.errorMessage = lambda title, message: errors.append((title, message))

    assert not win.loadFile(str(second))

    assert win.filename == str(first)
    assert win.imageData == original_data
    assert errors


@pytest.mark.gui
def test_forward_tracking_rejects_matching_polygon_without_mutating_file(
    qtbot, tmp_path
):
    image_file = tmp_path / "frame.jpg"
    shutil.copy(osp.join(data_dir, "raw/2011_000003.jpg"), image_file)
    label_file = tmp_path / "frame.json"
    polygon = {
        "label": "person",
        "points": [[1, 1], [5, 1], [5, 5]],
        "group_id": 1,
        "track_id": 1,
        "shape_type": "polygon",
        "flags": {},
        "description": "",
        "mask": None,
    }
    LabelFile().save(
        filename=str(label_file),
        shapes=[polygon],
        imagePath=image_file.name,
        imageData=None,
        imageHeight=375,
        imageWidth=500,
        flags={},
    )
    before = label_file.read_bytes()
    win = labelme.app.MainWindow(config=labelme.config.get_default_config())
    qtbot.addWidget(win)
    win.lastOpenDir = str(tmp_path)
    win._imageListCache = [str(image_file)]

    with pytest.raises(ValueError, match="non-rectangle"):
        win._trackResultRequest(
            str(image_file), "person", 1, 1, [[2, 2], [6, 6]], (375, 500)
        )

    assert label_file.read_bytes() == before


@pytest.mark.gui
def test_interpolation_mode_is_not_committed_when_first_frame_fails_to_load(
    qtbot, monkeypatch
):
    class AcceptedInterpolationDialog:
        def __init__(self, *args, **kwargs):
            pass

        def exec_(self):
            return QtWidgets.QDialog.Accepted

        def options(self):
            return type(
                "Options",
                (),
                {
                    "start_frame": 1,
                    "end_frame": 2,
                    "interval": 1,
                    "track_id": "1",
                    "label": "person",
                },
            )()

    win = labelme.app.MainWindow(config=labelme.config.get_default_config())
    qtbot.addWidget(win)
    win._imageListCache = ["/frames/1.jpg", "/frames/2.jpg"]
    win.filename = "/frames/1.jpg"
    win.mode = "NORMAL"
    monkeypatch.setattr(labelme.app, "InterpolationDialog", AcceptedInterpolationDialog)
    monkeypatch.setattr(win, "_ensureSavedForWorkflow", lambda _title: True)
    monkeypatch.setattr(win, "loadFile", lambda _filename: False)

    win.INTERPOLATION()

    assert win.mode == "NORMAL"
    assert win.INTERPOLATION_list == []
    assert win.INTERPOLATION_indices == []


def _rectangle(label="person", track_id=1, x1=1, y1=1, x2=5, y2=5):
    shape = Shape(
        label=label,
        group_id=track_id,
        track_id=track_id,
        shape_type="rectangle",
        flags={},
    )
    shape.points = [QtCore.QPointF(x1, y1), QtCore.QPointF(x2, y2)]
    shape.point_labels = [1, 1]
    return shape


@pytest.mark.gui
def test_label_and_id_edits_create_undo_snapshots(qtbot):
    class EditLabelDialog:
        def popUp(self, **_kwargs):
            return "vehicle", {}, 1, ""

        def addLabelHistory(self, _label):
            pass

    class EditIDDialog:
        def popUp(self, **_kwargs):
            return "2"

        def addIDHistory(self, _track_id):
            pass

    win = labelme.app.MainWindow(config=labelme.config.get_default_config())
    qtbot.addWidget(win)
    win.loadShapes([_rectangle()])
    win.labelDialog = EditLabelDialog()
    win.editLabel(next(iter(win.labelList)))
    assert win.canvas.shapes[0].label == "vehicle"
    win.undoShapeEdit()
    assert win.canvas.shapes[0].label == "person"

    win.IDDialog = EditIDDialog()
    win.editID(next(iter(win.IDList)))
    assert win.canvas.shapes[0].track_id == "2"
    win.undoShapeEdit()
    assert win.canvas.shapes[0].track_id == 1
    win.setClean()


@pytest.mark.gui
def test_multi_shape_ai_refinement_creates_one_undo_snapshot(qtbot, monkeypatch):
    class FakeModel:
        def predict_mask_from_box(self, _box):
            mask = np.zeros((12, 12), dtype=bool)
            mask[3:8, 4:9] = True
            return mask

    first = _rectangle(track_id=1, x1=1, y1=1, x2=5, y2=5)
    second = _rectangle(track_id=2, x1=2, y1=2, x2=6, y2=6)
    win = labelme.app.MainWindow(config=labelme.config.get_default_config())
    qtbot.addWidget(win)
    win.loadShapes([first, second])
    win.canvas.selectedShapes = [first, second]

    def initialize(_name):
        win.canvas._ai_model = FakeModel()

    monkeypatch.setattr(win.canvas, "initializeAiModel", initialize)

    win.refineBboxAI()

    assert len(win.canvas.shapesBackups) == 2
    assert win.canvas.shapes[0].points[0] == QtCore.QPointF(4, 3)
    assert win.canvas.shapes[1].points[0] == QtCore.QPointF(4, 3)
    win.undoShapeEdit()
    assert win.canvas.shapes[0].points[0] == QtCore.QPointF(1, 1)
    assert win.canvas.shapes[1].points[0] == QtCore.QPointF(2, 2)
    win.setClean()


@pytest.mark.gui
def test_multi_shape_ai_refinement_rolls_back_inference_failure(qtbot, monkeypatch):
    class FailingModel:
        def __init__(self):
            self.calls = 0

        def predict_mask_from_box(self, _box):
            self.calls += 1
            if self.calls == 2:
                raise RuntimeError("inference failed")
            return np.ones((12, 12), dtype=bool)

    first = _rectangle(track_id=1, x1=1, y1=1, x2=5, y2=5)
    second = _rectangle(track_id=2, x1=2, y1=2, x2=6, y2=6)
    win = labelme.app.MainWindow(config=labelme.config.get_default_config())
    qtbot.addWidget(win)
    win.loadShapes([first, second])
    win.canvas.selectedShapes = [first, second]
    monkeypatch.setattr(
        win.canvas,
        "initializeAiModel",
        lambda _name: setattr(win.canvas, "_ai_model", FailingModel()),
    )
    errors = []
    win.errorMessage = lambda title, message: errors.append((title, message))

    win.refineBboxAI()

    assert errors == [("Refine Bbox (AI)", "inference failed")]
    assert len(win.canvas.shapesBackups) == 1
    assert win.canvas.shapes[0].points[0] == QtCore.QPointF(1, 1)
    assert win.canvas.shapes[1].points[0] == QtCore.QPointF(2, 2)


@pytest.mark.gui
def test_cancelled_deferred_close_does_not_poison_future_workers(qtbot, monkeypatch):
    class Event:
        accepted = False
        ignored = False

        def accept(self):
            self.accepted = True

        def ignore(self):
            self.ignored = True

    win = labelme.app.MainWindow(config=labelme.config.get_default_config())
    qtbot.addWidget(win)
    win._close_after_tracking = True
    win._close_after_hosted_request = True
    monkeypatch.setattr(win, "mayContinue", lambda: False)
    event = Event()

    win.closeEvent(event)

    assert event.ignored
    assert not event.accepted
    assert not win._close_after_tracking
    assert not win._close_after_hosted_request
