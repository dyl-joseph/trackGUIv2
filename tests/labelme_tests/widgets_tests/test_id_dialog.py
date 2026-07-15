import pytest
from qtpy import QtCore
from qtpy import QtGui

from labelme.widgets import IDDialog


@pytest.mark.gui
def test_id_dialog_stays_inside_small_cursor_screen(qtbot, monkeypatch):
    class FakeScreen:
        def availableGeometry(self):
            return QtCore.QRect(0, 0, 200, 200)

    dialog = IDDialog(ids=["1"])
    qtbot.addWidget(dialog)
    monkeypatch.setattr(QtGui.QCursor, "pos", lambda: QtCore.QPoint(190, 190))
    monkeypatch.setattr(QtGui.QGuiApplication, "screenAt", lambda _pos: FakeScreen())
    monkeypatch.setattr(QtGui.QGuiApplication, "primaryScreen", lambda: FakeScreen())
    monkeypatch.setattr(dialog, "exec_", lambda: 0)

    dialog.popUp("1")

    frame = dialog.frameGeometry()
    available = FakeScreen().availableGeometry()
    assert frame.left() >= available.left()
    assert frame.top() >= available.top()
    assert frame.right() <= available.right()
    assert frame.bottom() <= available.bottom()


@pytest.mark.gui
def test_id_dialog_accepts_falsy_numeric_track_id(qtbot, monkeypatch):
    dialog = IDDialog(ids=[0, 1])
    qtbot.addWidget(dialog)
    monkeypatch.setattr(dialog, "exec_", lambda: 0)

    assert dialog.popUp(0, move=False) is None
    assert dialog.edit.text() == "0"
    assert dialog.edit.selectedText() == "0"
    assert dialog.IDList.findItems("0", QtCore.Qt.MatchExactly)
