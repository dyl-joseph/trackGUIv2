import pytest
from qtpy import QtCore
from qtpy import QtGui

from labelme.widgets import Canvas


@pytest.mark.gui
def test_canvas_crosshair_spans_scaled_pixmap(qtbot):
    canvas = Canvas()
    qtbot.addWidget(canvas)
    pixmap = QtGui.QPixmap(200, 100)
    pixmap.fill(QtGui.QColor("white"))
    canvas.scale = 0.5
    canvas.resize(100, 50)
    canvas.loadPixmap(pixmap)
    canvas.createMode = "rectangle"
    canvas.setEditing(False)
    canvas.prevMovePoint = QtCore.QPointF(150, 50)

    rendered = QtGui.QImage(canvas.size(), QtGui.QImage.Format_RGB32)
    rendered.fill(QtGui.QColor("white"))
    painter = QtGui.QPainter(rendered)
    canvas.render(painter)
    painter.end()

    assert rendered.pixelColor(90, 25) != QtGui.QColor("white")
    assert rendered.pixelColor(75, 45) != QtGui.QColor("white")
