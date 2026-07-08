import pytest
from qtpy import QtCore
from qtpy import QtGui

from labelme.shape import Shape
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


def test_rectangle_side_resize_preserves_opposite_side_and_axis():
    shape = Shape(shape_type="rectangle")
    shape.points = [QtCore.QPointF(10, 20), QtCore.QPointF(50, 80)]
    shape.point_labels = [1, 1]

    left_edge = shape.nearestRectangleEdge(QtCore.QPointF(10, 50), 1)
    shape.moveRectangleEdgeTo(left_edge, QtCore.QPointF(5, 60))

    assert shape.points[0] == QtCore.QPointF(5, 20)
    assert shape.points[1] == QtCore.QPointF(50, 80)

    bottom_edge = shape.nearestRectangleEdge(QtCore.QPointF(25, 80), 1)
    shape.moveRectangleEdgeTo(bottom_edge, QtCore.QPointF(25, 95))

    assert shape.points[0] == QtCore.QPointF(5, 20)
    assert shape.points[1] == QtCore.QPointF(50, 95)


def test_rectangle_side_hit_testing_handles_inverted_points():
    shape = Shape(shape_type="rectangle")
    shape.points = [QtCore.QPointF(50, 80), QtCore.QPointF(10, 20)]
    shape.point_labels = [1, 1]

    assert shape.nearestRectangleEdge(QtCore.QPointF(10, 50), 1) == ("left", 1)
    assert shape.nearestRectangleEdge(QtCore.QPointF(50, 50), 1) == ("right", 0)
    assert shape.nearestRectangleEdge(QtCore.QPointF(30, 20), 1) == ("top", 1)
    assert shape.nearestRectangleEdge(QtCore.QPointF(30, 80), 1) == ("bottom", 0)
    assert shape.nearestRectangleEdge(QtCore.QPointF(30, 101), 1) is None


@pytest.mark.gui
def test_canvas_rectangle_side_resize_clamps_to_pixmap(qtbot):
    canvas = Canvas()
    qtbot.addWidget(canvas)
    pixmap = QtGui.QPixmap(100, 100)
    pixmap.fill(QtGui.QColor("white"))
    canvas.loadPixmap(pixmap)

    shape = Shape(shape_type="rectangle")
    shape.points = [QtCore.QPointF(10, 20), QtCore.QPointF(50, 80)]
    shape.point_labels = [1, 1]
    canvas.hShape = shape
    canvas.hResizeEdge = ("left", 0)

    canvas.boundedResizeRectangleEdge(QtCore.QPointF(-25, 60))

    assert shape.points[0] == QtCore.QPointF(0, 20)
    assert shape.points[1] == QtCore.QPointF(50, 80)
