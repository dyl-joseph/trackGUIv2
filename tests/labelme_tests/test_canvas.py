import numpy as np
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


def test_mask_hit_testing_rejects_points_outside_mask_bounds():
    shape = Shape(shape_type="mask", mask=np.ones((2, 2), dtype=bool))
    shape.points = [QtCore.QPointF(10, 10), QtCore.QPointF(12, 12)]

    assert shape.containsPoint(QtCore.QPointF(10, 10))
    assert not shape.containsPoint(QtCore.QPointF(0, 0))
    assert not shape.containsPoint(QtCore.QPointF(9.6, 10))
    assert not shape.containsPoint(QtCore.QPointF(12, 12))


def test_linestrip_nearest_edge_does_not_wrap_last_point_to_first():
    shape = Shape(shape_type="linestrip")
    shape.points = [
        QtCore.QPointF(0, 0),
        QtCore.QPointF(10, 0),
        QtCore.QPointF(10, 10),
    ]

    assert shape.nearestEdge(QtCore.QPointF(5, 5), 0.5) is None
    assert shape.nearestEdge(QtCore.QPointF(10, 5), 0.5) == 2


def test_restoring_ai_prompt_restores_original_mask_state():
    shape = Shape(shape_type="points", mask=None)
    shape.points = [QtCore.QPointF(1, 1)]
    shape.point_labels = [1]
    shape.setShapeRefined(
        "mask",
        [QtCore.QPointF(0, 0), QtCore.QPointF(2, 2)],
        [1, 1],
        mask=np.ones((2, 2), dtype=bool),
    )

    shape.restoreShapeRaw()

    assert shape.shape_type == "points"
    assert shape.mask is None
    assert shape.points == [QtCore.QPointF(1, 1)]


@pytest.mark.gui
def test_ai_preview_is_debounced_and_cached_outside_paint(qtbot):
    class FakeModel:
        def __init__(self):
            self.calls = 0

        def predict_polygon_from_points(self, points, point_labels):
            self.calls += 1
            return [[0, 0], [2, 0], [2, 2]]

    canvas = Canvas()
    qtbot.addWidget(canvas)
    canvas._ai_model = FakeModel()
    points = [QtCore.QPointF(1, 1)]

    assert canvas._requestAiPreview("polygon", points, [1]) is None
    assert canvas._ai_model.calls == 0
    qtbot.waitUntil(lambda: canvas._ai_model.calls == 1, timeout=500)

    assert canvas._requestAiPreview("polygon", points, [1]) == [
        [0, 0],
        [2, 0],
        [2, 2],
    ]
    assert canvas._ai_model.calls == 1


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
