import os.path as osp
from math import sqrt

from qtpy import QtCore
from qtpy import QtGui
from qtpy import QtWidgets

here = osp.dirname(osp.abspath(__file__))


def newIcon(icon):
    icons_dir = osp.join(here, "../icons")
    return QtGui.QIcon(osp.join(":/", icons_dir, "%s.png" % icon))


def newButton(text, icon=None, slot=None):
    b = QtWidgets.QPushButton(text)
    if icon is not None:
        b.setIcon(newIcon(icon))
    if slot is not None:
        b.clicked.connect(slot)
    return b


def newAction(
    parent,
    text,
    slot=None,
    shortcut=None,
    icon=None,
    tip=None,
    checkable=False,
    enabled=True,
    checked=False,
):
    """Create a new action and assign callbacks, shortcuts, etc."""
    a = QtWidgets.QAction(text, parent)
    if icon is not None:
        a.setIconText(text.replace(" ", "\n"))
        a.setIcon(newIcon(icon))
    if shortcut is not None:
        if isinstance(shortcut, (list, tuple)):
            a.setShortcuts(shortcut)
        else:
            a.setShortcut(shortcut)
    if tip is not None:
        a.setToolTip(tip)
        a.setStatusTip(tip)
    if slot is not None:
        a.triggered.connect(slot)
    if checkable:
        a.setCheckable(True)
    a.setEnabled(enabled)
    a.setChecked(checked)
    return a


def addActions(widget, actions):
    for action in actions:
        if action is None:
            widget.addSeparator()
        elif isinstance(action, QtWidgets.QMenu):
            widget.addMenu(action)
        else:
            widget.addAction(action)


def labelValidator():
    return QtGui.QRegExpValidator(QtCore.QRegExp(r"^[^ \t].+"), None)


class struct(object):
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


def distance(p):
    return sqrt(p.x() * p.x() + p.y() * p.y())


def distancetoline(point, line):
    p1, p2 = line
    p1x, p1y = p1.x(), p1.y()
    p2x, p2y = p2.x(), p2.y()
    p3x, p3y = point.x(), point.y()
    dx, dy = p2x - p1x, p2y - p1y
    if dx == 0 and dy == 0:
        return sqrt((p3x - p1x) ** 2 + (p3y - p1y) ** 2)
    if (p3x - p1x) * dx + (p3y - p1y) * dy < 0:
        return sqrt((p3x - p1x) ** 2 + (p3y - p1y) ** 2)
    if (p3x - p2x) * (-dx) + (p3y - p2y) * (-dy) < 0:
        return sqrt((p3x - p2x) ** 2 + (p3y - p2y) ** 2)
    return abs(dx * (p1y - p3y) - dy * (p1x - p3x)) / sqrt(dx * dx + dy * dy)


def fmtShortcut(text):
    mod, key = text.split("+", 1)
    return "<b>%s</b>+<b>%s</b>" % (mod, key)
