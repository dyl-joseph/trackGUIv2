from qtpy import QtCore
from qtpy import QtGui


def move_to_safe_cursor_position(dialog, list_widget=None):
    """Size and position a popup wholly inside the cursor's screen."""
    dialog.adjustSize()
    cursor = QtGui.QCursor.pos()
    cursor_pos = cursor + QtCore.QPoint(16, 16)
    screen = QtGui.QGuiApplication.screenAt(cursor)
    if screen is None:
        screen = QtGui.QGuiApplication.primaryScreen()
    if screen is None:
        dialog.move(cursor_pos)
        return

    available = screen.availableGeometry()
    if list_widget is not None:
        list_widget.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAsNeeded)
        list_widget.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAsNeeded)
        list_widget.setMinimumSize(0, 0)
        list_widget.setMaximumSize(available.width(), available.height())

    frame_extra_width = max(0, dialog.frameGeometry().width() - dialog.width())
    frame_extra_height = max(0, dialog.frameGeometry().height() - dialog.height())
    # Leave a small frame-decoration allowance. Some headless Qt platforms do
    # not report those margins until after the first geometry change.
    width = min(dialog.sizeHint().width(), available.width() - frame_extra_width - 2)
    height = min(
        dialog.sizeHint().height(), available.height() - frame_extra_height - 2
    )
    dialog.setFixedSize(max(1, width), max(1, height))

    frame = dialog.frameGeometry()
    boundary_padding = 4
    frame_x = max(
        min(
            cursor_pos.x(),
            available.right() - frame.width() + 1 - boundary_padding,
        ),
        available.left(),
    )
    frame_y = max(
        min(
            cursor_pos.y(),
            available.bottom() - frame.height() + 1 - boundary_padding,
        ),
        available.top(),
    )
    client_offset = dialog.pos() - frame.topLeft()
    dialog.move(frame_x + client_offset.x(), frame_y + client_offset.y())
