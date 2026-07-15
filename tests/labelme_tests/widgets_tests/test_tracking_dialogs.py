import pytest

from labelme.widgets.deletetrack_dialog import DeletionDialog
from labelme.widgets.interpolation_dialog import InterpolationDialog


@pytest.mark.gui
def test_deletion_dialog_uses_bounded_frame_controls(qtbot):
    dialog = DeletionDialog()
    qtbot.addWidget(dialog)

    dialog.setFrameRange(1, 10, 4)

    assert dialog.start_frame_cell.minimum() == 1
    assert dialog.start_frame_cell.maximum() == 10
    assert dialog.start_frame_cell.value() == 4
    assert dialog.end_frame_cell.value() == 10


@pytest.mark.gui
def test_interpolation_dialog_returns_one_typed_options_object(qtbot):
    dialog = InterpolationDialog(1, 10)
    qtbot.addWidget(dialog)
    dialog.start_frame_cell.setValue(2)
    dialog.end_frame_cell.setValue(9)
    dialog.interval_cell.setValue(3)
    dialog.ID_cell.setText(" 0 ")
    dialog.label_cell.setText(" person ")

    options = dialog.options()

    assert (options.start_frame, options.end_frame, options.interval) == (2, 9, 3)
    assert (options.track_id, options.label) == ("0", "person")


@pytest.mark.gui
def test_interpolation_dialog_rejects_missing_identity(qtbot):
    dialog = InterpolationDialog(1, 2)
    qtbot.addWidget(dialog)

    with pytest.raises(ValueError, match="label and ID"):
        dialog.options()
