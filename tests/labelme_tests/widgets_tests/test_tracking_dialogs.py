import pytest

from labelme.widgets.deletetrack_dialog import DeletionDialog
from labelme.widgets.interpolation_dialog import InterpolationDialog
from labelme.widgets.track_dialog import TrackDialog


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


@pytest.mark.gui
def test_sort_dialog_allows_scratch_range_and_defaults_to_sequence_end(qtbot):
    dialog = TrackDialog(current_frame=10, total_frames=20)
    qtbot.addWidget(dialog)

    assert dialog.end_frame.minimum() == 1
    assert dialog.end_frame.maximum() == 20
    assert dialog.end_frame.value() == 20
