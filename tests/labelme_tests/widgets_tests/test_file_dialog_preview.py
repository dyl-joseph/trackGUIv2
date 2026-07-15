import pytest

from labelme.widgets import FileDialogPreview


@pytest.mark.gui
def test_invalid_json_preview_is_plain_bounded_error(qtbot, tmp_path):
    invalid_json = tmp_path / "broken.json"
    invalid_json.write_text("<b>not json</b>", encoding="utf-8")
    dialog = FileDialogPreview()
    qtbot.addWidget(dialog)

    dialog.onChange(str(invalid_json))

    assert dialog.labelPreview.label.text().startswith("Unable to preview JSON:")
    assert "<b>" not in dialog.labelPreview.label.text()
