import io
import json

import numpy as np
import PIL.Image
import pytest

from labelme import label_file as label_file_module
from labelme.label_file import LabelFile
from labelme.label_file import LabelFileError
from labelme.label_file import save_label_files_atomically


def _png_bytes(width=4, height=3):
    output = io.BytesIO()
    PIL.Image.new("RGB", (width, height), (10, 20, 30)).save(output, "PNG")
    return output.getvalue()


def _shape(**updates):
    shape = {
        "label": "person",
        "points": [[0, 0], [2, 2]],
        "group_id": 7,
        "track_id": 0,
        "shape_type": "rectangle",
        "flags": {"occluded": True},
        "description": "kept",
        "mask": None,
        "confidence": 0.75,
    }
    shape.update(updates)
    return shape


def test_roundtrip_preserves_falsy_track_id_metadata_and_embedded_image(tmp_path):
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    image_data = _png_bytes()
    LabelFile().save(
        filename=str(first),
        shapes=[_shape()],
        imagePath="frame.png",
        imageHeight=None,
        imageWidth=None,
        imageData=image_data,
        otherData={"review": {"state": "pending"}},
        flags={"approved": True},
    )

    loaded = LabelFile(str(first))

    assert loaded.shapes[0]["track_id"] == 0
    assert loaded.shapes[0]["other_data"] == {"confidence": 0.75}
    assert loaded.otherData == {"review": {"state": "pending"}}
    assert loaded.flags == {"approved": True}
    assert loaded.imageData == image_data
    assert (loaded.imageWidth, loaded.imageHeight) == (4, 3)

    LabelFile().save(
        filename=str(second),
        shapes=loaded.shapes,
        imagePath=loaded.imagePath,
        imageHeight=loaded.imageHeight,
        imageWidth=loaded.imageWidth,
        imageData=loaded.imageData,
        otherData=loaded.otherData,
        flags=loaded.flags,
    )
    serialized = json.loads(second.read_text(encoding="utf-8"))
    assert serialized["shapes"][0]["confidence"] == 0.75
    assert serialized["shapes"][0]["track_id"] == 0
    assert serialized["review"] == {"state": "pending"}
    assert serialized["imageData"] is not None


def test_failed_serialization_does_not_truncate_existing_file(tmp_path, monkeypatch):
    destination = tmp_path / "annotation.json"
    destination.write_text('{"original": true}', encoding="utf-8")

    def fail_dump(_data, handle, **_kwargs):
        handle.write('{"partial":')
        raise OSError("disk full")

    monkeypatch.setattr(label_file_module.json, "dump", fail_dump)

    with pytest.raises(LabelFileError, match="disk full"):
        LabelFile().save(
            filename=str(destination),
            shapes=[_shape()],
            imagePath="frame.png",
            imageHeight=3,
            imageWidth=4,
        )

    assert destination.read_text(encoding="utf-8") == '{"original": true}'
    assert list(tmp_path.glob("*.tmp")) == []


def test_batch_failure_rolls_back_every_destination(tmp_path, monkeypatch):
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    first.write_text('{"value": "first"}', encoding="utf-8")
    second.write_text('{"value": "second"}', encoding="utf-8")
    original_replace = label_file_module.os.replace

    def fail_second_install(source, destination):
        if destination == str(second) and "-stage-" in source:
            raise OSError("commit failed")
        return original_replace(source, destination)

    monkeypatch.setattr(label_file_module.os, "replace", fail_second_install)
    requests = [
        dict(
            filename=str(path),
            shapes=[_shape(label=path.stem)],
            imagePath="frame.png",
            imageHeight=3,
            imageWidth=4,
        )
        for path in (first, second)
    ]

    with pytest.raises(LabelFileError, match="commit failed"):
        save_label_files_atomically(requests)

    assert json.loads(first.read_text(encoding="utf-8")) == {"value": "first"}
    assert json.loads(second.read_text(encoding="utf-8")) == {"value": "second"}


def test_loaded_mask_can_be_saved_again_without_losing_pixels(tmp_path):
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    mask = np.array([[False, True], [True, False]])
    LabelFile().save(
        filename=str(first),
        shapes=[_shape(shape_type="mask", mask=mask)],
        imagePath="frame.png",
        imageHeight=3,
        imageWidth=4,
    )

    loaded = LabelFile(str(first))
    LabelFile().save(
        filename=str(second),
        shapes=loaded.shapes,
        imagePath=loaded.imagePath,
        imageHeight=loaded.imageHeight,
        imageWidth=loaded.imageWidth,
    )

    reloaded = LabelFile(str(second))
    assert np.array_equal(reloaded.shapes[0]["mask"].astype(bool), mask)
