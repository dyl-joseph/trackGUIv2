import importlib.util
import io
import json
import sys
import types
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from labelme.cli.draw_label_png import describe_label_values
from labelme.label_file import LabelFile


def _load_coco_exporter():
    path = (
        Path(__file__).parents[2]
        / "examples"
        / "instance_segmentation"
        / "labelme2coco.py"
    )
    spec = importlib.util.spec_from_file_location("labelme2coco", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_voc_exporter():
    path = (
        Path(__file__).parents[2]
        / "examples"
        / "semantic_segmentation"
        / "labelme2voc.py"
    )
    spec = importlib.util.spec_from_file_location("labelme2voc", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_nested_annotations(input_dir):
    output = io.BytesIO()
    Image.new("RGB", (8, 6), color=(10, 20, 30)).save(output, format="PNG")
    image_data = output.getvalue()
    for camera, track_id, points in [
        ("camera1", 1, [[1, 1], [5, 4]]),
        ("camera2", 2, [[1, 1], [5, 4]]),
        ("camera3", 3, [[20, 20], [30, 30]]),
    ]:
        filename = input_dir / camera / "frame.json"
        filename.parent.mkdir(parents=True)
        LabelFile().save(
            filename=str(filename),
            shapes=[
                {
                    "label": "person",
                    "points": points,
                    "group_id": track_id,
                    "track_id": track_id,
                    "shape_type": "rectangle",
                    "flags": {},
                    "description": "",
                    "mask": None,
                }
            ],
            imagePath="frame.png",
            imageData=image_data,
            imageHeight=6,
            imageWidth=8,
            flags={},
        )


def test_coco_categories_are_positive_and_skip_reserved_names(tmp_path):
    labels = tmp_path / "labels.txt"
    labels.write_text("__ignore__\n_background_\nperson\nvehicle\n", encoding="utf-8")

    assert _load_coco_exporter().load_categories(labels) == {
        "person": 1,
        "vehicle": 2,
    }


def test_coco_categories_reject_duplicates(tmp_path):
    labels = tmp_path / "labels.txt"
    labels.write_text("person\nperson\n", encoding="utf-8")

    with pytest.raises(ValueError, match="Duplicate"):
        _load_coco_exporter().load_categories(labels)


def test_draw_label_names_handles_ignore_without_negative_indexing():
    assert describe_label_values([-1, 0], ["background"]) == [
        "-1:__ignore__",
        "0:background",
    ]


def test_draw_label_names_rejects_unmapped_values():
    with pytest.raises(ValueError, match="no matching names"):
        describe_label_values([2], ["background"])


def test_voc_exporter_recurses_and_preserves_duplicate_basename_paths(
    tmp_path, monkeypatch
):
    input_dir = tmp_path / "annotations"
    output_dir = tmp_path / "voc"
    _write_nested_annotations(input_dir)
    module = _load_voc_exporter()
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "labelme2voc",
            str(input_dir),
            str(output_dir),
            "--labels",
            "__ignore__,_background_,person",
            "--noobject",
            "--nonpy",
            "--noviz",
        ],
    )

    module.main()

    assert (output_dir / "JPEGImages" / "camera1" / "frame.jpg").is_file()
    assert (output_dir / "JPEGImages" / "camera2" / "frame.jpg").is_file()
    assert (output_dir / "SegmentationClass" / "camera1" / "frame.png").is_file()
    assert (output_dir / "SegmentationClass" / "camera2" / "frame.png").is_file()


def test_coco_exporter_recurses_and_preserves_duplicate_basename_paths(
    tmp_path, monkeypatch
):
    class FakeMaskApi:
        @staticmethod
        def encode(mask):
            ys, xs = np.where(mask)
            return {
                "counts": b"encoded",
                "size": list(mask.shape),
                "area": float(mask.sum()),
                "bbox": [
                    float(xs.min()),
                    float(ys.min()),
                    float(xs.max() - xs.min() + 1),
                    float(ys.max() - ys.min() + 1),
                ],
            }

        @staticmethod
        def area(encoded):
            return encoded["area"]

        @staticmethod
        def toBbox(encoded):
            return np.asarray(encoded["bbox"], dtype=float)

    input_dir = tmp_path / "annotations"
    output_dir = tmp_path / "coco"
    labels = tmp_path / "labels.txt"
    labels.write_text("person\n", encoding="utf-8")
    _write_nested_annotations(input_dir)
    module = _load_coco_exporter()
    module.pycocotools = types.SimpleNamespace(mask=FakeMaskApi)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "labelme2coco",
            str(input_dir),
            str(output_dir),
            "--labels",
            str(labels),
            "--noviz",
        ],
    )

    module.main()

    assert (output_dir / "JPEGImages" / "camera1" / "frame.jpg").is_file()
    assert (output_dir / "JPEGImages" / "camera2" / "frame.jpg").is_file()
    data = json.loads((output_dir / "annotations.json").read_text(encoding="utf-8"))
    assert [image["file_name"] for image in data["images"]] == [
        "JPEGImages/camera1/frame.jpg",
        "JPEGImages/camera2/frame.jpg",
        "JPEGImages/camera3/frame.jpg",
    ]
    assert len(data["annotations"]) == 2
    assert all(annotation["area"] > 0 for annotation in data["annotations"])


def test_exporters_reject_nonfinite_shape_coordinates():
    shape = {
        "label": "person",
        "points": [[float("nan"), 1], [2, 3]],
        "shape_type": "rectangle",
    }

    assert not _load_coco_exporter().shape_points_are_finite(shape)
    assert (
        _load_voc_exporter().valid_shapes_for_image(
            [shape], (6, 8), {"person": 1}, "bad.json"
        )
        == []
    )
