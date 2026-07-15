import importlib.util
import io
import json
import sys
import types
import xml.etree.ElementTree as ElementTree
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


def _load_bbox_voc_exporter():
    path = Path(__file__).parents[2] / "examples" / "bbox_detection" / "labelme2voc.py"
    spec = importlib.util.spec_from_file_location("bbox_labelme2voc", path)
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


def test_coco_serialized_paths_use_posix_separators_on_every_host():
    module = _load_coco_exporter()

    assert (
        module.portable_dataset_path(r"JPEGImages\camera1\frame.jpg")
        == "JPEGImages/camera1/frame.jpg"
    )


def test_exporter_discovery_ignores_hidden_annotation_artifacts(tmp_path):
    visible = tmp_path / "visible.json"
    hidden = tmp_path / ".frame-backup.json"
    hidden_dir = tmp_path / ".transaction" / "frame.json"
    visible.write_text("{}", encoding="utf-8")
    hidden.write_text("{}", encoding="utf-8")
    hidden_dir.parent.mkdir()
    hidden_dir.write_text("{}", encoding="utf-8")

    for module in (
        _load_coco_exporter(),
        _load_voc_exporter(),
        _load_bbox_voc_exporter(),
    ):
        assert module.find_label_files(tmp_path) == [str(visible)]


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


def test_bbox_voc_exporter_recurses_and_preserves_duplicate_basename_paths(
    tmp_path, monkeypatch
):
    input_dir = tmp_path / "annotations"
    output_dir = tmp_path / "bbox-voc"
    labels = tmp_path / "labels.txt"
    labels.write_text("__ignore__\n_background_\nperson\n", encoding="utf-8")
    _write_nested_annotations(input_dir)
    module = _load_bbox_voc_exporter()
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "bbox-labelme2voc",
            str(input_dir),
            str(output_dir),
            "--labels",
            str(labels),
            "--noviz",
        ],
    )

    module.main()

    for camera in ["camera1", "camera2", "camera3"]:
        assert (output_dir / "JPEGImages" / camera / "frame.jpg").is_file()
        assert (output_dir / "Annotations" / camera / "frame.xml").is_file()
    first_xml = ElementTree.parse(
        output_dir / "Annotations" / "camera1" / "frame.xml"
    ).getroot()
    second_xml = ElementTree.parse(
        output_dir / "Annotations" / "camera2" / "frame.xml"
    ).getroot()
    invalid_xml = ElementTree.parse(
        output_dir / "Annotations" / "camera3" / "frame.xml"
    ).getroot()
    assert first_xml.findtext("filename") == "camera1/frame.jpg"
    assert second_xml.findtext("filename") == "camera2/frame.jpg"
    assert len(first_xml.findall("object")) == 1
    assert len(second_xml.findall("object")) == 1
    assert invalid_xml.findall("object") == []


def test_bbox_voc_emits_integer_text_for_inverted_and_clipped_rectangles(
    tmp_path, monkeypatch
):
    input_dir = tmp_path / "annotations"
    output_dir = tmp_path / "bbox-voc"
    labels = tmp_path / "labels.txt"
    labels.write_text("__ignore__\n_background_\nperson\n", encoding="utf-8")
    image_output = io.BytesIO()
    Image.new("RGB", (8, 6), color=(10, 20, 30)).save(image_output, format="PNG")
    annotation = input_dir / "frame.json"
    input_dir.mkdir()
    shapes = []
    for points in (
        [[5.8, 4.2], [1.2, 0.2]],
        [[-2.4, 1.2], [3.1, 8.8]],
    ):
        shapes.append(
            {
                "label": "person",
                "points": points,
                "group_id": None,
                "track_id": None,
                "shape_type": "rectangle",
                "flags": {},
                "description": "",
                "mask": None,
            }
        )
    LabelFile().save(
        filename=str(annotation),
        shapes=shapes,
        imagePath="frame.png",
        imageData=image_output.getvalue(),
        imageHeight=6,
        imageWidth=8,
        flags={},
    )
    module = _load_bbox_voc_exporter()
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "bbox-labelme2voc",
            str(input_dir),
            str(output_dir),
            "--labels",
            str(labels),
            "--noviz",
        ],
    )

    module.main()

    root = ElementTree.parse(output_dir / "Annotations" / "frame.xml").getroot()
    coordinates = [
        tuple(box.findtext(name) for name in ("xmin", "ymin", "xmax", "ymax"))
        for box in root.findall("object/bndbox")
    ]
    assert coordinates == [("1", "0", "6", "5"), ("0", "1", "4", "6")]


def test_exporters_reject_nonfinite_shape_coordinates():
    shape = {
        "label": "person",
        "points": [[float("nan"), 1], [2, 3]],
        "shape_type": "rectangle",
    }

    assert not _load_coco_exporter().shape_points_are_finite(shape)
    with pytest.raises(ValueError, match="finite numbers"):
        _load_bbox_voc_exporter().normalized_clipped_rectangle(shape["points"], 8, 6)
    assert (
        _load_voc_exporter().valid_shapes_for_image(
            [shape], (6, 8), {"person": 1}, "bad.json"
        )
        == []
    )


@pytest.mark.parametrize(
    "shape",
    [
        {
            "label": "person",
            "points": [[2, 2], [2, 2]],
            "shape_type": "circle",
        },
        {
            "label": "person",
            "points": [[0, 0], [2, 2], [4, 4]],
            "shape_type": "polygon",
        },
        {
            "label": "person",
            "points": [[1, 1], [1, 5]],
            "shape_type": "rectangle",
        },
    ],
)
def test_coco_exporter_rejects_degenerate_geometry(shape):
    assert not _load_coco_exporter().shape_geometry_is_valid(shape)
