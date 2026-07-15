import importlib.util
from pathlib import Path

import pytest

from labelme.cli.draw_label_png import describe_label_values


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
