from labelme.annotation_path import canonical_annotation_path
from labelme.annotation_path import resolve_annotation_path


def test_output_paths_preserve_nested_duplicate_basenames(tmp_path):
    image_root = tmp_path / "images"
    first = image_root / "camera1" / "frame.jpg"
    second = image_root / "camera2" / "frame.jpg"
    output = tmp_path / "labels"

    assert canonical_annotation_path(first, output, image_root) == str(
        output / "camera1" / "frame.jpg.json"
    )
    assert canonical_annotation_path(second, output, image_root) == str(
        output / "camera2" / "frame.jpg.json"
    )


def test_legacy_flat_file_is_read_only_when_basename_is_unambiguous(tmp_path):
    image_root = tmp_path / "images"
    image = image_root / "camera1" / "frame.jpg"
    duplicate = image_root / "camera2" / "frame.jpg"
    output = tmp_path / "labels"
    output.mkdir()
    legacy = output / "frame.json"
    legacy.write_text("{}", encoding="utf-8")
    canonical = output / "camera1" / "frame.jpg.json"

    assert resolve_annotation_path(
        image, output, image_root, [image], for_write=False
    ) == str(legacy)
    assert resolve_annotation_path(
        image, output, image_root, [image], for_write=True
    ) == str(canonical)
    assert resolve_annotation_path(
        image, output, image_root, [image, duplicate], for_write=False
    ) == str(canonical)


def test_output_paths_disambiguate_same_stem_different_image_extensions(tmp_path):
    image_root = tmp_path / "images"
    jpeg = image_root / "camera1" / "frame.jpg"
    png = image_root / "camera1" / "frame.png"
    output = tmp_path / "labels"
    output.mkdir()
    legacy = output / "frame.json"
    legacy.write_text("{}", encoding="utf-8")

    jpeg_path = canonical_annotation_path(jpeg, output, image_root)
    png_path = canonical_annotation_path(png, output, image_root)

    assert jpeg_path == str(output / "camera1" / "frame.jpg.json")
    assert png_path == str(output / "camera1" / "frame.png.json")
    assert jpeg_path != png_path
    assert (
        resolve_annotation_path(jpeg, output, image_root, [jpeg, png], for_write=False)
        == jpeg_path
    )
    assert (
        resolve_annotation_path(png, output, image_root, [jpeg, png], for_write=False)
        == png_path
    )


def test_images_outside_sequence_root_get_collision_safe_output_paths(tmp_path):
    image_root = tmp_path / "sequence"
    first = tmp_path / "external1" / "frame.jpg"
    second = tmp_path / "external2" / "frame.jpg"
    output = tmp_path / "labels"

    first_path = canonical_annotation_path(first, output, image_root)
    second_path = canonical_annotation_path(second, output, image_root)

    assert first_path != second_path
    assert first_path.startswith(str(output / "_external"))
    assert second_path.startswith(str(output / "_external"))


def test_images_without_sequence_root_get_collision_safe_output_paths(tmp_path):
    first = tmp_path / "external1" / "frame.jpg"
    second = tmp_path / "external2" / "frame.jpg"
    output = tmp_path / "labels"

    first_path = canonical_annotation_path(first, output)
    second_path = canonical_annotation_path(second, output)

    assert first_path != second_path
    assert first_path.startswith(str(output / "_external"))
    assert second_path.startswith(str(output / "_external"))
