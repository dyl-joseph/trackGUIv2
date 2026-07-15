import hashlib
import os.path as osp


def _replace_extension(path, extension=".json"):
    return osp.splitext(path)[0] + extension


def _is_within(path, directory):
    try:
        common_path = osp.commonpath([osp.abspath(path), osp.abspath(directory)])
        return common_path == osp.abspath(directory)
    except ValueError:
        return False


def canonical_annotation_path(image_path, output_dir=None, image_root=None):
    """Return the one authoritative annotation path for an image.

    Output directories mirror the image's path below ``image_root``. Images that
    are outside an established root use a stable external-parent namespace so
    duplicate basenames can never overwrite one another.
    """
    if image_path is None:
        return None
    image_path = osp.abspath(osp.normpath(image_path))
    if image_path.lower().endswith(".json"):
        return image_path
    if output_dir is None:
        return _replace_extension(image_path)

    relative_path = osp.basename(image_path)
    if image_root:
        if _is_within(image_path, image_root):
            relative_path = osp.relpath(image_path, osp.abspath(image_root))
        else:
            parent_key = hashlib.sha256(
                osp.dirname(image_path).encode("utf-8")
            ).hexdigest()[:12]
            relative_path = osp.join("_external", parent_key, relative_path)
    else:
        parent_key = hashlib.sha256(
            osp.dirname(image_path).encode("utf-8")
        ).hexdigest()[:12]
        relative_path = osp.join("_external", parent_key, relative_path)
    return osp.join(osp.abspath(output_dir), relative_path + ".json")


def legacy_annotation_paths(
    image_path, output_dir=None, image_root=None, image_paths=()
):
    """Return existing, unambiguous legacy flat paths for an image."""
    if image_path is None:
        return []
    image_path = osp.abspath(osp.normpath(image_path))
    if image_path.lower().endswith(".json"):
        return []
    canonical = canonical_annotation_path(image_path, output_dir, image_root)
    basename = osp.basename(image_path)
    legacy_basename = _replace_extension(basename)
    matching_basenames = [
        path
        for path in image_paths
        if _replace_extension(osp.basename(osp.normpath(str(path)))) == legacy_basename
    ]
    if len(matching_basenames) > 1:
        return []

    candidates = []
    if output_dir:
        candidates.append(
            osp.join(osp.abspath(output_dir), _replace_extension(basename))
        )
    elif image_root:
        candidates.append(
            osp.join(osp.abspath(image_root), _replace_extension(basename))
        )
    paths = []
    for candidate in candidates:
        candidate = osp.abspath(candidate)
        if (
            candidate != osp.abspath(canonical)
            and candidate not in paths
            and osp.isfile(candidate)
        ):
            paths.append(candidate)
    return paths


def resolve_annotation_path(
    image_path,
    output_dir=None,
    image_root=None,
    image_paths=(),
    for_write=False,
    explicit_label_path=None,
):
    """Resolve an annotation path, with an unambiguous legacy read fallback."""
    if explicit_label_path:
        return osp.abspath(explicit_label_path)

    canonical = canonical_annotation_path(image_path, output_dir, image_root)
    if for_write or canonical is None or osp.isfile(canonical):
        return canonical

    for legacy_path in legacy_annotation_paths(
        image_path,
        output_dir=output_dir,
        image_root=image_root,
        image_paths=image_paths,
    ):
        return legacy_path
    return canonical
