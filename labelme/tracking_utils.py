import math


def shape_track_id(shape):
    value = shape.get("track_id")
    if value is None or value == "":
        value = shape.get("group_id")
    return value


def interpolation_indices(start_frame, end_frame, interval, frame_count):
    """Return zero-based reference frames, always including the selected end."""
    if not 1 <= start_frame <= end_frame <= frame_count:
        raise ValueError("frames must satisfy 1 <= start <= end <= frame count")
    if interval <= 0:
        raise ValueError("interval must be greater than zero")
    indices = list(range(start_frame - 1, end_frame, interval))
    endpoint = end_frame - 1
    if indices[-1] != endpoint:
        indices.append(endpoint)
    return indices


def normalized_rectangle_points(points):
    try:
        point_count = len(points)
    except TypeError as exc:
        raise ValueError("rectangle points must be a sequence") from exc
    if point_count != 2:
        raise ValueError("rectangle must have exactly two points")
    try:
        (x1, y1), (x2, y2) = points
    except (TypeError, ValueError) as exc:
        raise ValueError("rectangle points must be coordinate pairs") from exc
    coordinates = (x1, y1, x2, y2)
    if any(isinstance(value, bool) for value in coordinates):
        raise ValueError("rectangle coordinates must be finite numbers")
    try:
        x1, y1, x2, y2 = (float(value) for value in coordinates)
    except (TypeError, ValueError) as exc:
        raise ValueError("rectangle coordinates must be finite numbers") from exc
    if not all(math.isfinite(value) for value in (x1, y1, x2, y2)):
        raise ValueError("rectangle coordinates must be finite numbers")
    left, right = sorted((x1, x2))
    top, bottom = sorted((y1, y2))
    if left == right or top == bottom:
        raise ValueError("rectangle must have positive width and height")
    return [[left, top], [right, bottom]]


def prediction_to_clamped_rectangle(prediction, image_width, image_height):
    """Convert a center/size prediction to a finite, in-image rectangle."""
    if image_width <= 0 or image_height <= 0:
        raise ValueError("target image dimensions must be positive")
    try:
        center_x, center_y, width, height = prediction
    except (TypeError, ValueError) as exc:
        raise ValueError("interpolation prediction must have four values") from exc
    values = (center_x, center_y, width, height)
    if any(isinstance(value, bool) for value in values):
        raise ValueError("interpolation prediction must contain finite numbers")
    try:
        center_x, center_y, width, height = (float(value) for value in values)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "interpolation prediction must contain finite numbers"
        ) from exc
    if not all(math.isfinite(value) for value in (center_x, center_y, width, height)):
        raise ValueError("interpolation prediction must contain finite numbers")
    if width <= 0 or height <= 0:
        raise ValueError("interpolation predicted a non-positive box size")

    x1 = min(max(0.0, center_x - width / 2), max(0.0, image_width - 1.0))
    y1 = min(max(0.0, center_y - height / 2), max(0.0, image_height - 1.0))
    x2 = min(max(0.0, center_x + width / 2), float(image_width))
    y2 = min(max(0.0, center_y + height / 2), float(image_height))
    if x2 <= x1 or y2 <= y1:
        raise ValueError("interpolation produced an empty box")
    return [[x1, y1], [x2, y2]]


def upsert_tracked_rectangle(shapes, label, track_id, points, group_id=None):
    """Replace duplicate tracked rectangles while preserving stored metadata."""
    matching = [
        shape
        for shape in shapes
        if shape.get("label") == label and str(shape_track_id(shape)) == str(track_id)
    ]
    conflicting = [
        shape for shape in matching if shape.get("shape_type", "polygon") != "rectangle"
    ]
    if conflicting:
        raise ValueError(
            "Track {}-{} conflicts with {} non-rectangle shape(s).".format(
                label, track_id, len(conflicting)
            )
        )
    rectangles = [shape for shape in matching if shape.get("shape_type") == "rectangle"]
    new_shape = dict(rectangles[0]) if rectangles else {}
    stored_track_id = new_shape.get("track_id")
    if stored_track_id is None or stored_track_id == "":
        stored_track_id = new_shape.get("group_id")
    if stored_track_id is None or stored_track_id == "":
        stored_track_id = track_id

    if "group_id" in new_shape:
        stored_group_id = new_shape["group_id"]
    elif group_id is not None:
        stored_group_id = group_id
    else:
        stored_group_id = int(track_id) if str(track_id).isdigit() else track_id
    new_shape.update(
        label=label,
        points=points,
        shape_type="rectangle",
        flags=new_shape.get("flags", {}),
        description=new_shape.get("description", ""),
        group_id=stored_group_id,
        track_id=stored_track_id,
        mask=None,
    )
    remaining = [
        shape
        for shape in shapes
        if not (
            shape.get("label") == label
            and shape.get("shape_type") == "rectangle"
            and str(shape_track_id(shape)) == str(track_id)
        )
    ]
    remaining.append(new_shape)
    return remaining
