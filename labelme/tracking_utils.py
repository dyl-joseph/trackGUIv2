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
    if len(points) != 2:
        raise ValueError("rectangle must have exactly two points")
    (x1, y1), (x2, y2) = points
    left, right = sorted((float(x1), float(x2)))
    top, bottom = sorted((float(y1), float(y2)))
    if left == right or top == bottom:
        raise ValueError("rectangle must have positive width and height")
    return [[left, top], [right, bottom]]
