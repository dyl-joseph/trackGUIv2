import math
import uuid

import numpy as np
import PIL.Image
import PIL.ImageDraw

from labelme.logger import logger
from labelme.utils.image import img_b64_to_arr


def polygons_to_mask(img_shape, polygons, shape_type=None):
    logger.warning(
        "The 'polygons_to_mask' function is deprecated, use 'shape_to_mask' instead."
    )
    return shape_to_mask(img_shape, points=polygons, shape_type=shape_type)


def shape_to_mask(img_shape, points, shape_type=None, line_width=10, point_size=5):
    mask = np.zeros(img_shape[:2], dtype=np.uint8)
    mask = PIL.Image.fromarray(mask)
    draw = PIL.ImageDraw.Draw(mask)
    xy = [tuple(point) for point in points]
    if shape_type == "circle":
        assert len(xy) == 2, "Shape of shape_type=circle must have 2 points"
        (cx, cy), (px, py) = xy
        d = math.sqrt((cx - px) ** 2 + (cy - py) ** 2)
        draw.ellipse([cx - d, cy - d, cx + d, cy + d], outline=1, fill=1)
    elif shape_type == "rectangle":
        assert len(xy) == 2, "Shape of shape_type=rectangle must have 2 points"
        (x1, y1), (x2, y2) = xy
        draw.rectangle(
            [(min(x1, x2), min(y1, y2)), (max(x1, x2), max(y1, y2))],
            outline=1,
            fill=1,
        )
    elif shape_type == "line":
        assert len(xy) == 2, "Shape of shape_type=line must have 2 points"
        draw.line(xy=xy, fill=1, width=line_width)
    elif shape_type == "linestrip":
        draw.line(xy=xy, fill=1, width=line_width)
    elif shape_type == "point":
        assert len(xy) == 1, "Shape of shape_type=point must have 1 points"
        cx, cy = xy[0]
        r = point_size
        draw.ellipse([cx - r, cy - r, cx + r, cy + r], outline=1, fill=1)
    else:
        assert len(xy) > 2, "Polygon must have points more than 2"
        draw.polygon(xy=xy, outline=1, fill=1)
    mask = np.array(mask, dtype=bool)
    return mask


def shapes_to_label(img_shape, shapes, label_name_to_value):
    cls = np.zeros(img_shape[:2], dtype=np.int32)
    ins = np.zeros_like(cls)
    instances = []
    for shape in shapes:
        points = shape["points"]
        label = shape["label"]
        group_id = shape.get("group_id")
        if group_id is None:
            group_id = uuid.uuid1()
        shape_type = shape.get("shape_type", None)

        cls_name = label
        instance = (cls_name, group_id)

        if instance not in instances:
            instances.append(instance)
        ins_id = instances.index(instance) + 1
        cls_id = label_name_to_value[cls_name]

        if shape_type == "mask":
            stored_mask = shape.get("mask")
            if stored_mask is None:
                raise ValueError("A mask shape must include stored mask data")
            if isinstance(stored_mask, str):
                stored_mask = img_b64_to_arr(stored_mask)
            stored_mask = np.asarray(stored_mask, dtype=bool)
            if stored_mask.ndim != 2 or not stored_mask.any():
                raise ValueError("A mask shape must contain a non-empty 2D mask")
            if not points:
                raise ValueError("A mask shape must include an origin point")
            x1, y1 = (int(round(value)) for value in points[0])
            mask = np.zeros(img_shape[:2], dtype=bool)
            src_x1 = max(0, -x1)
            src_y1 = max(0, -y1)
            dst_x1 = max(0, x1)
            dst_y1 = max(0, y1)
            width = min(stored_mask.shape[1] - src_x1, mask.shape[1] - dst_x1)
            height = min(stored_mask.shape[0] - src_y1, mask.shape[0] - dst_y1)
            if width > 0 and height > 0:
                mask[dst_y1 : dst_y1 + height, dst_x1 : dst_x1 + width] = stored_mask[
                    src_y1 : src_y1 + height,
                    src_x1 : src_x1 + width,
                ]
        else:
            mask = shape_to_mask(img_shape[:2], points, shape_type)
        cls[mask] = cls_id
        ins[mask] = ins_id

    return cls, ins


def labelme_shapes_to_label(img_shape, shapes):
    logger.warning(
        "labelme_shapes_to_label is deprecated, so please use shapes_to_label."
    )

    label_name_to_value = {"_background_": 0}
    for shape in shapes:
        label_name = shape["label"]
        if label_name in label_name_to_value:
            label_value = label_name_to_value[label_name]
        else:
            label_value = len(label_name_to_value)
            label_name_to_value[label_name] = label_value

    lbl, _ = shapes_to_label(img_shape, shapes, label_name_to_value)
    return lbl, label_name_to_value


def masks_to_bboxes(masks):
    masks = np.asarray(masks)
    if masks.size == 0:
        raise ValueError("masks must contain at least one non-empty mask")
    if masks.ndim != 3:
        raise ValueError("masks.ndim must be 3, but it is {}".format(masks.ndim))
    if masks.dtype != bool:
        raise ValueError(
            "masks.dtype must be bool type, but it is {}".format(masks.dtype)
        )
    bboxes = []
    for index, mask in enumerate(masks):
        where = np.argwhere(mask)
        if where.size == 0:
            raise ValueError("mask at index {} is empty".format(index))
        (y1, x1), (y2, x2) = where.min(0), where.max(0) + 1
        bboxes.append((y1, x1, y2, x2))
    bboxes = np.asarray(bboxes, dtype=np.float32)
    return bboxes
