#!/usr/bin/env python

import argparse
import collections
import datetime
import glob
import json
import math
import os
import os.path as osp
import sys
import uuid

import imgviz
import numpy as np

import labelme

try:
    import pycocotools.mask
except ImportError:
    pycocotools = None


def load_categories(filename):
    """Return positive COCO category IDs, excluding LabelMe sentinels."""
    with open(filename, encoding="utf-8") as handle:
        class_names = [line.strip() for line in handle if line.strip()]
    class_name_to_id = {}
    for class_name in class_names:
        if class_name in {"__ignore__", "_background_", "__background__"}:
            continue
        if class_name in class_name_to_id:
            raise ValueError("Duplicate category name: {}".format(class_name))
        class_name_to_id[class_name] = len(class_name_to_id) + 1
    return class_name_to_id


def find_label_files(input_dir):
    pattern = osp.join(osp.abspath(input_dir), "**", "*.json")
    return sorted(glob.glob(pattern, recursive=True))


def relative_output_stem(filename, input_dir):
    relative = osp.relpath(osp.abspath(filename), osp.abspath(input_dir))
    return osp.splitext(relative)[0]


def ensure_parent(filename):
    os.makedirs(osp.dirname(filename), exist_ok=True)


def shape_points_are_finite(shape):
    try:
        points = np.asarray(shape.get("points", []), dtype=float)
    except (TypeError, ValueError):
        return False
    return points.ndim == 2 and points.shape[1:] == (2,) and np.isfinite(points).all()


def main():
    if pycocotools is None:
        print("Please install pycocotools:\n\n    pip install pycocotools\n")
        sys.exit(1)
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("input_dir", help="input annotated directory")
    parser.add_argument("output_dir", help="output dataset directory")
    parser.add_argument("--labels", help="labels file", required=True)
    parser.add_argument("--noviz", help="no visualization", action="store_true")
    args = parser.parse_args()

    if osp.exists(args.output_dir):
        print("Output directory already exists:", args.output_dir)
        sys.exit(1)
    os.makedirs(args.output_dir)
    os.makedirs(osp.join(args.output_dir, "JPEGImages"))
    if not args.noviz:
        os.makedirs(osp.join(args.output_dir, "Visualization"))
    print("Creating dataset:", args.output_dir)

    now = datetime.datetime.now()

    data = dict(
        info=dict(
            description=None,
            url=None,
            version=None,
            year=now.year,
            contributor=None,
            date_created=now.strftime("%Y-%m-%d %H:%M:%S.%f"),
        ),
        licenses=[
            dict(
                url=None,
                id=0,
                name=None,
            )
        ],
        images=[
            # license, url, file_name, height, width, date_captured, id
        ],
        type="instances",
        annotations=[
            # segmentation, area, iscrowd, image_id, bbox, category_id, id
        ],
        categories=[
            # supercategory, id, name
        ],
    )

    class_name_to_id = load_categories(args.labels)
    for class_name, class_id in class_name_to_id.items():
        data["categories"].append(
            dict(
                supercategory=None,
                id=class_id,
                name=class_name,
            )
        )

    out_ann_file = osp.join(args.output_dir, "annotations.json")
    label_files = find_label_files(args.input_dir)
    for image_id, filename in enumerate(label_files, start=1):
        print("Generating dataset from:", filename)

        label_file = labelme.LabelFile(filename=filename)

        relative_stem = relative_output_stem(filename, args.input_dir)
        out_img_file = osp.join(args.output_dir, "JPEGImages", relative_stem + ".jpg")
        ensure_parent(out_img_file)

        img = labelme.utils.img_data_to_arr(label_file.imageData)
        imgviz.io.imsave(out_img_file, img)
        data["images"].append(
            dict(
                license=0,
                url=None,
                file_name=osp.relpath(out_img_file, osp.dirname(out_ann_file)),
                height=img.shape[0],
                width=img.shape[1],
                date_captured=None,
                id=image_id,
            )
        )

        masks = {}  # for area
        segmentations = collections.defaultdict(list)  # for segmentation
        requires_rle = collections.defaultdict(bool)
        for shape in label_file.shapes:
            if not shape_points_are_finite(shape):
                print("Skipping shape with invalid coordinates in:", filename)
                continue
            points = shape["points"]
            label = shape["label"]
            group_id = shape.get("group_id")
            shape_type = shape.get("shape_type", "polygon")
            try:
                mask, _ = labelme.utils.shapes_to_label(
                    img.shape[:2], [shape], {label: 1}
                )
            except (AssertionError, OverflowError, TypeError, ValueError) as exc:
                print("Skipping invalid shape in {}: {}".format(filename, exc))
                continue
            mask = mask.astype(bool)
            if not mask.any():
                print("Skipping out-of-bounds or empty shape in:", filename)
                continue

            if group_id is None:
                group_id = uuid.uuid1()

            instance = (label, group_id)

            if instance in masks:
                masks[instance] = masks[instance] | mask
            else:
                masks[instance] = mask

            if shape_type == "polygon" and len(points) >= 3:
                points = np.asarray(points).flatten().tolist()
                segmentations[instance].append(points)
            else:
                requires_rle[instance] = True
        segmentations = dict(segmentations)

        for instance, mask in masks.items():
            cls_name, group_id = instance
            if cls_name not in class_name_to_id:
                continue
            cls_id = class_name_to_id[cls_name]

            mask = np.asfortranarray(mask.astype(np.uint8))
            encoded_mask = pycocotools.mask.encode(mask)
            area = float(pycocotools.mask.area(encoded_mask))
            bbox = pycocotools.mask.toBbox(encoded_mask).flatten().tolist()
            if (
                not mask.any()
                or not math.isfinite(area)
                or area <= 0
                or len(bbox) != 4
                or not all(math.isfinite(float(value)) for value in bbox)
                or float(bbox[2]) <= 0
                or float(bbox[3]) <= 0
            ):
                print("Skipping empty or invalid instance in:", filename)
                continue
            if requires_rle[instance]:
                encoded_mask["counts"] = encoded_mask["counts"].decode("ascii")
                segmentation = encoded_mask
            else:
                segmentation = segmentations[instance]

            data["annotations"].append(
                dict(
                    id=len(data["annotations"]) + 1,
                    image_id=image_id,
                    category_id=cls_id,
                    segmentation=segmentation,
                    area=area,
                    bbox=bbox,
                    iscrowd=0,
                )
            )

        if not args.noviz:
            viz = labelme.utils.img_arr_to_rgb(img)
            if masks:
                known_instances = [
                    (class_name_to_id[cnm], cnm, msk)
                    for (cnm, _), msk in masks.items()
                    if cnm in class_name_to_id
                ]
                if known_instances:
                    labels, captions, known_masks = zip(*known_instances)
                    viz = imgviz.instances2rgb(
                        image=viz,
                        labels=labels,
                        masks=known_masks,
                        captions=captions,
                        font_size=15,
                        line_width=2,
                    )
            out_viz_file = osp.join(
                args.output_dir, "Visualization", relative_stem + ".jpg"
            )
            ensure_parent(out_viz_file)
            imgviz.io.imsave(out_viz_file, viz)

    with open(out_ann_file, "w") as f:
        json.dump(data, f)


if __name__ == "__main__":
    main()
