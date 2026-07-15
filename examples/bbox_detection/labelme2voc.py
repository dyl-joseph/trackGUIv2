#!/usr/bin/env python

from __future__ import print_function

import argparse
import glob
import math
import os
import os.path as osp
import sys
import xml.etree.ElementTree as ElementTree

import imgviz
import numpy as np

import labelme


def find_label_files(input_dir):
    pattern = osp.join(osp.abspath(input_dir), "**", "*.json")
    files = glob.glob(pattern, recursive=True)
    return sorted(
        filename
        for filename in files
        if not any(
            part.startswith(".")
            for part in osp.relpath(filename, input_dir).replace("\\", "/").split("/")
        )
    )


def relative_output_stem(filename, input_dir):
    relative = osp.relpath(osp.abspath(filename), osp.abspath(input_dir))
    return osp.splitext(relative)[0]


def ensure_parent(filename):
    os.makedirs(osp.dirname(filename), exist_ok=True)


def normalized_clipped_rectangle(points, image_width, image_height):
    """Return integer pixel bounds using floor(min), ceil(max), then clipping."""
    if image_width <= 0 or image_height <= 0:
        raise ValueError("image dimensions must be positive")
    try:
        point_values = list(points)
    except TypeError as exc:
        raise ValueError("rectangle points must be a sequence") from exc
    if len(point_values) != 2 or any(
        not isinstance(point, (list, tuple)) or len(point) != 2
        for point in point_values
    ):
        raise ValueError("rectangle must contain exactly two coordinate pairs")
    if any(isinstance(value, bool) for point in point_values for value in point):
        raise ValueError("rectangle coordinates must be finite numbers")
    try:
        coordinates = np.asarray(point_values, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError("rectangle coordinates must be finite numbers") from exc
    if not np.isfinite(coordinates).all():
        raise ValueError("rectangle coordinates must be finite numbers")

    (x1, y1), (x2, y2) = coordinates
    xmin = max(0, math.floor(min(x1, x2)))
    ymin = max(0, math.floor(min(y1, y2)))
    xmax = min(image_width, math.ceil(max(x1, x2)))
    ymax = min(image_height, math.ceil(max(y1, y2)))
    if xmax <= xmin or ymax <= ymin:
        raise ValueError("rectangle has no positive in-image area")
    return xmin, ymin, xmax, ymax


def add_xml_text(parent, tag, value=""):
    element = ElementTree.SubElement(parent, tag)
    element.text = str(value)
    return element


def main():
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
    os.makedirs(osp.join(args.output_dir, "Annotations"))
    if not args.noviz:
        os.makedirs(osp.join(args.output_dir, "AnnotationsVisualization"))
    print("Creating dataset:", args.output_dir)

    class_names = []
    class_name_to_id = {}
    with open(args.labels, encoding="utf-8") as handle:
        label_lines = list(handle)
    for i, line in enumerate(label_lines):
        class_id = i - 1  # starts with -1
        class_name = line.strip()
        class_name_to_id[class_name] = class_id
        if class_id == -1:
            assert class_name == "__ignore__"
            continue
        elif class_id == 0:
            assert class_name == "_background_"
        class_names.append(class_name)
    class_names = tuple(class_names)
    print("class_names:", class_names)
    out_class_names_file = osp.join(args.output_dir, "class_names.txt")
    with open(out_class_names_file, "w") as f:
        f.writelines("\n".join(class_names))
    print("Saved class_names:", out_class_names_file)

    for filename in find_label_files(args.input_dir):
        print("Generating dataset from:", filename)

        label_file = labelme.LabelFile(filename=filename)

        relative_stem = relative_output_stem(filename, args.input_dir)
        relative_image_name = (relative_stem + ".jpg").replace(os.sep, "/")
        out_img_file = osp.join(args.output_dir, "JPEGImages", relative_stem + ".jpg")
        out_xml_file = osp.join(args.output_dir, "Annotations", relative_stem + ".xml")
        if not args.noviz:
            out_viz_file = osp.join(
                args.output_dir,
                "AnnotationsVisualization",
                relative_stem + ".jpg",
            )
        ensure_parent(out_img_file)
        ensure_parent(out_xml_file)
        if not args.noviz:
            ensure_parent(out_viz_file)

        img = labelme.utils.img_data_to_arr(label_file.imageData)
        imgviz.io.imsave(out_img_file, img)

        xml = ElementTree.Element("annotation")
        add_xml_text(xml, "folder")
        add_xml_text(xml, "filename", relative_image_name)
        add_xml_text(xml, "database")
        add_xml_text(xml, "annotation")
        add_xml_text(xml, "image")
        size_element = ElementTree.SubElement(xml, "size")
        add_xml_text(size_element, "height", img.shape[0])
        add_xml_text(size_element, "width", img.shape[1])
        add_xml_text(size_element, "depth", labelme.utils.img_arr_channel_count(img))
        add_xml_text(xml, "segmented")

        bboxes = []
        labels = []
        for shape in label_file.shapes:
            if shape["shape_type"] != "rectangle":
                print(
                    "Skipping shape: label={label}, shape_type={shape_type}".format(
                        **shape
                    )
                )
                continue

            class_name = shape["label"]
            if class_name not in class_names:
                print("Skipping shape with unknown label:", class_name)
                continue
            class_id = class_names.index(class_name)
            try:
                xmin, ymin, xmax, ymax = normalized_clipped_rectangle(
                    shape.get("points"), img.shape[1], img.shape[0]
                )
            except ValueError as exc:
                print("Skipping invalid rectangle in {}: {}".format(filename, exc))
                continue

            bboxes.append([ymin, xmin, ymax, xmax])
            labels.append(class_id)

            object_element = ElementTree.SubElement(xml, "object")
            add_xml_text(object_element, "name", shape["label"])
            add_xml_text(object_element, "pose")
            add_xml_text(object_element, "truncated")
            add_xml_text(object_element, "difficult")
            bbox_element = ElementTree.SubElement(object_element, "bndbox")
            add_xml_text(bbox_element, "xmin", xmin)
            add_xml_text(bbox_element, "ymin", ymin)
            add_xml_text(bbox_element, "xmax", xmax)
            add_xml_text(bbox_element, "ymax", ymax)

        if not args.noviz:
            captions = [class_names[label] for label in labels]
            viz = labelme.utils.img_arr_to_rgb(img)
            if bboxes:
                viz = imgviz.instances2rgb(
                    image=viz,
                    labels=labels,
                    bboxes=bboxes,
                    captions=captions,
                    font_size=15,
                )
            imgviz.io.imsave(out_viz_file, viz)

        tree = ElementTree.ElementTree(xml)
        ElementTree.indent(tree, space="  ")
        tree.write(out_xml_file, encoding="utf-8", xml_declaration=True)


if __name__ == "__main__":
    main()
