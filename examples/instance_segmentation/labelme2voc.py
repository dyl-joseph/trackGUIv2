#!/usr/bin/env python

from __future__ import print_function

import argparse
import glob
import os
import os.path as osp
import sys

import imgviz
import numpy as np

import labelme


def find_label_files(input_dir):
    pattern = osp.join(osp.abspath(input_dir), "**", "*.json")
    return sorted(glob.glob(pattern, recursive=True))


def relative_output_stem(filename, input_dir):
    relative = osp.relpath(osp.abspath(filename), osp.abspath(input_dir))
    return osp.splitext(relative)[0]


def ensure_parent(filename):
    os.makedirs(osp.dirname(filename), exist_ok=True)


def valid_shapes_for_image(shapes, img_shape, label_name_to_value, filename):
    valid_shapes = []
    for shape in shapes:
        label = shape.get("label")
        if label not in label_name_to_value:
            print("Skipping unknown label {!r} in: {}".format(label, filename))
            continue
        try:
            points = np.asarray(shape.get("points", []), dtype=float)
        except (TypeError, ValueError):
            points = np.empty(0)
        if (
            points.ndim != 2
            or points.shape[1:] != (2,)
            or not np.isfinite(points).all()
        ):
            print("Skipping shape with invalid coordinates in:", filename)
            continue
        try:
            _, probe_instances = labelme.utils.shapes_to_label(
                img_shape=img_shape,
                shapes=[shape],
                label_name_to_value={label: 1},
            )
        except (AssertionError, OverflowError, TypeError, ValueError) as exc:
            print("Skipping invalid shape in {}: {}".format(filename, exc))
            continue
        if not probe_instances.any():
            print("Skipping out-of-bounds or empty shape in:", filename)
            continue
        valid_shapes.append(shape)
    return valid_shapes


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("input_dir", help="Input annotated directory")
    parser.add_argument("output_dir", help="Output dataset directory")
    parser.add_argument(
        "--labels", help="Labels file or comma separated text", required=True
    )
    parser.add_argument(
        "--noobject", help="Flag not to generate object label", action="store_true"
    )
    parser.add_argument(
        "--nonpy", help="Flag not to generate .npy files", action="store_true"
    )
    parser.add_argument(
        "--noviz", help="Flag to disable visualization", action="store_true"
    )
    args = parser.parse_args()

    if osp.exists(args.output_dir):
        print("Output directory already exists:", args.output_dir)
        sys.exit(1)
    os.makedirs(args.output_dir)
    os.makedirs(osp.join(args.output_dir, "JPEGImages"))
    os.makedirs(osp.join(args.output_dir, "SegmentationClass"))
    if not args.nonpy:
        os.makedirs(osp.join(args.output_dir, "SegmentationClassNpy"))
    if not args.noviz:
        os.makedirs(osp.join(args.output_dir, "SegmentationClassVisualization"))
    if not args.noobject:
        os.makedirs(osp.join(args.output_dir, "SegmentationObject"))
        if not args.nonpy:
            os.makedirs(osp.join(args.output_dir, "SegmentationObjectNpy"))
        if not args.noviz:
            os.makedirs(osp.join(args.output_dir, "SegmentationObjectVisualization"))
    print("Creating dataset:", args.output_dir)

    if osp.exists(args.labels):
        with open(args.labels) as f:
            labels = [label.strip() for label in f if label]
    else:
        labels = [label.strip() for label in args.labels.split(",")]

    class_names = []
    class_name_to_id = {}
    for i, label in enumerate(labels):
        class_id = i - 1  # starts with -1
        class_name = label.strip()
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
        out_img_file = osp.join(args.output_dir, "JPEGImages", relative_stem + ".jpg")
        out_clsp_file = osp.join(
            args.output_dir, "SegmentationClass", relative_stem + ".png"
        )
        if not args.nonpy:
            out_cls_file = osp.join(
                args.output_dir, "SegmentationClassNpy", relative_stem + ".npy"
            )
        if not args.noviz:
            out_clsv_file = osp.join(
                args.output_dir,
                "SegmentationClassVisualization",
                relative_stem + ".jpg",
            )
        if not args.noobject:
            out_insp_file = osp.join(
                args.output_dir, "SegmentationObject", relative_stem + ".png"
            )
            if not args.nonpy:
                out_ins_file = osp.join(
                    args.output_dir,
                    "SegmentationObjectNpy",
                    relative_stem + ".npy",
                )
            if not args.noviz:
                out_insv_file = osp.join(
                    args.output_dir,
                    "SegmentationObjectVisualization",
                    relative_stem + ".jpg",
                )

        output_files = [out_img_file, out_clsp_file]
        if not args.nonpy:
            output_files.append(out_cls_file)
        if not args.noviz:
            output_files.append(out_clsv_file)
        if not args.noobject:
            output_files.append(out_insp_file)
            if not args.nonpy:
                output_files.append(out_ins_file)
            if not args.noviz:
                output_files.append(out_insv_file)
        for output_file in output_files:
            ensure_parent(output_file)

        img = labelme.utils.img_data_to_arr(label_file.imageData)
        imgviz.io.imsave(out_img_file, img)

        shapes = valid_shapes_for_image(
            label_file.shapes,
            img.shape,
            class_name_to_id,
            filename,
        )
        cls, ins = labelme.utils.shapes_to_label(
            img_shape=img.shape,
            shapes=shapes,
            label_name_to_value=class_name_to_id,
        )
        ins[cls == -1] = 0  # ignore it.

        # class label
        labelme.utils.lblsave(out_clsp_file, cls)
        if not args.nonpy:
            np.save(out_cls_file, cls)
        if not args.noviz:
            clsv = imgviz.label2rgb(
                cls,
                imgviz.asgray(img),
                label_names=class_names,
                font_size=15,
                loc="rb",
            )
            imgviz.io.imsave(out_clsv_file, clsv)

        if not args.noobject:
            # instance label
            labelme.utils.lblsave(out_insp_file, ins)
            if not args.nonpy:
                np.save(out_ins_file, ins)
            if not args.noviz:
                instance_ids = np.unique(ins)
                instance_names = [str(i) for i in range(max(instance_ids) + 1)]
                insv = imgviz.label2rgb(
                    ins,
                    imgviz.asgray(img),
                    label_names=instance_names,
                    font_size=15,
                    loc="rb",
                )
                imgviz.io.imsave(out_insv_file, insv)


if __name__ == "__main__":
    main()
