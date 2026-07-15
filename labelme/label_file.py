import base64
import contextlib
import io
import json
import os
import os.path as osp
import tempfile

import numpy as np
import PIL.Image

from labelme import PY2
from labelme import QT4
from labelme import __version__
from labelme import utils
from labelme.logger import logger

PIL.Image.MAX_IMAGE_PIXELS = None


@contextlib.contextmanager
def open(name, mode):
    assert mode in ["r", "w"]
    if PY2:
        mode += "b"
        encoding = None
    else:
        encoding = "utf-8"
    with io.open(name, mode, encoding=encoding) as handle:
        yield handle


class LabelFileError(Exception):
    pass


class LabelFile(object):
    suffix = ".json"

    def __init__(self, filename=None):
        self.shapes = []
        self.imagePath = None
        self.imageData = None
        self.imageDataEmbedded = False
        self.imageHeight = None
        self.imageWidth = None
        self.flags = {}
        self.otherData = {}
        if filename is not None:
            self.load(filename)
        self.filename = filename

    @staticmethod
    def load_image_file(filename):
        if not (PY2 and QT4):
            try:
                with io.open(filename, "rb") as f:
                    image_data = f.read()
            except IOError:
                logger.error("Failed opening image file: {}".format(filename))
                return

            ext = osp.splitext(filename)[1].lower()
            if ext in [".jpg", ".jpeg"]:
                try:
                    image_pil = PIL.Image.open(io.BytesIO(image_data))
                except IOError:
                    logger.error("Failed opening image file: {}".format(filename))
                    return
                orientation = image_pil.getexif().get(274)
                if not orientation or orientation == 1:
                    return image_data
            elif ext in [
                ".bmp",
                ".gif",
                ".ico",
                ".png",
                ".pbm",
                ".pgm",
                ".ppm",
                ".tif",
                ".tiff",
                ".webp",
            ]:
                return image_data

        try:
            image_pil = PIL.Image.open(filename)
        except IOError:
            logger.error("Failed opening image file: {}".format(filename))
            return

        # apply orientation to image according to exif
        image_pil = utils.apply_exif_orientation(image_pil)

        with io.BytesIO() as f:
            ext = osp.splitext(filename)[1].lower()
            if PY2 and QT4:
                format = "PNG"
            elif ext in [".jpg", ".jpeg"]:
                format = "JPEG"
            else:
                format = "PNG"
            image_pil.save(f, format=format)
            f.seek(0)
            return f.read()

    def load(self, filename):
        keys = [
            "version",
            "imageData",
            "imagePath",
            "shapes",  # polygonal annotations
            "flags",  # image level flags
            "imageHeight",
            "imageWidth",
        ]
        try:
            with open(filename, "r") as f:
                data = json.load(f)

            imageDataEmbedded = data.get("imageData") is not None
            if imageDataEmbedded:
                imageData = base64.b64decode(data["imageData"])
                if PY2 and QT4:
                    imageData = utils.img_data_to_png_data(imageData)
            else:
                imageData = None
                json_dir = osp.dirname(filename)
                img_name = data["imagePath"]
                candidates = [
                    osp.join(json_dir, img_name),
                    osp.splitext(filename)[0] + ".jpg",
                ]
                dir_base = osp.basename(json_dir)
                if dir_base.startswith("labelme_"):
                    stripped = dir_base[len("labelme_") :]
                    # labelme_videoXXXX/videoXXXX/ subdirectory
                    candidates.append(osp.join(json_dir, stripped, img_name))
                    # videoXXXX/ sibling directory
                    candidates.append(
                        osp.join(osp.dirname(json_dir), stripped, img_name)
                    )
                for candidate in candidates:
                    if osp.isfile(candidate):
                        imageData = self.load_image_file(candidate)
                        if imageData is not None:
                            break
            flags = data.get("flags") or {}
            imagePath = data["imagePath"]
            imageHeight = data.get("imageHeight")
            imageWidth = data.get("imageWidth")
            if imageData is not None:
                imageHeight, imageWidth = self._check_image_height_and_width(
                    imageData,
                    imageHeight,
                    imageWidth,
                )
            shape_keys = {
                "label",
                "points",
                "shape_type",
                "flags",
                "description",
                "group_id",
                "track_id",
                "mask",
            }
            shapes = []
            for shape_data in data["shapes"]:
                track_id = shape_data.get("track_id")
                if track_id is None or track_id == "":
                    track_id = shape_data.get("group_id")
                shapes.append(
                    dict(
                        label=shape_data["label"],
                        points=shape_data["points"],
                        shape_type=shape_data.get("shape_type", "polygon"),
                        flags=shape_data.get("flags", {}),
                        description=shape_data.get("description"),
                        group_id=shape_data.get("group_id"),
                        track_id=track_id,
                        mask=(
                            utils.img_b64_to_arr(shape_data["mask"])
                            if shape_data.get("mask")
                            else None
                        ),
                        other_data={
                            key: value
                            for key, value in shape_data.items()
                            if key not in shape_keys
                        },
                    )
                )
        except Exception as e:
            raise LabelFileError(e)

        otherData = {}
        for key, value in data.items():
            if key not in keys:
                otherData[key] = value

        # Only replace data after everything is loaded.
        self.flags = flags
        self.shapes = shapes
        self.imagePath = imagePath
        self.imageData = imageData
        self.imageDataEmbedded = imageDataEmbedded
        self.imageHeight = imageHeight
        self.imageWidth = imageWidth
        self.filename = filename
        self.otherData = otherData

    @staticmethod
    def _check_image_height_and_width(imageData, imageHeight, imageWidth):
        with PIL.Image.open(io.BytesIO(imageData)) as img:
            img.load()
            w, h = img.size
        if imageHeight is None:
            imageHeight = h
        elif h != imageHeight:
            logger.error(
                "imageHeight does not match with imageData or imagePath, "
                "so getting imageHeight from actual image."
            )
            imageHeight = h
        if imageWidth is None:
            imageWidth = w
        elif w != imageWidth:
            logger.error(
                "imageWidth does not match with imageData or imagePath, "
                "so getting imageWidth from actual image."
            )
            imageWidth = w
        return imageHeight, imageWidth

    def save(
        self,
        filename,
        shapes,
        imagePath,
        imageHeight,
        imageWidth,
        imageData=None,
        otherData=None,
        flags=None,
    ):
        raw_image_data = imageData
        if imageData is not None:
            imageHeight, imageWidth = self._check_image_height_and_width(
                imageData, imageHeight, imageWidth
            )
            imageData = base64.b64encode(imageData).decode("utf-8")
        if otherData is None:
            otherData = {}
        if flags is None:
            flags = {}
        serialized_shapes = []
        for shape in shapes:
            shape = dict(shape)
            shape_other_data = shape.pop("other_data", None) or {}
            mask = shape.get("mask")
            if mask is not None and not isinstance(mask, str):
                shape["mask"] = utils.img_arr_to_b64(np.asarray(mask))
            serialized_shape = dict(shape_other_data)
            serialized_shape.update(shape)
            serialized_shapes.append(serialized_shape)

        data = dict(
            version=__version__,
            flags=flags,
            shapes=serialized_shapes,
            imagePath=imagePath,
            imageData=imageData,
            imageHeight=imageHeight,
            imageWidth=imageWidth,
        )
        for key, value in otherData.items():
            if key in data:
                raise LabelFileError(
                    "Top-level metadata cannot replace reserved key: {}".format(key)
                )
            data[key] = value
        directory = osp.dirname(osp.abspath(filename))
        temporary_path = None
        try:
            os.makedirs(directory, exist_ok=True)
            fd, temporary_path = tempfile.mkstemp(
                dir=directory,
                prefix=".{}-".format(osp.basename(filename)),
                suffix=".tmp",
            )
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(
                    data,
                    handle,
                    ensure_ascii=False,
                    indent=2,
                    allow_nan=False,
                )
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary_path, filename)
            temporary_path = None
            self.filename = filename
            self.shapes = list(shapes)
            self.imagePath = imagePath
            self.imageData = raw_image_data
            self.imageDataEmbedded = raw_image_data is not None
            self.imageHeight = imageHeight
            self.imageWidth = imageWidth
            self.flags = flags
            self.otherData = otherData
        except Exception as e:
            raise LabelFileError(e)
        finally:
            if temporary_path is not None:
                try:
                    os.unlink(temporary_path)
                except OSError:
                    pass

    @staticmethod
    def is_label_file(filename):
        return osp.splitext(filename)[1].lower() == LabelFile.suffix


def save_label_files_atomically(requests):
    """Stage and commit label files, including rollback-safe legacy retirement.

    A request may include the internal ``_retire_sources`` list. Those sources
    are moved aside only after every destination has been installed, and are
    restored if any later part of the transaction fails.
    """
    if not requests:
        return
    staged = []
    committed = []
    retired = []
    try:
        destinations = [osp.abspath(request["filename"]) for request in requests]
        if len(destinations) != len(set(destinations)):
            raise LabelFileError("Batch contains duplicate annotation destinations")
        retirement_sources = []
        for request in requests:
            sources = request.get("_retire_sources", [])
            if isinstance(sources, (str, bytes)):
                sources = [sources]
            for source in sources:
                source = osp.abspath(source)
                if source != osp.abspath(request["filename"]):
                    retirement_sources.append(source)
        if len(retirement_sources) != len(set(retirement_sources)):
            raise LabelFileError("Batch contains duplicate legacy annotation sources")
        if set(retirement_sources) & set(destinations):
            raise LabelFileError(
                "A legacy annotation source cannot also be a batch destination"
            )

        for request, destination in zip(requests, destinations):
            directory = osp.dirname(destination)
            os.makedirs(directory, exist_ok=True)
            fd, stage_path = tempfile.mkstemp(
                dir=directory,
                prefix=".{}-stage-".format(osp.basename(destination)),
                suffix=".json",
            )
            os.close(fd)
            os.unlink(stage_path)
            arguments = dict(request)
            arguments.pop("_retire_sources", None)
            arguments["filename"] = stage_path
            LabelFile().save(**arguments)
            staged.append((stage_path, destination))

        for stage_path, destination in staged:
            backup_path = None
            if osp.exists(destination):
                fd, backup_path = tempfile.mkstemp(
                    dir=osp.dirname(destination),
                    prefix=".{}-backup-".format(osp.basename(destination)),
                    suffix=".json",
                )
                os.close(fd)
                os.unlink(backup_path)
                os.replace(destination, backup_path)
            record = [destination, backup_path, False]
            committed.append(record)
            os.replace(stage_path, destination)
            record[2] = True

        for source in retirement_sources:
            if not osp.exists(source):
                continue
            fd, retired_path = tempfile.mkstemp(
                dir=osp.dirname(source),
                prefix=".{}-migrated-".format(osp.basename(source)),
                suffix=".json",
            )
            os.close(fd)
            os.unlink(retired_path)
            os.replace(source, retired_path)
            retired.append((source, retired_path))

        for _, backup_path, _ in committed:
            if backup_path and osp.exists(backup_path):
                try:
                    os.unlink(backup_path)
                except OSError:
                    logger.warning("Could not remove annotation backup %r", backup_path)
        for _, retired_path in retired:
            if osp.exists(retired_path):
                try:
                    os.unlink(retired_path)
                except OSError:
                    logger.warning(
                        "Could not remove migrated annotation backup %r", retired_path
                    )
    except Exception as exc:
        for source, retired_path in reversed(retired):
            try:
                if osp.exists(retired_path):
                    os.replace(retired_path, source)
            except OSError:
                logger.exception("Failed restoring legacy annotation %r", source)
        for destination, backup_path, installed in reversed(committed):
            try:
                if installed and osp.exists(destination):
                    os.unlink(destination)
                if backup_path and osp.exists(backup_path):
                    os.replace(backup_path, destination)
            except OSError:
                logger.exception("Failed rolling back annotation %r", destination)
        if isinstance(exc, LabelFileError):
            raise
        raise LabelFileError(exc)
    finally:
        for stage_path, _ in staged:
            if osp.exists(stage_path):
                try:
                    os.unlink(stage_path)
                except OSError:
                    pass
