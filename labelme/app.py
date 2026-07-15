# -*- coding: utf-8 -*-

import functools
import hashlib
import html
import math
import os
import os.path as osp
import re
import threading
import webbrowser
from collections import OrderedDict

import cv2  # noqa: F401
import imgviz
import natsort
import numpy as np
from qtpy import QtCore
from qtpy import QtGui
from qtpy import QtWidgets
from qtpy.QtCore import Qt
from scipy.optimize import linear_sum_assignment

from labelme import PY2
from labelme import __appname__
from labelme.ai import MODELS
from labelme.annotation_path import canonical_annotation_path
from labelme.annotation_path import legacy_annotation_paths
from labelme.annotation_path import resolve_annotation_path
from labelme.config import get_config
from labelme.hosted_sam2_client import HostedSam2Client
from labelme.hosted_sam2_client import HostedSam2Error
from labelme.label_file import LabelFile
from labelme.label_file import LabelFileError
from labelme.label_file import save_label_files_atomically
from labelme.logger import logger
from labelme.shape import Shape
from labelme.track_algo import KalmanBoxTracker
from labelme.track_algo import SORT_main
from labelme.tracking_utils import interpolation_indices
from labelme.tracking_utils import intersect_xyxy_with_image
from labelme.tracking_utils import load_oriented_cv_image
from labelme.tracking_utils import normalized_rectangle_points
from labelme.tracking_utils import prediction_to_clamped_rectangle
from labelme.tracking_utils import shape_track_id
from labelme.tracking_utils import upsert_tracked_rectangle
from labelme.widgets import BrightnessContrastDialog
from labelme.widgets import Canvas
from labelme.widgets import DeletionDialog
from labelme.widgets import FileDialogPreview
from labelme.widgets import IDDialog
from labelme.widgets import IDListWidget
from labelme.widgets import IDListWidgetItem
from labelme.widgets import InterpolationDialog
from labelme.widgets import InterpolationRefineInfo_Dialog
from labelme.widgets import IterpolationRefineWidget
from labelme.widgets import LabelDialog
from labelme.widgets import LabelListWidget
from labelme.widgets import LabelListWidgetItem
from labelme.widgets import NavigationWidget
from labelme.widgets import ToolBar
from labelme.widgets import TrackDialog
from labelme.widgets import UniqueLabelQListWidget
from labelme.widgets import ZoomWidget

from . import utils

# FIXME
# - [medium] Set max zoom value to something big enough for FitWidth/Window

# TODO(unknown):
# - Zoom is too "steppy".


LABEL_COLORMAP = imgviz.label_colormap()


class HostedSam2RequestWorker(QtCore.QObject):
    finished = QtCore.Signal(object)
    failed = QtCore.Signal(object)

    def __init__(self, function, *args, **kwargs):
        super().__init__()
        self._function = function
        self._args = args
        self._kwargs = kwargs

    @QtCore.Slot()
    def run(self):
        try:
            self.finished.emit(self._function(*self._args, **self._kwargs))
        except HostedSam2Error as exc:
            self.failed.emit(exc)
        except Exception as exc:
            logger.exception("Hosted SAM2 request failed")
            self.failed.emit(exc)


class CancellableTrackingWorker(QtCore.QObject):
    finished = QtCore.Signal(object)
    failed = QtCore.Signal(object)
    progress = QtCore.Signal(int, str)

    def __init__(self, function):
        super().__init__()
        self._function = function
        self._cancel_event = threading.Event()

    def cancel(self):
        self._cancel_event.set()

    def _reportProgress(self, value, message):
        self.progress.emit(int(value), str(message))

    @QtCore.Slot()
    def run(self):
        try:
            result = self._function(self._cancel_event, self._reportProgress)
        except Exception as exc:
            logger.exception("Forward tracking failed")
            self.failed.emit(exc)
        else:
            self.finished.emit(result)


class MainWindow(QtWidgets.QMainWindow):
    FIT_WINDOW, FIT_WIDTH, MANUAL_ZOOM = 0, 1, 2
    THEME_SYSTEM, THEME_LIGHT, THEME_DARK = "system", "light", "dark"

    def __init__(
        self,
        config=None,
        filename=None,
        output=None,
        output_file=None,
        output_dir=None,
    ):
        if output is not None:
            logger.warning("argument output is deprecated, use output_file instead")
            if output_file is None:
                output_file = output

        # see labelme/config/default_config.yaml for valid configuration
        if config is None:
            config = get_config()
        self._config = config

        # set default shape colors
        Shape.line_color = QtGui.QColor(*self._config["shape"]["line_color"])
        Shape.fill_color = QtGui.QColor(*self._config["shape"]["fill_color"])
        Shape.select_line_color = QtGui.QColor(
            *self._config["shape"]["select_line_color"]
        )
        Shape.select_fill_color = QtGui.QColor(
            *self._config["shape"]["select_fill_color"]
        )
        Shape.vertex_fill_color = QtGui.QColor(
            *self._config["shape"]["vertex_fill_color"]
        )
        Shape.hvertex_fill_color = QtGui.QColor(
            *self._config["shape"]["hvertex_fill_color"]
        )

        # Set point size from config file
        Shape.point_size = self._config["shape"]["point_size"]

        super(MainWindow, self).__init__()
        self.setWindowTitle(__appname__)

        # Whether we need to save or not.
        self.dirty = False

        self._noSelectionSlot = False
        self._syncing_visibility = False

        self._copied_shapes = None
        self._hosted_sam2_client = HostedSam2Client.from_config(self._config)
        self._hosted_sam2_image_cache = OrderedDict()
        self._hosted_sam2_thread = None
        self._hosted_sam2_worker = None
        self._hosted_sam2_request_active = False
        self._hosted_sam2_request_context = None
        self._hosted_sam2_on_success = None
        self._hosted_sam2_cache_frames = bool(
            self._config.get("hosted_sam2", {}).get("cache_frames", True)
        )
        self._hosted_sam2_max_cached_frames = 8
        self._close_after_hosted_request = False
        self._tracking_thread = None
        self._tracking_worker = None
        self._tracking_progress = None
        self._tracking_context = None
        self._tracking_request_active = False
        self._close_after_tracking = False

        # Main widgets and related state.
        self.labelDialog = LabelDialog(
            parent=self,
            labels=self._config["labels"],
            sort_labels=self._config["sort_labels"],
            show_text_field=self._config["show_label_text_field"],
            completion=self._config["label_completion"],
            fit_to_content=self._config["fit_to_content"],
            flags=self._config["label_flags"],
        )
        self.labelDialog.edit_group_id.setPlaceholderText("Track ID")

        self.IDDialog = IDDialog(
            parent=self,
            ids=[],
            sort_ids=self._config["sort_labels"],
            show_text_field=self._config["show_label_text_field"],
            completion=self._config["label_completion"],
            fit_to_content=self._config["fit_to_content"],
        )

        self.labelList = LabelListWidget()
        self.lastOpenDir = None

        self.flag_dock = self.flag_widget = None
        self.flag_dock = QtWidgets.QDockWidget(self.tr("Flags"), self)
        self.flag_dock.setObjectName("Flags")
        self.flag_widget = QtWidgets.QListWidget()
        if config["flags"]:
            self.loadFlags({k: False for k in config["flags"]})
        self.flag_dock.setWidget(self.flag_widget)
        self.flag_widget.itemChanged.connect(self.setDirty)

        self.mode = "None"
        self.list_length = ""
        self.start_INP0 = 0
        self.end_INP0 = 0
        self.interval_INPO = 0
        self.ID_INPO = ""
        self.label_INPO = ""
        self.INTERPOLATION_list = []
        self.INTERPOLATION_indices = []
        self.INTERPOLATION_filename = None
        self.navigation_list = NavigationWidget()
        self.navigation_list.button1.clicked.connect(self.OKAY)
        self.navigation_list.button2.clicked.connect(self.openNextImg)
        self.navigation_list.button3.clicked.connect(self.openPrevImg)
        self.navigation_dock = QtWidgets.QDockWidget(self.tr("Navigation"), self)
        self.navigation_dock.setObjectName("Navigation")
        self.navigation_dock.setWidget(self.navigation_list)

        self.interpolationrefine_list = IterpolationRefineWidget()
        self.interpolationrefine_list.button.clicked.connect(self.editIR_info)
        self.interpolationrefine_dock = QtWidgets.QDockWidget(
            self.tr("Interpolation Refinement"), self
        )
        self.interpolationrefine_dock.setObjectName("Interpolation Refinement")
        self.interpolationrefine_dock.setWidget(self.interpolationrefine_list)
        self.ir_name = "None"
        self.ir_id = "None"
        self.ir_old_shapes = []
        self.ir_old_shape = "None"
        self.ir_mod_shape = "None"
        self.ir_activated = False
        # self.interpolationrefine_list.checkBox.isChecked()

        self.labelList.itemSelectionChanged.connect(self.labelSelectionChanged)
        self.labelList.itemDoubleClicked.connect(self.editLabel)
        self.labelList.itemChanged.connect(self.labelItemChanged)
        self.labelList.itemDropped.connect(self.labelOrderChanged)
        self.shape_dock = QtWidgets.QDockWidget(self.tr("Polygon Labels"), self)
        self.shape_dock.setObjectName("Labels")
        self.shape_dock.setWidget(self.labelList)

        self.IDList = IDListWidget()
        self.IDList.itemSelectionChanged.connect(self.IDSelectionChanged)
        self.IDList.itemDoubleClicked.connect(self.editID)
        self.IDList.itemChanged.connect(self.IDItemChanged)
        self.IDList.itemDropped.connect(self.IDOrderChanged)
        self.id_dock = QtWidgets.QDockWidget(self.tr("Polygon IDs"), self)
        self.id_dock.setObjectName("IDs")
        self.id_dock.setWidget(self.IDList)

        self.uniqLabelList = UniqueLabelQListWidget()
        self.uniqLabelList.setToolTip(
            self.tr("Select label to start annotating for it. Press 'Esc' to deselect.")
        )
        if self._config["labels"]:
            for label in self._config["labels"]:
                item = self.uniqLabelList.createItemFromLabel(label)
                self.uniqLabelList.addItem(item)
                rgb = self._get_rgb_by_label(label)
                self.uniqLabelList.setItemLabel(item, label, rgb)
        self.label_dock = QtWidgets.QDockWidget(self.tr("Label List"), self)
        self.label_dock.setObjectName("Label List")
        self.label_dock.setWidget(self.uniqLabelList)

        self.fileSearch = QtWidgets.QLineEdit()
        self.fileSearch.setPlaceholderText(self.tr("Search Filename"))
        self.fileSearch.textChanged.connect(self.fileSearchChanged)
        self.fileListWidget = QtWidgets.QListWidget()
        self.fileListWidget.itemSelectionChanged.connect(self.fileSelectionChanged)
        fileListLayout = QtWidgets.QVBoxLayout()
        fileListLayout.setContentsMargins(0, 0, 0, 0)
        fileListLayout.setSpacing(0)
        fileListLayout.addWidget(self.fileSearch)
        fileListLayout.addWidget(self.fileListWidget)
        self.file_dock = QtWidgets.QDockWidget(self.tr("File List"), self)
        self.file_dock.setObjectName("Files")
        fileListWidget = QtWidgets.QWidget()
        fileListWidget.setLayout(fileListLayout)
        self.file_dock.setWidget(fileListWidget)

        self.zoomWidget = ZoomWidget()
        self.setAcceptDrops(True)

        self.canvas = self.labelList.canvas = Canvas(
            epsilon=self._config["epsilon"],
            double_click=self._config["canvas"]["double_click"],
            num_backups=self._config["canvas"]["num_backups"],
            crosshair=self._config["canvas"]["crosshair"],
        )
        self.canvas.zoomRequest.connect(self.zoomRequest)

        scrollArea = QtWidgets.QScrollArea()
        scrollArea.setWidget(self.canvas)
        scrollArea.setWidgetResizable(True)
        self.scrollBars = {
            Qt.Vertical: scrollArea.verticalScrollBar(),
            Qt.Horizontal: scrollArea.horizontalScrollBar(),
        }
        self.canvas.scrollRequest.connect(self.scrollRequest)

        self.canvas.newShape.connect(self.newShape)
        self.canvas.shapeMoved.connect(self.setDirty)
        self.canvas.selectionChanged.connect(self.shapeSelectionChanged)
        self.canvas.drawingPolygon.connect(self.toggleDrawingSensitive)
        self.canvas.pointPromptRequested.connect(self._hostedSam2PointPrompt)
        self.canvas.pointPromptCancelled.connect(self._hostedSam2PointPromptCancelled)
        self.canvas.aiPredictionFailed.connect(
            lambda message: self.status(message, delay=8000)
        )

        self._theme_mode = self._config.get("theme", self.THEME_SYSTEM)
        self._theme_action_group = QtWidgets.QActionGroup(self)
        self._theme_action_group.setExclusive(True)

        self.setCentralWidget(scrollArea)

        for dock in ["flag_dock", "label_dock", "shape_dock", "file_dock"]:
            features = QtWidgets.QDockWidget.DockWidgetFeatures()
            if self._config[dock]["closable"]:
                features = features | QtWidgets.QDockWidget.DockWidgetClosable
            if self._config[dock]["floatable"]:
                features = features | QtWidgets.QDockWidget.DockWidgetFloatable
            if self._config[dock]["movable"]:
                features = features | QtWidgets.QDockWidget.DockWidgetMovable
            getattr(self, dock).setFeatures(features)
            if self._config[dock]["show"] is False:
                getattr(self, dock).setVisible(False)
        self.id_dock.setFeatures(self.shape_dock.features())
        self.id_dock.setVisible(self._config["shape_dock"]["show"])

        self.addDockWidget(Qt.RightDockWidgetArea, self.navigation_dock)
        self.addDockWidget(Qt.RightDockWidgetArea, self.interpolationrefine_dock)
        self.addDockWidget(Qt.RightDockWidgetArea, self.flag_dock)
        self.addDockWidget(Qt.RightDockWidgetArea, self.label_dock)
        self.addDockWidget(Qt.RightDockWidgetArea, self.shape_dock)
        self.addDockWidget(Qt.RightDockWidgetArea, self.id_dock)
        self.addDockWidget(Qt.RightDockWidgetArea, self.file_dock)

        # Actions
        action = functools.partial(utils.newAction, self)
        shortcuts = self._config["shortcuts"]
        quit = action(
            self.tr("&Quit"),
            self.close,
            shortcuts["quit"],
            "quit",
            self.tr("Quit application"),
        )
        open_ = action(
            self.tr("&Open\n"),
            self.openFile,
            shortcuts["open"],
            "open",
            self.tr("Open image or label file"),
        )
        opendir = action(
            self.tr("Open Dir"),
            self.openDirDialog,
            shortcuts["open_dir"],
            "open",
            self.tr("Open Dir"),
        )
        openNextImg = action(
            self.tr("&Next Image"),
            self.openNextImg,
            shortcuts["open_next"],
            "next",
            self.tr("Open next (hold Ctl+Shift to copy labels)"),
            enabled=False,
        )
        openPrevImg = action(
            self.tr("&Prev Image"),
            self.openPrevImg,
            shortcuts["open_prev"],
            "prev",
            self.tr("Open prev (hold Ctl+Shift to copy labels)"),
            enabled=False,
        )
        save = action(
            self.tr("&Save\n"),
            self.saveFile,
            shortcuts["save"],
            "save",
            self.tr("Save labels to file"),
            enabled=False,
        )
        saveAs = action(
            self.tr("&Save As"),
            self.saveFileAs,
            shortcuts["save_as"],
            "save-as",
            self.tr("Save labels to a different file"),
            enabled=False,
        )

        deleteFile = action(
            self.tr("&Delete File"),
            self.deleteFile,
            shortcuts["delete_file"],
            "delete",
            self.tr("Delete current label file"),
            enabled=False,
        )

        changeOutputDir = action(
            self.tr("&Change Output Dir"),
            slot=self.changeOutputDirDialog,
            shortcut=shortcuts["save_to"],
            icon="open",
            tip=self.tr("Change where annotations are loaded/saved"),
        )

        saveAuto = action(
            text=self.tr("Save &Automatically"),
            slot=self.enableAutoSave,
            icon="save",
            tip=self.tr("Save automatically"),
            checkable=True,
            enabled=True,
        )
        saveAuto.setChecked(self._config["auto_save"])

        saveWithImageData = action(
            text="Save With Image Data",
            slot=self.enableSaveImageWithData,
            tip="Save image data in label file",
            checkable=True,
            checked=self._config["store_data"],
        )

        close = action(
            "&Close",
            self.closeFile,
            shortcuts["close"],
            "close",
            "Close current file",
        )

        toggle_keep_prev_mode = action(
            self.tr("Keep Previous Annotation"),
            self.toggleKeepPrevMode,
            shortcuts["toggle_keep_prev_mode"],
            None,
            self.tr('Toggle "keep pevious annotation" mode'),
            checkable=True,
        )
        toggle_keep_prev_mode.setChecked(self._config["keep_prev"])

        createMode = action(
            self.tr("Create Polygons"),
            lambda: self.toggleDrawMode(False, createMode="polygon"),
            shortcuts["create_polygon"],
            "objects",
            self.tr("Start drawing polygons"),
            enabled=False,
        )
        createRectangleMode = action(
            self.tr("Create Rectangle"),
            lambda: self.toggleDrawMode(False, createMode="rectangle"),
            shortcuts["create_rectangle"],
            "objects",
            self.tr("Start drawing rectangles"),
            enabled=False,
        )
        createCircleMode = action(
            self.tr("Create Circle"),
            lambda: self.toggleDrawMode(False, createMode="circle"),
            shortcuts["create_circle"],
            "objects",
            self.tr("Start drawing circles"),
            enabled=False,
        )
        createLineMode = action(
            self.tr("Create Line"),
            lambda: self.toggleDrawMode(False, createMode="line"),
            shortcuts["create_line"],
            "objects",
            self.tr("Start drawing lines"),
            enabled=False,
        )
        createPointMode = action(
            self.tr("Create Point"),
            lambda: self.toggleDrawMode(False, createMode="point"),
            shortcuts["create_point"],
            "objects",
            self.tr("Start drawing points"),
            enabled=False,
        )
        createLineStripMode = action(
            self.tr("Create LineStrip"),
            lambda: self.toggleDrawMode(False, createMode="linestrip"),
            shortcuts["create_linestrip"],
            "objects",
            self.tr("Start drawing linestrip. Ctrl+LeftClick ends creation."),
            enabled=False,
        )
        createAiPolygonMode = action(
            self.tr("Create AI-Polygon"),
            lambda: self.toggleDrawMode(False, createMode="ai_polygon"),
            None,
            "objects",
            self.tr("Start drawing ai_polygon. Ctrl+LeftClick ends creation."),
            enabled=False,
        )
        createAiPolygonMode.changed.connect(
            lambda: (
                self.canvas.initializeAiModel(
                    name=self._selectAiModelComboBox.currentText()
                )
                if self.canvas.createMode == "ai_polygon"
                else None
            )
        )
        createAiMaskMode = action(
            self.tr("Create AI-Mask"),
            lambda: self.toggleDrawMode(False, createMode="ai_mask"),
            None,
            "objects",
            self.tr("Start drawing ai_mask. Ctrl+LeftClick ends creation."),
            enabled=False,
        )
        createAiMaskMode.changed.connect(
            lambda: (
                self.canvas.initializeAiModel(
                    name=self._selectAiModelComboBox.currentText()
                )
                if self.canvas.createMode == "ai_mask"
                else None
            )
        )
        editMode = action(
            self.tr("Edit Polygons"),
            self.setEditMode,
            shortcuts["edit_polygon"],
            "edit",
            self.tr("Move and edit the selected polygons"),
            enabled=False,
        )

        delete = action(
            self.tr("Delete Polygons"),
            self.deleteSelectedShape,
            shortcuts["delete_polygon"],
            "cancel",
            self.tr("Delete the selected polygons"),
            enabled=False,
        )
        duplicate = action(
            self.tr("Duplicate Polygons"),
            self.duplicateSelectedShape,
            shortcuts["duplicate_polygon"],
            "copy",
            self.tr("Create a duplicate of the selected polygons"),
            enabled=False,
        )
        copy = action(
            self.tr("Copy Polygons"),
            self.copySelectedShape,
            shortcuts["copy_polygon"],
            "copy_clipboard",
            self.tr("Copy selected polygons to clipboard"),
            enabled=False,
        )
        paste = action(
            self.tr("Paste Polygons"),
            self.pasteSelectedShape,
            shortcuts["paste_polygon"],
            "paste",
            self.tr("Paste copied polygons"),
            enabled=False,
        )
        undoLastPoint = action(
            self.tr("Undo last point"),
            self.canvas.undoLastPoint,
            shortcuts["undo_last_point"],
            "undo",
            self.tr("Undo last drawn point"),
            enabled=False,
        )
        removePoint = action(
            text="Remove Selected Point",
            slot=self.removeSelectedPoint,
            shortcut=shortcuts["remove_selected_point"],
            icon="edit",
            tip="Remove selected point from polygon",
            enabled=False,
        )

        undo = action(
            self.tr("Undo\n"),
            self.undoShapeEdit,
            shortcuts["undo"],
            "undo",
            self.tr("Undo last add and edit of shape"),
            enabled=False,
        )

        hideAll = action(
            self.tr("&Hide\nPolygons"),
            functools.partial(self.togglePolygons, False),
            shortcuts["hide_all_polygons"],
            icon="eye",
            tip=self.tr("Hide all polygons"),
            enabled=False,
        )
        showAll = action(
            self.tr("&Show\nPolygons"),
            functools.partial(self.togglePolygons, True),
            shortcuts["show_all_polygons"],
            icon="eye",
            tip=self.tr("Show all polygons"),
            enabled=False,
        )
        toggleAll = action(
            self.tr("&Toggle\nPolygons"),
            functools.partial(self.togglePolygons, None),
            shortcuts["toggle_all_polygons"],
            icon="eye",
            tip=self.tr("Toggle all polygons"),
            enabled=False,
        )

        help = action(
            self.tr("&Tutorial"),
            self.tutorial,
            icon="help",
            tip=self.tr("Show tutorial page"),
        )

        zoom = QtWidgets.QWidgetAction(self)
        zoomBoxLayout = QtWidgets.QVBoxLayout()
        zoomLabel = QtWidgets.QLabel("Zoom")
        zoomLabel.setAlignment(Qt.AlignCenter)
        zoomBoxLayout.addWidget(zoomLabel)
        zoomBoxLayout.addWidget(self.zoomWidget)
        zoom.setDefaultWidget(QtWidgets.QWidget())
        zoom.defaultWidget().setLayout(zoomBoxLayout)
        self.zoomWidget.setWhatsThis(
            str(
                self.tr(
                    "Zoom in or out of the image. Also accessible with "
                    "{} and {} from the canvas."
                )
            ).format(
                utils.fmtShortcut(
                    "{},{}".format(shortcuts["zoom_in"], shortcuts["zoom_out"])
                ),
                utils.fmtShortcut(self.tr("Ctrl+Wheel")),
            )
        )
        self.zoomWidget.setEnabled(False)

        zoomIn = action(
            self.tr("Zoom &In"),
            functools.partial(self.addZoom, 1.1),
            shortcuts["zoom_in"],
            "zoom-in",
            self.tr("Increase zoom level"),
            enabled=False,
        )
        zoomOut = action(
            self.tr("&Zoom Out"),
            functools.partial(self.addZoom, 0.9),
            shortcuts["zoom_out"],
            "zoom-out",
            self.tr("Decrease zoom level"),
            enabled=False,
        )
        zoomOrg = action(
            self.tr("&Original size"),
            functools.partial(self.setZoom, 100),
            shortcuts["zoom_to_original"],
            "zoom",
            self.tr("Zoom to original size"),
            enabled=False,
        )
        keepPrevScale = action(
            self.tr("&Keep Previous Scale"),
            self.enableKeepPrevScale,
            tip=self.tr("Keep previous zoom scale"),
            checkable=True,
            checked=self._config["keep_prev_scale"],
            enabled=True,
        )
        fitWindow = action(
            self.tr("&Fit Window"),
            self.setFitWindow,
            shortcuts["fit_window"],
            "fit-window",
            self.tr("Zoom follows window size"),
            checkable=True,
            enabled=False,
        )
        fitWidth = action(
            self.tr("Fit &Width"),
            self.setFitWidth,
            shortcuts["fit_width"],
            "fit-width",
            self.tr("Zoom follows window width"),
            checkable=True,
            enabled=False,
        )
        brightnessContrast = action(
            "&Brightness Contrast",
            self.brightnessContrast,
            None,
            "color",
            "Adjust brightness and contrast",
            enabled=False,
        )
        themeSystem = action(
            self.tr("&System Theme"),
            functools.partial(self.setThemeMode, self.THEME_SYSTEM),
            checkable=True,
        )
        themeLight = action(
            self.tr("&Light Theme"),
            functools.partial(self.setThemeMode, self.THEME_LIGHT),
            checkable=True,
        )
        themeDark = action(
            self.tr("&Dark Theme"),
            functools.partial(self.setThemeMode, self.THEME_DARK),
            checkable=True,
        )
        for theme_action in (themeSystem, themeLight, themeDark):
            self._theme_action_group.addAction(theme_action)
        # Group zoom controls into a list for easier toggling.
        zoomActions = (
            self.zoomWidget,
            zoomIn,
            zoomOut,
            zoomOrg,
            fitWindow,
            fitWidth,
        )
        self.zoomMode = self.FIT_WINDOW
        fitWindow.setChecked(Qt.Checked)
        self.scalers = {
            self.FIT_WINDOW: self.scaleFitWindow,
            self.FIT_WIDTH: self.scaleFitWidth,
            # Set to one to scale to 100% when loading files.
            self.MANUAL_ZOOM: lambda: 1,
        }

        edit = action(
            self.tr("&Edit Label"),
            self.editLabel,
            shortcuts["edit_label"],
            "edit",
            self.tr("Modify the label of the selected polygon"),
            enabled=False,
        )

        edit_ID = action(
            self.tr("&Edit ID"),
            self.editID,
            shortcuts["edit_id"],
            "edit",
            self.tr("Modify the ID of the selected polygon"),
            enabled=False,
        )

        call_sort = action(
            self.tr("&ID Association"),
            self.SORT,
            None,
            "edit",
            self.tr("ID Association"),
            enabled=False,
        )

        call_interpolation = action(
            self.tr("&Box/ID Interpolation"),
            self.INTERPOLATION,
            None,
            "edit",
            self.tr("Box/ID Interpolation"),
            enabled=False,
        )

        call_deletion = action(
            self.tr("&Box/ID Modification"),
            self.DELETION,
            None,
            "edit",
            self.tr("Box/ID Modification"),
            enabled=False,
        )

        call_track_forward = action(
            self.tr("&Track Forward (CSRT)"),
            self.trackForward,
            "Ctrl+T",
            "edit",
            self.tr("Track selected bbox forward using OpenCV CSRT"),
            enabled=False,
        )

        call_track_forward_botsort = action(
            self.tr("&Track Forward (BoTSORT)"),
            self.trackForwardBoTSORT,
            "Ctrl+Shift+T",
            "edit",
            self.tr("Track selected bbox forward using YOLO + BoTSORT"),
            enabled=False,
        )

        refine_bbox_ai = action(
            self.tr("&Refine Bbox (AI)"),
            self.refineBboxAI,
            "R",
            "edit",
            self.tr("Refine selected bboxes using EfficientSAM"),
            enabled=False,
        )

        hosted_sam2_point_prompt = action(
            self.tr("&Point Prompt (SAM2)"),
            self.startHostedSam2PointPrompt,
            "Shift+P",
            "edit",
            self.tr("Create a bbox from a hosted SAM2 point prompt"),
            enabled=False,
        )

        hideSelected = action(
            self.tr("&Hide Selected"),
            self.hideSelectedShape,
            "H",
            "eye",
            self.tr("Hide the selected shape"),
            enabled=False,
        )

        fill_drawing = action(
            self.tr("Fill Drawing Polygon"),
            self.canvas.setFillDrawing,
            None,
            "color",
            self.tr("Fill polygon while drawing"),
            checkable=True,
            enabled=True,
        )
        if self._config["canvas"]["fill_drawing"]:
            fill_drawing.trigger()

        # Lavel list context menu.
        labelMenu = QtWidgets.QMenu()
        utils.addActions(labelMenu, (edit, delete))
        self.labelList.setContextMenuPolicy(Qt.CustomContextMenu)
        self.labelList.customContextMenuRequested.connect(self.popLabelListMenu)

        IDMenu = QtWidgets.QMenu()
        utils.addActions(IDMenu, (edit_ID, delete))
        self.IDList.setContextMenuPolicy(Qt.CustomContextMenu)
        self.IDList.customContextMenuRequested.connect(self.popIDListMenu)

        # Store actions for further handling.
        self.actions = utils.struct(
            saveAuto=saveAuto,
            saveWithImageData=saveWithImageData,
            changeOutputDir=changeOutputDir,
            save=save,
            saveAs=saveAs,
            open=open_,
            close=close,
            deleteFile=deleteFile,
            toggleKeepPrevMode=toggle_keep_prev_mode,
            delete=delete,
            edit=edit,
            edit_id=edit_ID,
            SORT=call_sort,
            INPO=call_interpolation,
            DELE=call_deletion,
            trackForward=call_track_forward,
            trackForwardBoTSORT=call_track_forward_botsort,
            refineBboxAI=refine_bbox_ai,
            hostedSam2PointPrompt=hosted_sam2_point_prompt,
            duplicate=duplicate,
            copy=copy,
            paste=paste,
            undoLastPoint=undoLastPoint,
            undo=undo,
            removePoint=removePoint,
            createMode=createMode,
            editMode=editMode,
            createRectangleMode=createRectangleMode,
            createCircleMode=createCircleMode,
            createLineMode=createLineMode,
            createPointMode=createPointMode,
            createLineStripMode=createLineStripMode,
            createAiPolygonMode=createAiPolygonMode,
            createAiMaskMode=createAiMaskMode,
            zoom=zoom,
            zoomIn=zoomIn,
            zoomOut=zoomOut,
            zoomOrg=zoomOrg,
            keepPrevScale=keepPrevScale,
            fitWindow=fitWindow,
            fitWidth=fitWidth,
            brightnessContrast=brightnessContrast,
            themeSystem=themeSystem,
            themeLight=themeLight,
            themeDark=themeDark,
            zoomActions=zoomActions,
            openNextImg=openNextImg,
            openPrevImg=openPrevImg,
            fileMenuActions=(open_, opendir, save, saveAs, close, quit),
            tool=(),
            # XXX: need to add some actions here to activate the shortcut
            editMenu=(
                edit,
                edit_ID,
                duplicate,
                copy,
                paste,
                delete,
                None,
                undo,
                undoLastPoint,
                None,
                removePoint,
                None,
                toggle_keep_prev_mode,
            ),
            # menu shown at right click
            menu=(
                createMode,
                createRectangleMode,
                createCircleMode,
                createLineMode,
                createPointMode,
                createLineStripMode,
                createAiPolygonMode,
                createAiMaskMode,
                editMode,
                edit,
                edit_ID,
                duplicate,
                copy,
                paste,
                delete,
                undo,
                undoLastPoint,
                removePoint,
            ),
            onLoadActive=(
                close,
                createMode,
                createRectangleMode,
                createCircleMode,
                createLineMode,
                createPointMode,
                createLineStripMode,
                createAiPolygonMode,
                createAiMaskMode,
                editMode,
                brightnessContrast,
                hosted_sam2_point_prompt,
            ),
            onShapesPresent=(saveAs, hideAll, showAll, toggleAll, hideSelected),
        )

        self.canvas.vertexSelected.connect(self.actions.removePoint.setEnabled)

        self.menus = utils.struct(
            file=self.menu(self.tr("&File")),
            edit=self.menu(self.tr("&Edit")),
            track=self.menu(self.tr("&Track")),
            view=self.menu(self.tr("&View")),
            help=self.menu(self.tr("&Help")),
            recentFiles=QtWidgets.QMenu(self.tr("Open &Recent")),
            labelList=labelMenu,
            IDList=IDMenu,
        )

        utils.addActions(
            self.menus.file,
            (
                open_,
                openNextImg,
                openPrevImg,
                opendir,
                self.menus.recentFiles,
                save,
                saveAs,
                saveAuto,
                changeOutputDir,
                saveWithImageData,
                close,
                deleteFile,
                None,
                quit,
            ),
        )
        utils.addActions(self.menus.help, (help,))
        utils.addActions(
            self.menus.view,
            (
                self.flag_dock.toggleViewAction(),
                self.label_dock.toggleViewAction(),
                self.shape_dock.toggleViewAction(),
                self.id_dock.toggleViewAction(),
                self.file_dock.toggleViewAction(),
                None,
                themeSystem,
                themeLight,
                themeDark,
                None,
                fill_drawing,
                None,
                hideAll,
                showAll,
                toggleAll,
                hideSelected,
                None,
                zoomIn,
                zoomOut,
                zoomOrg,
                keepPrevScale,
                None,
                fitWindow,
                fitWidth,
                None,
                brightnessContrast,
            ),
        )

        self._syncThemeActions()
        self.applyTheme(self._theme_mode)
        utils.addActions(
            self.menus.track,
            (
                call_interpolation,
                call_sort,
                call_deletion,
                call_track_forward,
                call_track_forward_botsort,
                refine_bbox_ai,
                hosted_sam2_point_prompt,
            ),
        )

        self.menus.file.aboutToShow.connect(self.updateFileMenu)

        # Custom context menu for the canvas widget:
        utils.addActions(self.canvas.menus[0], self.actions.menu)
        # utils.addActions(self.canvas.editID)
        utils.addActions(
            self.canvas.menus[1],
            (
                action("&Copy here", self.copyShape),
                action("&Move here", self.moveShape),
            ),
        )

        selectAiModel = QtWidgets.QWidgetAction(self)
        selectAiModel.setDefaultWidget(QtWidgets.QWidget())
        selectAiModel.defaultWidget().setLayout(QtWidgets.QVBoxLayout())
        #
        selectAiModelLabel = QtWidgets.QLabel(self.tr("AI Model"))
        selectAiModelLabel.setAlignment(QtCore.Qt.AlignCenter)
        selectAiModel.defaultWidget().layout().addWidget(selectAiModelLabel)
        #
        self._selectAiModelComboBox = QtWidgets.QComboBox()
        selectAiModel.defaultWidget().layout().addWidget(self._selectAiModelComboBox)
        model_names = [model.name for model in MODELS]
        self._selectAiModelComboBox.addItems(model_names)
        if self._config["ai"]["default"] in model_names:
            model_index = model_names.index(self._config["ai"]["default"])
        else:
            logger.warning(
                "Default AI model is not found: %r",
                self._config["ai"]["default"],
            )
            model_index = 0
        self._selectAiModelComboBox.setCurrentIndex(model_index)
        self._selectAiModelComboBox.currentIndexChanged.connect(
            lambda: (
                self.canvas.initializeAiModel(
                    name=self._selectAiModelComboBox.currentText()
                )
                if self.canvas.createMode in ["ai_polygon", "ai_mask"]
                else None
            )
        )

        self.tools = self.toolbar("Tools")
        self.actions.tool = (
            open_,
            opendir,
            openPrevImg,
            openNextImg,
            save,
            deleteFile,
            None,
            createMode,
            editMode,
            duplicate,
            delete,
            undo,
            brightnessContrast,
            None,
            fitWindow,
            zoom,
            None,
            selectAiModel,
        )

        self.statusBar().showMessage(str(self.tr("%s started.")) % __appname__)
        self.statusBar().show()

        if output_file is not None and self._config["auto_save"]:
            logger.warning(
                "If `auto_save` argument is True, `output_file` argument "
                "is ignored and output filename is automatically "
                "set as IMAGE_BASENAME.json."
            )
        self.output_file = output_file
        self.output_dir = output_dir

        # Application state.
        self.image = QtGui.QImage()
        self.imagePath = None
        self.recentFiles = []
        self.maxRecent = 7
        self.otherData = None
        self._explicit_label_path = None
        self._dirty_revision = 0
        self._pending_auto_save_filename = None
        self._pending_auto_save_target = None
        self.zoom_level = 100
        self.fit_window = False
        self.zoom_values = {}  # key=filename, value=(zoom_mode, zoom_value)
        self.brightnessContrast_values = {}
        self.scroll_values = {
            Qt.Horizontal: {},
            Qt.Vertical: {},
        }  # key=filename, value=scroll_value

        initial_directory = (
            filename if filename is not None and osp.isdir(filename) else None
        )
        if initial_directory is not None:
            self.importDirImages(filename, load=False)
        else:
            self.filename = filename

        if config["file_search"]:
            self.fileSearch.setText(config["file_search"])
            self.fileSearchChanged()

        # XXX: Could be completely declarative.
        # Restore application settings.
        self.settings = QtCore.QSettings("labelme", "labelme")
        self.recentFiles = self.settings.value("recentFiles", []) or []
        size = self.settings.value("window/size", QtCore.QSize(600, 500))
        position = self.settings.value("window/position", QtCore.QPoint(0, 0))
        state = self.settings.value("window/state", QtCore.QByteArray())
        self.resize(size)
        self.move(position)
        # or simply:
        # self.restoreGeometry(settings['window/geometry']
        self.restoreState(state)

        # Populate the File menu dynamically.
        self.updateFileMenu()
        # Since loading the file may take some time,
        # make sure it runs in the background.
        initial_file = self.filename
        if initial_directory is not None and self.imageList:
            initial_file = self.imageList[0]
        if initial_file is not None:
            self.queueEvent(functools.partial(self.loadFile, initial_file))

        # Callbacks:
        self.zoomWidget.valueChanged.connect(self.paintCanvas)

        self.populateModeActions()

        self._save_timer = QtCore.QTimer(self)
        self._save_timer.setSingleShot(True)
        self._save_timer.setInterval(500)
        self._save_timer.timeout.connect(self._debouncedSave)

        self._imageListCache = None

        # self.firstStart = True
        # if self.firstStart:
        #    QWhatsThis.enterWhatsThisMode()

    def menu(self, title, actions=None):
        menu = self.menuBar().addMenu(title)
        if actions:
            utils.addActions(menu, actions)
        return menu

    def toolbar(self, title, actions=None):
        toolbar = ToolBar(title)
        toolbar.setObjectName("%sToolBar" % title)
        # toolbar.setOrientation(Qt.Vertical)
        toolbar.setToolButtonStyle(Qt.ToolButtonTextUnderIcon)
        if actions:
            utils.addActions(toolbar, actions)
        self.addToolBar(Qt.TopToolBarArea, toolbar)
        return toolbar

    # Support Functions

    def noShapes(self):
        return not len(self.labelList)

    def populateModeActions(self):
        tool, menu = self.actions.tool, self.actions.menu
        self.tools.clear()
        utils.addActions(self.tools, tool)
        self.canvas.menus[0].clear()
        utils.addActions(self.canvas.menus[0], menu)
        self.menus.edit.clear()
        actions = (
            self.actions.createMode,
            self.actions.createRectangleMode,
            self.actions.createCircleMode,
            self.actions.createLineMode,
            self.actions.createPointMode,
            self.actions.createLineStripMode,
            self.actions.createAiPolygonMode,
            self.actions.createAiMaskMode,
            self.actions.editMode,
        )
        utils.addActions(self.menus.edit, actions + self.actions.editMenu)

    def _syncThemeActions(self):
        theme = self._theme_mode
        self.actions.themeSystem.setChecked(theme == self.THEME_SYSTEM)
        self.actions.themeLight.setChecked(theme == self.THEME_LIGHT)
        self.actions.themeDark.setChecked(theme == self.THEME_DARK)

    def setThemeMode(self, theme_mode):
        self._theme_mode = theme_mode
        self._config["theme"] = theme_mode
        self._syncThemeActions()
        self.applyTheme(theme_mode)

    def applyTheme(self, theme_mode):
        app = QtWidgets.QApplication.instance()
        if app is None:
            return
        if theme_mode == self.THEME_SYSTEM:
            app.setPalette(app.style().standardPalette())
            app.setStyleSheet("")
            return

        palette = QtGui.QPalette()
        if theme_mode == self.THEME_DARK:
            palette.setColor(QtGui.QPalette.Window, QtGui.QColor(45, 45, 45))
            palette.setColor(QtGui.QPalette.WindowText, QtCore.Qt.white)
            palette.setColor(QtGui.QPalette.Base, QtGui.QColor(30, 30, 30))
            palette.setColor(QtGui.QPalette.AlternateBase, QtGui.QColor(45, 45, 45))
            palette.setColor(QtGui.QPalette.ToolTipBase, QtCore.Qt.white)
            palette.setColor(QtGui.QPalette.ToolTipText, QtCore.Qt.white)
            palette.setColor(QtGui.QPalette.Text, QtCore.Qt.white)
            palette.setColor(QtGui.QPalette.Button, QtGui.QColor(45, 45, 45))
            palette.setColor(QtGui.QPalette.ButtonText, QtCore.Qt.white)
            palette.setColor(QtGui.QPalette.BrightText, QtCore.Qt.red)
            palette.setColor(QtGui.QPalette.Highlight, QtGui.QColor(42, 130, 218))
            palette.setColor(QtGui.QPalette.HighlightedText, QtCore.Qt.black)
        else:
            palette = app.style().standardPalette()
        app.setPalette(palette)
        app.setStyleSheet("")

    def _resolveJsonPath(self, image_path=None, for_write=True, image_paths=None):
        current_filename = getattr(self, "filename", None)
        image_path = image_path or current_filename
        explicit_label_path = None
        if image_path == current_filename:
            explicit_label_path = self._explicit_label_path
        return resolve_annotation_path(
            image_path=image_path,
            output_dir=self.output_dir,
            image_root=self.lastOpenDir,
            image_paths=self.imageList if image_paths is None else image_paths,
            for_write=for_write,
            explicit_label_path=explicit_label_path,
        )

    def _canonicalJsonPath(self, image_path=None):
        image_path = image_path or getattr(self, "filename", None)
        return canonical_annotation_path(
            image_path=image_path,
            output_dir=self.output_dir,
            image_root=self.lastOpenDir,
        )

    def _legacyAnnotationSources(self, image_path, label_file, destination):
        """Return only the legacy source actually loaded for this save."""
        source = getattr(label_file, "filename", None) if label_file else None
        canonical = self._canonicalJsonPath(image_path)
        if not source or not canonical or not destination:
            return []
        source = osp.abspath(source)
        canonical = osp.abspath(canonical)
        destination = osp.abspath(destination)
        if destination != canonical:
            return []
        candidates = legacy_annotation_paths(
            image_path,
            output_dir=self.output_dir,
            image_root=self.lastOpenDir,
            image_paths=self.imageList,
        )
        return [source] if source in candidates else []

    def _loadLabelForImage(self, image_path):
        path = self._resolveJsonPath(image_path=image_path, for_write=False)
        if path and osp.isfile(path):
            return LabelFile(path)
        return None

    @staticmethod
    def _relativeImagePath(image_path, annotation_path):
        image_path = osp.abspath(image_path)
        try:
            relative_path = osp.relpath(image_path, osp.dirname(annotation_path))
        except ValueError:
            return image_path
        return relative_path

    def _labelSaveRequest(self, image_path, shapes, label_file=None):
        destination = self._resolveJsonPath(image_path=image_path, for_write=True)
        image_height = label_file.imageHeight if label_file else None
        image_width = label_file.imageWidth if label_file else None
        if not image_height or not image_width:
            image = load_oriented_cv_image(image_path)
            if image is None:
                raise LabelFileError("Cannot read target image: {}".format(image_path))
            image_height, image_width = image.shape[:2]
        image_data = None
        if label_file and label_file.imageDataEmbedded:
            image_data = label_file.imageData
        elif self._config["store_data"]:
            image_data = LabelFile.load_image_file(image_path)
            if image_data is None:
                raise LabelFileError("Cannot embed target image: {}".format(image_path))
        request = dict(
            filename=destination,
            shapes=shapes,
            imagePath=self._relativeImagePath(image_path, destination),
            imageData=image_data,
            imageHeight=image_height,
            imageWidth=image_width,
            otherData=label_file.otherData if label_file else {},
            flags=label_file.flags if label_file else {},
        )
        legacy_sources = self._legacyAnnotationSources(
            image_path, label_file, destination
        )
        if legacy_sources:
            request["_retire_sources"] = legacy_sources
        return request

    def _saveLabelBatch(self, requests, title):
        try:
            save_label_files_atomically(requests)
        except LabelFileError as exc:
            self.errorMessage(title, str(exc))
            return False
        for request in requests:
            image_path = osp.abspath(
                osp.normpath(
                    osp.join(osp.dirname(request["filename"]), request["imagePath"])
                )
            )
            items = self.fileListWidget.findItems(image_path, Qt.MatchExactly)
            for item in items:
                item.setCheckState(Qt.Checked)
        return True

    def setDirty(self):
        self.actions.undo.setEnabled(self.canvas.isShapeRestorable)
        self.dirty = True
        self._dirty_revision += 1
        self.actions.save.setEnabled(True)
        title = __appname__
        if self.filename is not None:
            title = "{} - {}*".format(title, self.filename)
        self.setWindowTitle(title)
        if self.filename and (
            self._config["auto_save"] or self.actions.saveAuto.isChecked()
        ):
            self._pending_auto_save_filename = self.filename
            self._pending_auto_save_target = self._resolveJsonPath(for_write=True)
            self._save_timer.start()

    def _debouncedSave(self):
        filename = self._pending_auto_save_filename
        target = self._pending_auto_save_target
        revision = self._dirty_revision
        if not filename or not target:
            return False
        if filename != self.filename:
            logger.error(
                "Refusing to autosave %r while frame %r is displayed",
                filename,
                self.filename,
            )
            return False
        if self.saveLabels(target):
            if revision == self._dirty_revision:
                self.setClean()
            return True
        return False

    def _flushPendingAutoSave(self):
        save_timer = getattr(self, "_save_timer", None)
        if save_timer is not None and save_timer.isActive():
            save_timer.stop()
        if self.dirty and self._pending_auto_save_target:
            return self._debouncedSave()
        return True

    def _ensureSavedForWorkflow(self, title):
        if not self._flushPendingAutoSave():
            return False
        if not self.dirty:
            return True
        answer = QtWidgets.QMessageBox.question(
            self,
            title,
            self.tr("Save current-frame edits before running this operation?"),
            QtWidgets.QMessageBox.Save | QtWidgets.QMessageBox.Cancel,
            QtWidgets.QMessageBox.Save,
        )
        return answer == QtWidgets.QMessageBox.Save and bool(self.saveFile())

    def setClean(self):
        save_timer = getattr(self, "_save_timer", None)
        if save_timer is not None:
            save_timer.stop()
        self._pending_auto_save_filename = None
        self._pending_auto_save_target = None
        self.dirty = False
        self.actions.save.setEnabled(False)
        self.actions.createMode.setEnabled(True)
        self.actions.createRectangleMode.setEnabled(True)
        self.actions.createCircleMode.setEnabled(True)
        self.actions.createLineMode.setEnabled(True)
        self.actions.createPointMode.setEnabled(True)
        self.actions.createLineStripMode.setEnabled(True)
        self.actions.createAiPolygonMode.setEnabled(True)
        self.actions.createAiMaskMode.setEnabled(True)
        self.actions.SORT.setEnabled(True)
        self.actions.INPO.setEnabled(True)
        self.actions.DELE.setEnabled(True)
        can_track_forward = False
        if self.filename in self.imageList:
            can_track_forward = self.imageList.index(self.filename) + 1 < len(
                self.imageList
            )
        self.actions.trackForward.setEnabled(can_track_forward)
        self.actions.trackForwardBoTSORT.setEnabled(can_track_forward)
        self.actions.refineBboxAI.setEnabled(True)
        self.actions.hostedSam2PointPrompt.setEnabled(True)
        title = __appname__
        if self.filename is not None:
            title = "{} - {}".format(title, self.filename)
        self.setWindowTitle(title)

        if self.hasLabelFile():
            self.actions.deleteFile.setEnabled(True)
        else:
            self.actions.deleteFile.setEnabled(False)

    def toggleActions(self, value=True):
        """Enable/Disable widgets which depend on an opened image."""
        for z in self.actions.zoomActions:
            z.setEnabled(value)
        for action in self.actions.onLoadActive:
            action.setEnabled(value)

    def queueEvent(self, function):
        QtCore.QTimer.singleShot(0, function)

    def status(self, message, delay=5000):
        self.statusBar().showMessage(message, delay)

    def resetState(self, release_ai_model=True):
        if not self._hosted_sam2_cache_frames:
            self._hosted_sam2_image_cache.clear()
        self.labelList.clear()
        self.IDList.clear()
        self.filename = None
        self._explicit_label_path = None
        self.imagePath = None
        self.imageData = None
        self.image = QtGui.QImage()
        self.labelFile = None
        self.otherData = None
        self.canvas.resetState(release_ai_model=release_ai_model)

    def currentItem(self):
        items_l = self.labelList.selectedItems()
        items_i = self.IDList.selectedItems()
        if items_l and items_i:
            return items_l[0], items_i[0]
        if items_l:
            shape = items_l[0].shape()
            try:
                return items_l[0], self.IDList.findItemByShape(shape)
            except ValueError:
                return items_l[0], None
        if items_i:
            shape = items_i[0].shape()
            try:
                return self.labelList.findItemByShape(shape), items_i[0]
            except ValueError:
                return None, items_i[0]
        return None, None

    def addRecentFile(self, filename):
        if filename in self.recentFiles:
            self.recentFiles.remove(filename)
        elif len(self.recentFiles) >= self.maxRecent:
            self.recentFiles.pop()
        self.recentFiles.insert(0, filename)

    def _brightnessContrastKey(self, filename=None):
        if filename is None:
            filename = self.filename

        if filename:
            filename = osp.abspath(str(filename))
            if self.lastOpenDir and filename in self.imageList:
                return osp.normpath(osp.abspath(self.lastOpenDir))
            return osp.normpath(osp.dirname(filename))

        if self.lastOpenDir:
            return osp.normpath(osp.abspath(self.lastOpenDir))
        return None

    def _previousBrightnessContrastValues(self, current_key):
        for filename in self.recentFiles:
            previous_key = self._brightnessContrastKey(filename)
            if previous_key and previous_key != current_key:
                return self.brightnessContrast_values.get(previous_key, (None, None))
        return (None, None)

    # Callbacks

    def undoShapeEdit(self):
        if not self.canvas.restoreShape():
            return False
        self.labelList.clear()
        self.IDList.clear()
        self.loadShapes(self.canvas.shapes)
        self.setDirty()
        return True

    def tutorial(self):
        url = "https://github.com/wkentaro/labelme/tree/main/examples/tutorial"  # NOQA
        webbrowser.open(url)

    def toggleDrawingSensitive(self, drawing=True):
        """Toggle drawing sensitive.

        In the middle of drawing, toggling between modes should be disabled.
        """
        self.actions.editMode.setEnabled(not drawing)
        self.actions.undoLastPoint.setEnabled(drawing)
        self.actions.undo.setEnabled(not drawing)
        self.actions.delete.setEnabled(not drawing)

    def toggleDrawMode(self, edit=True, createMode="polygon"):
        draw_actions = {
            "polygon": self.actions.createMode,
            "rectangle": self.actions.createRectangleMode,
            "circle": self.actions.createCircleMode,
            "point": self.actions.createPointMode,
            "line": self.actions.createLineMode,
            "linestrip": self.actions.createLineStripMode,
            "ai_polygon": self.actions.createAiPolygonMode,
            "ai_mask": self.actions.createAiMaskMode,
        }
        ai_modes = {"ai_polygon", "ai_mask"}
        previous_create_mode = self.canvas.createMode

        self.canvas.setEditing(edit)
        self.canvas.createMode = createMode
        if previous_create_mode in ai_modes and createMode not in ai_modes:
            self.canvas.releaseAiModel()
        if edit:
            for draw_action in draw_actions.values():
                draw_action.setEnabled(True)
        else:
            for draw_mode, draw_action in draw_actions.items():
                draw_action.setEnabled(createMode != draw_mode)
        self.actions.editMode.setEnabled(not edit)

    def setEditMode(self):
        self.toggleDrawMode(True)

    def updateFileMenu(self):
        current = self.filename

        def exists(filename):
            return osp.exists(str(filename))

        menu = self.menus.recentFiles
        menu.clear()
        files = [f for f in self.recentFiles if f != current and exists(f)]
        for i, f in enumerate(files):
            icon = utils.newIcon("labels")
            action = QtWidgets.QAction(
                icon, "&%d %s" % (i + 1, QtCore.QFileInfo(f).fileName()), self
            )
            action.triggered.connect(functools.partial(self.loadRecent, f))
            menu.addAction(action)

    def popLabelListMenu(self, point):
        self.menus.labelList.exec_(self.labelList.mapToGlobal(point))

    def popIDListMenu(self, point):
        self.menus.IDList.exec_(self.IDList.mapToGlobal(point))

    def validateLabel(self, label):
        # no validation
        if self._config["validate_label"] is None:
            return True

        for i in range(self.uniqLabelList.count()):
            label_i = self.uniqLabelList.item(i).data(Qt.UserRole)
            if self._config["validate_label"] in ["exact"]:
                if label_i == label:
                    return True
        return False

    def editIR_info(self, item=None):
        dialog = InterpolationRefineInfo_Dialog(
            parent=self,
        )
        if dialog.exec_() != QtWidgets.QDialog.Accepted:
            return

        self.ir_name = dialog.name
        self.ir_id = dialog.id

        self.interpolationrefine_list.statusBar.showMessage(
            f"Name: {self.ir_name} | ID: {self.ir_id}"
        )

    def SORT(self, item=None):
        if not self.imageList or self.filename not in self.imageList:
            self.errorMessage("Track IDs", "Open a frame sequence before tracking.")
            return
        if not self._ensureSavedForWorkflow("Track IDs"):
            return
        current_index = self.imageList.index(self.filename)
        dialog = TrackDialog(
            current_frame=current_index + 1,
            total_frames=len(self.imageList),
            parent=self,
        )
        if dialog.exec_() != QtWidgets.QDialog.Accepted:
            return
        if dialog.option_value not in {1, 2}:
            return
        start_index = 0 if dialog.option_value == 1 else current_index
        end_index = dialog.end_frame.value()
        if end_index <= start_index:
            self.errorMessage("Track IDs", "End frame must include the start frame.")
            return

        frame_data = []
        try:
            for image_path in self.imageList[start_index:end_index]:
                label_file = self._loadLabelForImage(image_path)
                shapes = list(label_file.shapes) if label_file else []
                rectangles = []
                for shape_index, shape in enumerate(shapes):
                    if shape.get("shape_type") != "rectangle":
                        continue
                    points = normalized_rectangle_points(shape.get("points", []))
                    (x1, y1), (x2, y2) = points
                    rectangles.append((shape_index, [x1, y1, x2, y2]))
                frame_data.append((image_path, label_file, shapes, rectangles))
        except (LabelFileError, ValueError) as exc:
            self.errorMessage("Track IDs", str(exc))
            return
        if not frame_data[0][3]:
            self.errorMessage(
                "Track IDs", "The first target frame has no valid rectangles."
            )
            return

        tracker = SORT_main(max_age=1, min_hits=1, iou_threshold=0.1)
        KalmanBoxTracker.count = 0
        seeded_id_values = {}
        if dialog.option_value == 2:
            seed_identity_keys = set()
            for shape_index, box in frame_data[0][3]:
                value = shape_track_id(frame_data[0][2][shape_index])
                try:
                    if isinstance(value, bool):
                        raise ValueError
                    numeric_value = float(value)
                    if not math.isfinite(numeric_value):
                        raise ValueError
                except (TypeError, ValueError, OverflowError):
                    self.errorMessage(
                        "Track IDs",
                        "Every seed rectangle must have a numeric track ID.",
                    )
                    return
                identity_key = (type(value).__name__, repr(value))
                if identity_key in seed_identity_keys:
                    self.errorMessage(
                        "Track IDs",
                        "Every seed rectangle must have a unique track ID.",
                    )
                    return
                seed_identity_keys.add(identity_key)
                tracker_id = len(seeded_id_values)
                seeded_id_values[tracker_id] = value
                tracker.trackers.append(
                    KalmanBoxTracker(np.asarray(box, dtype=float), id=tracker_id)
                )
            KalmanBoxTracker.count = len(seeded_id_values)

        requests = []
        try:
            for frame_number, (image_path, label_file, shapes, rectangles) in enumerate(
                frame_data
            ):
                detections = np.asarray(
                    [box + [1.0] for _, box in rectangles], dtype=float
                )
                if detections.size == 0:
                    detections = np.empty((0, 5), dtype=float)
                tracks = tracker.update(detections)
                if not rectangles or tracks.size == 0:
                    continue

                rectangle_centers = np.asarray(
                    [
                        [(box[0] + box[2]) / 2, (box[1] + box[3]) / 2]
                        for _, box in rectangles
                    ]
                )
                track_centers = np.column_stack(
                    (
                        (tracks[:, 0] + tracks[:, 2]) / 2,
                        (tracks[:, 1] + tracks[:, 3]) / 2,
                    )
                )
                costs = np.linalg.norm(
                    rectangle_centers[:, None, :] - track_centers[None, :, :], axis=2
                )
                shape_rows, track_columns = linear_sum_assignment(costs)
                changed = False
                for shape_row, track_column in zip(shape_rows, track_columns):
                    shape_index = rectangles[shape_row][0]
                    internal_id = int(tracks[track_column, 4])
                    is_seeded_id = internal_id in seeded_id_values
                    new_id = (
                        seeded_id_values[internal_id]
                        if is_seeded_id
                        else str(internal_id)
                    )
                    current_id = shape_track_id(shapes[shape_index])
                    if type(current_id) is not type(new_id) or current_id != new_id:
                        shapes[shape_index]["track_id"] = new_id
                        shapes[shape_index]["group_id"] = (
                            new_id if is_seeded_id else internal_id
                        )
                        changed = True
                if changed:
                    requests.append(
                        self._labelSaveRequest(image_path, shapes, label_file)
                    )
        except (LabelFileError, ValueError, FloatingPointError) as exc:
            self.errorMessage("Track IDs", str(exc))
            return

        if not requests:
            self.informationMessage(
                "Track IDs", "SORT completed, but no track IDs needed changing."
            )
            return
        if not self._saveLabelBatch(requests, "Track IDs"):
            return
        self.loadFile(self.filename)
        self.informationMessage("Track IDs", "SORT ID association is complete.")

    def INTERPOLATION(self, item=None):
        if not self.imageList or self.filename not in self.imageList:
            self.errorMessage(
                "Box Interpolation", "Open a frame sequence before interpolating."
            )
            return
        if not self._ensureSavedForWorkflow("Box Interpolation"):
            return
        dialog = InterpolationDialog(
            min_val=1, max_val=len(self.imageList), parent=self
        )
        if dialog.exec_() != QtWidgets.QDialog.Accepted:
            return
        try:
            options = dialog.options()
            start_frame = options.start_frame
            end_frame = options.end_frame
            interval = options.interval
            img_indices = interpolation_indices(
                start_frame, end_frame, interval, len(self.imageList)
            )
        except ValueError as exc:
            self.errorMessage("Box Interpolation", str(exc))
            return
        if start_frame == end_frame:
            self.errorMessage(
                "Box Interpolation", "Interpolation requires at least two frames."
            )
            return

        interpolation_list = [self.imageList[index] for index in img_indices]
        interpolation_filename = interpolation_list[0]
        if not self.loadFile(interpolation_filename):
            return

        self.start_INP0 = start_frame
        self.end_INP0 = end_frame
        self.interval_INPO = interval
        self.ID_INPO = options.track_id
        self.label_INPO = options.label
        self.mode = "TRACK INTERPOLATION"
        self.INTERPOLATION_indices = img_indices
        self.INTERPOLATION_list = interpolation_list
        self.INTERPOLATION_filename = interpolation_filename

    def OKAY(self, item=None):
        from sklearn.gaussian_process import GaussianProcessRegressor
        from sklearn.gaussian_process.kernels import RationalQuadratic

        if self.mode != "TRACK INTERPOLATION":
            return
        if not self._ensureSavedForWorkflow("Box Interpolation"):
            return

        reference_boxes = []
        try:
            for image_path in self.INTERPOLATION_list:
                label_file = self._loadLabelForImage(image_path)
                if label_file is None:
                    raise ValueError(
                        "Reference frame has no annotation: {}".format(image_path)
                    )
                matches = [
                    shape
                    for shape in label_file.shapes
                    if shape.get("label") == self.label_INPO
                    and str(shape_track_id(shape)) == self.ID_INPO
                ]
                if len(matches) != 1:
                    raise ValueError(
                        "Reference frame {} must contain exactly one {}-{} "
                        "rectangle; found {}.".format(
                            osp.basename(image_path),
                            self.label_INPO,
                            self.ID_INPO,
                            len(matches),
                        )
                    )
                if matches[0].get("shape_type") != "rectangle":
                    raise ValueError(
                        "Reference shape in {} is not a rectangle.".format(
                            osp.basename(image_path)
                        )
                    )
                points = normalized_rectangle_points(matches[0]["points"])
                (x1, y1), (x2, y2) = points
                reference_boxes.append([(x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1])
        except (LabelFileError, ValueError) as exc:
            self.errorMessage("Box Interpolation", str(exc))
            return

        reference_indices = np.asarray(self.INTERPOLATION_indices).reshape(-1, 1)
        reference_boxes = np.asarray(reference_boxes, dtype=float)
        target_indices = np.arange(self.start_INP0 - 1, self.end_INP0)
        predictions = []
        for coordinate in range(4):
            model = GaussianProcessRegressor(
                kernel=RationalQuadratic(), random_state=0
            ).fit(reference_indices, reference_boxes[:, coordinate])
            predictions.append(model.predict(target_indices.reshape(-1, 1)))
        predictions = np.stack(predictions, axis=1)
        if not np.isfinite(predictions).all():
            self.errorMessage(
                "Box Interpolation", "Interpolation produced non-finite coordinates."
            )
            return

        requests = []
        reference_set = set(self.INTERPOLATION_indices)
        try:
            for offset, image_index in enumerate(target_indices):
                if image_index in reference_set:
                    continue
                image_path = self.imageList[image_index]
                label_file = self._loadLabelForImage(image_path)
                shapes = list(label_file.shapes) if label_file else []
                image = load_oriented_cv_image(image_path)
                if image is None:
                    raise ValueError("Cannot read target image: {}".format(image_path))

                points = prediction_to_clamped_rectangle(
                    predictions[offset], image.shape[1], image.shape[0]
                )

                shapes = upsert_tracked_rectangle(
                    shapes,
                    self.label_INPO,
                    self.ID_INPO,
                    points,
                )
                requests.append(self._labelSaveRequest(image_path, shapes, label_file))
        except (LabelFileError, ValueError) as exc:
            self.errorMessage("Box Interpolation", str(exc))
            return

        if not self._saveLabelBatch(requests, "Box Interpolation"):
            return
        self.mode = "NORMAL"
        start_filename = self.imageList[self.start_INP0 - 1]
        get_index = self.imageList.index(start_filename) + 1
        self.navigation_list.statusBar.showMessage(
            f"Status: {get_index}/{len(self.imageList)} | Mode: {self.mode}"
        )
        self.loadFile(start_filename)
        self.informationMessage(
            "Box Interpolation",
            (
                f"Track {self.label_INPO}-{self.ID_INPO} from frame "
                f"{self.start_INP0} to {self.end_INP0} interpolation is complete."
            ),
        )

    def DELETION(self, item=None):
        dialog = DeletionDialog(parent=self)
        current_frame = (
            self.imageList.index(self.filename) + 1
            if self.filename in self.imageList
            else 1
        )
        if hasattr(dialog, "setFrameRange"):
            dialog.setFrameRange(1, len(self.imageList), current_frame)
        if dialog.exec_() != QtWidgets.QDialog.Accepted:
            return
        try:
            start_frame = (
                dialog.start_frame_cell.value()
                if hasattr(dialog.start_frame_cell, "value")
                else int(dialog.start_frame_cell.text().strip())
            )
            end_frame = (
                dialog.end_frame_cell.value()
                if hasattr(dialog.end_frame_cell, "value")
                else int(dialog.end_frame_cell.text().strip())
            )
        except ValueError:
            self.errorMessage(
                "Track Modification", "Start and end frames must be whole numbers."
            )
            return
        track_id = dialog.ID_cell.text().strip()
        label = dialog.label_cell.text().strip()
        if not track_id or not label:
            self.errorMessage("Track Modification", "Object label and ID are required.")
            return
        if not 1 <= start_frame <= end_frame <= len(self.imageList):
            self.errorMessage(
                "Track Modification",
                "Frames must satisfy 1 <= start <= end <= frame count.",
            )
            return

        mode = dialog.mode
        new_id = dialog.new_ID_cell.text().strip()
        new_label = dialog.new_label_cell.text().strip()
        if mode == "Swap ID" and not new_id:
            self.errorMessage(
                "Track Modification", "New ID is required when swapping IDs."
            )
            return
        if mode == "Swap Label" and not new_label:
            self.errorMessage(
                "Track Modification", "New label is required when swapping labels."
            )
            return
        if mode == "Remove Box" and (new_id or new_label):
            self.errorMessage(
                "Track Modification",
                "Remove Box deletes matching boxes. Choose Swap ID or Swap Label "
                "to apply the new value.",
            )
            return
        if mode not in {"Remove Box", "Swap ID", "Swap Label"}:
            self.errorMessage("Track Modification", "Unsupported modification mode.")
            return
        if mode == "Swap ID" and str(new_id) == str(track_id):
            self.informationMessage(
                "Track Modification",
                f"Track {label}-{track_id} already has ID {new_id}; "
                "no files were changed.",
            )
            return
        if mode == "Swap Label" and new_label == label:
            self.informationMessage(
                "Track Modification",
                f"Track {label}-{track_id} already has label {new_label}; "
                "no files were changed.",
            )
            return
        if not self._ensureSavedForWorkflow("Track Modification"):
            return

        def identity_matches(shape, wanted_id=track_id):
            return shape.get("label") == label and str(shape_track_id(shape)) == str(
                wanted_id
            )

        def validate_box(shape, role):
            if shape.get("shape_type") != "rectangle":
                raise ValueError(
                    "{} {}-{} collides with a non-rectangle shape.".format(
                        role, label, shape_track_id(shape)
                    )
                )
            try:
                normalized_rectangle_points(shape.get("points"))
            except ValueError as exc:
                raise ValueError(
                    "{} {}-{} has invalid rectangle geometry: {}".format(
                        role, label, shape_track_id(shape), exc
                    )
                ) from exc

        def set_track_id(shape, value):
            shape["track_id"] = value
            shape["group_id"] = int(value) if value.isdigit() else value

        requests = []
        source_count = 0
        try:
            for image_path in self.imageList[start_frame - 1 : end_frame]:
                label_file = self._loadLabelForImage(image_path)
                if label_file is None:
                    continue
                shapes = label_file.shapes
                source_indices = set()
                for index, shape in enumerate(shapes):
                    if identity_matches(shape):
                        validate_box(shape, "Source track")
                        source_indices.add(index)
                frame_has_source = bool(source_indices)
                source_count += len(source_indices)
                destination_indices = set()
                if mode == "Swap ID" and frame_has_source:
                    for index, shape in enumerate(shapes):
                        if identity_matches(shape, new_id):
                            validate_box(shape, "Destination track")
                            destination_indices.add(index)
                if mode == "Swap Label" and frame_has_source:
                    for index, shape in enumerate(shapes):
                        if shape.get("label") == new_label and str(
                            shape_track_id(shape)
                        ) == str(track_id):
                            validate_box(shape, "Destination track")
                            destination_indices.add(index)
                    if destination_indices:
                        raise ValueError(
                            "Destination track {}-{} already exists; no labels "
                            "were changed.".format(new_label, track_id)
                        )
                updated_shapes = []
                changed = False
                for index, shape in enumerate(shapes):
                    shape = dict(shape)
                    if index in source_indices:
                        changed = True
                        if mode == "Remove Box":
                            continue
                        if mode == "Swap ID":
                            set_track_id(shape, new_id)
                        elif mode == "Swap Label":
                            shape["label"] = new_label
                    elif index in destination_indices:
                        changed = True
                        set_track_id(shape, track_id)
                    updated_shapes.append(shape)
                if changed:
                    requests.append(
                        self._labelSaveRequest(image_path, updated_shapes, label_file)
                    )
        except (LabelFileError, ValueError) as exc:
            self.errorMessage("Track Modification", str(exc))
            return

        if source_count == 0:
            self.informationMessage(
                "Track Modification",
                f"Track {label}-{track_id} was not found; no files were changed.",
            )
            return
        if not self._saveLabelBatch(requests, "Track Modification"):
            return

        start_filename = self.imageList[start_frame - 1]
        self.navigation_list.statusBar.showMessage(
            f"Status: {start_frame}/{len(self.imageList)} | Mode: {self.mode}"
        )
        self.loadFile(start_filename)
        if mode == "Remove Box":
            message = (
                f"Track {label}-{track_id} from frame {start_frame} to "
                f"{end_frame} was deleted."
            )
        elif mode == "Swap ID":
            message = (
                f"Track {label}-{track_id} swapped with ID {new_id} from frame "
                f"{start_frame} to {end_frame}."
            )
        else:
            message = (
                f"Track {label}-{track_id} changed to label {new_label} from frame "
                f"{start_frame} to {end_frame}."
            )
        self.informationMessage("Track Modification", message)

    def _trackResultRequest(
        self, img_path, label, track_id, group_id, new_points, img_shape
    ):
        loaded = self._loadLabelForImage(img_path)
        shapes = list(loaded.shapes) if loaded else []
        points = normalized_rectangle_points(new_points)
        shapes = upsert_tracked_rectangle(
            shapes,
            label,
            track_id,
            points,
            group_id=group_id,
        )

        request = self._labelSaveRequest(img_path, shapes, loaded)
        request["imageHeight"] = img_shape[0]
        request["imageWidth"] = img_shape[1]
        return request

    def _saveTrackResult(
        self, img_path, label, track_id, group_id, new_points, img_shape
    ):
        request = self._trackResultRequest(
            img_path, label, track_id, group_id, new_points, img_shape
        )
        return self._saveLabelBatch([request], "Object Tracking")

    def _getSelectedRect(self, title):
        if len(self.canvas.selectedShapes) != 1:
            self.errorMessage(title, "Select exactly one rectangle to track.")
            return None
        shape = self.canvas.selectedShapes[0]
        if shape.shape_type != "rectangle" or len(shape.points) != 2:
            self.errorMessage(title, "Only rectangle bounding boxes can be tracked.")
            return None
        if shape.track_id is None or (
            isinstance(shape.track_id, str) and not shape.track_id.strip()
        ):
            self.errorMessage(title, "Assign a track ID before tracking this box.")
            return None
        try:
            normalized_rectangle_points(
                [[point.x(), point.y()] for point in shape.points]
            )
        except ValueError as exc:
            self.errorMessage(title, str(exc))
            return None
        return shape

    def _getTrackEndFrame(self, title, curr_index, total_frames):
        if curr_index + 1 >= total_frames:
            self.errorMessage(title, "The current frame is already the final frame.")
            return None
        end_frame, ok = QtWidgets.QInputDialog.getInt(
            self,
            title,
            f"Track to frame (current: {curr_index + 1}, total: {total_frames}):",
            total_frames,
            curr_index + 2,
            total_frames,
        )
        if not ok:
            return None
        return end_frame

    @staticmethod
    def _runCsrtTracking(
        image_paths,
        curr_index,
        end_frame,
        initial_bbox,
        cancel_event,
        report_progress,
    ):
        x, y, width, height = initial_bbox
        if width <= 0 or height <= 0:
            raise ValueError("The selected rectangle is too small to track.")
        report_progress(curr_index + 1, "Initializing CSRT tracker...")
        current_frame = load_oriented_cv_image(image_paths[curr_index])
        if current_frame is None:
            raise ValueError("Cannot read current frame image.")
        try:
            tracker = cv2.TrackerCSRT.create()
            initialized = tracker.init(current_frame, initial_bbox)
        except (AttributeError, cv2.error) as exc:
            raise RuntimeError("Cannot initialize CSRT: {}".format(exc)) from exc
        if initialized is False:
            raise RuntimeError("CSRT rejected the selected box.")

        frames = []
        last_tracked = curr_index
        stop_reason = "selected end frame reached"
        for index in range(curr_index + 1, end_frame):
            if cancel_event.is_set():
                stop_reason = "canceled"
                break
            report_progress(index + 1, "Tracking frame {}...".format(index + 1))
            frame = load_oriented_cv_image(image_paths[index])
            if frame is None:
                stop_reason = "frame {} could not be read".format(index + 1)
                break
            try:
                success, bbox = tracker.update(frame)
            except cv2.error as exc:
                stop_reason = "tracker error on frame {}: {}".format(index + 1, exc)
                break
            if not success:
                stop_reason = "tracking failed on frame {}".format(index + 1)
                break
            box_x, box_y, box_width, box_height = [float(value) for value in bbox]
            try:
                points = intersect_xyxy_with_image(
                    [box_x, box_y, box_x + box_width, box_y + box_height],
                    frame.shape[1],
                    frame.shape[0],
                )
            except ValueError:
                stop_reason = "tracker returned an empty box on frame {}".format(
                    index + 1
                )
                break
            frames.append(
                {
                    "image_path": image_paths[index],
                    "points": points,
                    "image_shape": frame.shape,
                }
            )
            last_tracked = index
        return {
            "frames": frames,
            "last_index": last_tracked,
            "stop_reason": stop_reason,
        }

    @staticmethod
    def _runBoTSORTTracking(
        image_paths,
        curr_index,
        end_frame,
        initial_box,
        use_refine,
        cancel_event,
        report_progress,
    ):
        from labelme.track_algo import BoTSORTForwardTracker

        tracker = None
        ai_model = None
        try:
            report_progress(curr_index + 1, "Loading YOLO + BoTSORT...")
            current_frame = load_oriented_cv_image(image_paths[curr_index])
            if current_frame is None:
                raise ValueError("Cannot read current frame image.")
            tracker = BoTSORTForwardTracker()
            if cancel_event.is_set():
                return {
                    "frames": [],
                    "last_index": curr_index,
                    "stop_reason": "canceled",
                }
            if not tracker.init(current_frame, initial_box):
                raise ValueError(
                    "No YOLO detection matched the selected box (IOU < 0.3). "
                    "Ensure the object is clearly visible."
                )

            if use_refine:
                report_progress(curr_index + 1, "Loading EfficientSAM...")
                model_definition = next(
                    model for model in MODELS if model.name == "EfficientSam (speed)"
                )
                ai_model = model_definition()

            frames = []
            last_tracked = curr_index
            stop_reason = "selected end frame reached"
            for index in range(curr_index + 1, end_frame):
                if cancel_event.is_set():
                    stop_reason = "canceled"
                    break
                report_progress(
                    index + 1, "Tracking frame {} with BoTSORT...".format(index + 1)
                )
                frame = load_oriented_cv_image(image_paths[index])
                if frame is None:
                    stop_reason = "frame {} could not be read".format(index + 1)
                    break
                success, xyxy = tracker.update(frame)
                if not success:
                    stop_reason = "tracking failed on frame {}".format(index + 1)
                    break
                box_x1, box_y1, box_x2, box_y2 = (
                    int(xyxy[0]),
                    int(xyxy[1]),
                    int(xyxy[2]),
                    int(xyxy[3]),
                )

                if ai_model is not None and not cancel_event.is_set():
                    ai_model.set_image(frame[:, :, ::-1])
                    mask = ai_model.predict_mask_from_box(
                        [box_x1, box_y1, box_x2, box_y2]
                    )
                    mask = None if mask is None else np.asarray(mask, dtype=bool)
                    if mask is not None and mask.ndim == 2 and mask.any():
                        ys, xs = np.where(mask)
                        box_x1, box_y1, box_x2, box_y2 = (
                            int(xs.min()),
                            int(ys.min()),
                            int(xs.max() + 1),
                            int(ys.max() + 1),
                        )

                if cancel_event.is_set():
                    stop_reason = "canceled"
                    break
                try:
                    points = intersect_xyxy_with_image(
                        [box_x1, box_y1, box_x2, box_y2],
                        frame.shape[1],
                        frame.shape[0],
                    )
                except ValueError:
                    stop_reason = "tracker returned an empty box on frame {}".format(
                        index + 1
                    )
                    break
                frames.append(
                    {
                        "image_path": image_paths[index],
                        "points": points,
                        "image_shape": frame.shape,
                    }
                )
                last_tracked = index
            return {
                "frames": frames,
                "last_index": last_tracked,
                "stop_reason": stop_reason,
            }
        finally:
            if tracker is not None:
                tracker.reset()
            if ai_model is not None:
                close = getattr(ai_model, "close", None)
                if close is not None:
                    close()

    def _startTrackingWorker(
        self, title, function, minimum, maximum, context, initial_message
    ):
        if self._tracking_thread is not None and self._tracking_thread.isRunning():
            self.errorMessage(title, "Another tracking operation is already running.")
            return False
        progress = QtWidgets.QProgressDialog(
            initial_message, "Cancel", minimum, maximum, self
        )
        progress.setWindowTitle(title)
        progress.setWindowModality(Qt.WindowModal)
        progress.setAutoClose(False)
        progress.setAutoReset(False)
        progress.setValue(minimum)

        worker = CancellableTrackingWorker(function)
        thread = QtCore.QThread(self)
        worker.moveToThread(thread)
        self._tracking_worker = worker
        self._tracking_thread = thread
        self._tracking_progress = progress
        self._tracking_context = context
        self._tracking_request_active = True

        thread.started.connect(worker.run)
        worker.progress.connect(self._trackingProgressUpdated)
        worker.finished.connect(self._trackingWorkerFinished)
        worker.failed.connect(self._trackingWorkerFailed)
        worker.finished.connect(worker.deleteLater)
        worker.failed.connect(worker.deleteLater)
        worker.finished.connect(thread.quit)
        worker.failed.connect(thread.quit)
        progress.canceled.connect(self._cancelTrackingWorker)
        thread.finished.connect(thread.deleteLater)
        thread.finished.connect(self._trackingWorkerCleanedUp)
        progress.show()
        thread.start()
        return True

    @QtCore.Slot()
    def _cancelTrackingWorker(self):
        if self._tracking_worker is not None:
            self._tracking_worker.cancel()
        if self._tracking_progress is not None:
            self._tracking_progress.setLabelText("Canceling tracking...")

    @QtCore.Slot(int, str)
    def _trackingProgressUpdated(self, value, message):
        if self._tracking_progress is not None:
            self._tracking_progress.setLabelText(message)
            self._tracking_progress.setValue(value)

    @QtCore.Slot(object)
    def _trackingWorkerFinished(self, result):
        self._tracking_request_active = False
        context = self._tracking_context or {}
        if self._close_after_tracking:
            return
        title = context.get("title", "Object Tracking")
        stop_reason = result.get("stop_reason", "completed")
        if stop_reason == "canceled":
            self.informationMessage(title, "Tracking was canceled; no files changed.")
            return

        try:
            requests = [
                self._trackResultRequest(
                    frame["image_path"],
                    context["label"],
                    context["track_id"],
                    context["group_id"],
                    frame["points"],
                    frame["image_shape"],
                )
                for frame in result.get("frames", [])
            ]
        except (LabelFileError, ValueError) as exc:
            self.errorMessage(title, str(exc))
            return
        if requests and not self._saveLabelBatch(requests, title):
            return

        source_filename = context.get("source_filename")
        if source_filename == self.filename:
            self.loadFile(source_filename)
        tracked_count = len(requests)
        last_index = result.get("last_index", context.get("start_index", 0))
        self.informationMessage(
            title,
            "Tracked {}-{} forward {} frames (frame {} to {}); {}.".format(
                context.get("label"),
                context.get("track_id"),
                tracked_count,
                context.get("start_index", 0) + 1,
                last_index + 1,
                stop_reason,
            ),
        )

    @QtCore.Slot(object)
    def _trackingWorkerFailed(self, error):
        self._tracking_request_active = False
        if self._close_after_tracking:
            return
        context = self._tracking_context or {}
        title = context.get("title", "Object Tracking")
        help_text = context.get("failure_help", "")
        message = str(error)
        if help_text:
            message = "{}\n{}".format(message, help_text)
        self.errorMessage(title, message)

    @QtCore.Slot()
    def _trackingWorkerCleanedUp(self):
        thread = self.sender()
        if self._tracking_thread is thread:
            if self._tracking_progress is not None:
                self._tracking_progress.close()
            self._tracking_thread = None
            self._tracking_worker = None
            self._tracking_progress = None
            self._tracking_context = None
        if self._close_after_tracking and not self._tracking_request_active:
            QtCore.QTimer.singleShot(0, self.close)

    def trackForward(self):
        if not self.imageList or self.filename not in self.imageList:
            self.errorMessage("Track Forward", "Open a frame sequence before tracking.")
            return
        if not self._ensureSavedForWorkflow("Track Forward"):
            return
        shape = self._getSelectedRect("Track Forward")
        if shape is None:
            return

        curr_index = self.imageList.index(self.filename)
        end_frame = self._getTrackEndFrame(
            "Track Forward", curr_index, len(self.imageList)
        )
        if end_frame is None:
            return

        label = shape.label
        track_id = shape.track_id
        group_id = shape.group_id
        p1, p2 = shape.points[0], shape.points[1]
        x = int(min(p1.x(), p2.x()))
        y = int(min(p1.y(), p2.y()))
        w = int(abs(p2.x() - p1.x()))
        h = int(abs(p2.y() - p1.y()))

        function = functools.partial(
            self._runCsrtTracking,
            tuple(self.imageList),
            curr_index,
            end_frame,
            (x, y, w, h),
        )
        self._startTrackingWorker(
            "Track Forward",
            function,
            curr_index + 1,
            end_frame,
            {
                "title": "Track Forward",
                "label": label,
                "track_id": track_id,
                "group_id": group_id,
                "source_filename": self.filename,
                "start_index": curr_index,
            },
            "Initializing CSRT tracker...",
        )

    def trackForwardBoTSORT(self):
        if not self.imageList or self.filename not in self.imageList:
            self.errorMessage(
                "Track Forward (BoTSORT)",
                "Open a frame sequence before tracking.",
            )
            return
        if not self._ensureSavedForWorkflow("Track Forward (BoTSORT)"):
            return
        shape = self._getSelectedRect("Track Forward (BoTSORT)")
        if shape is None:
            return

        curr_index = self.imageList.index(self.filename)
        total_frames = len(self.imageList)
        if curr_index + 1 >= total_frames:
            self.errorMessage(
                "Track Forward (BoTSORT)",
                "The current frame is already the final frame.",
            )
            return

        dialog = QtWidgets.QDialog(self)
        dialog.setWindowTitle("Track Forward (BoTSORT)")
        layout = QtWidgets.QFormLayout(dialog)
        end_spin = QtWidgets.QSpinBox()
        end_spin.setRange(curr_index + 2, total_frames)
        end_spin.setValue(total_frames)
        layout.addRow("Track to frame:", end_spin)
        refine_check = QtWidgets.QCheckBox("Refine with EfficientSAM")
        layout.addRow(refine_check)
        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel
        )
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        layout.addRow(buttons)
        if dialog.exec_() != QtWidgets.QDialog.Accepted:
            return

        end_frame = end_spin.value()
        use_refine = refine_check.isChecked()

        label = shape.label
        track_id = shape.track_id
        group_id = shape.group_id
        p1, p2 = shape.points[0], shape.points[1]
        x1 = int(min(p1.x(), p2.x()))
        y1 = int(min(p1.y(), p2.y()))
        x2 = int(max(p1.x(), p2.x()))
        y2 = int(max(p1.y(), p2.y()))

        function = functools.partial(
            self._runBoTSORTTracking,
            tuple(self.imageList),
            curr_index,
            end_frame,
            [x1, y1, x2, y2],
            use_refine,
        )
        self._startTrackingWorker(
            "Track Forward (BoTSORT)",
            function,
            curr_index + 1,
            end_frame,
            {
                "title": "Track Forward (BoTSORT)",
                "label": label,
                "track_id": track_id,
                "group_id": group_id,
                "source_filename": self.filename,
                "start_index": curr_index,
                "failure_help": (
                    "Install the optional ultralytics dependencies and configure "
                    "a valid YOLO model."
                ),
            },
            "Loading YOLO + BoTSORT...",
        )

    def refineBboxAI(self):
        shapes = [
            s
            for s in self.canvas.selectedShapes
            if s.shape_type == "rectangle" and len(s.points) == 2
        ]
        if not shapes:
            self.errorMessage(
                "Refine Bbox (AI)",
                "Select at least one rectangle to refine.",
            )
            return

        refinements = []
        try:
            self.canvas.initializeAiModel("EfficientSam (speed)")
            ai_model = self.canvas._ai_model
            for shape in shapes:
                p1, p2 = shape.points[0], shape.points[1]
                box = [
                    min(p1.x(), p2.x()),
                    min(p1.y(), p2.y()),
                    max(p1.x(), p2.x()),
                    max(p1.y(), p2.y()),
                ]
                mask = ai_model.predict_mask_from_box(box)
                if mask is None or not np.asarray(mask).any():
                    continue
                ys, xs = np.where(mask)
                refinements.append(
                    (
                        shape,
                        QtCore.QPointF(float(xs.min()), float(ys.min())),
                        QtCore.QPointF(float(xs.max() + 1), float(ys.max() + 1)),
                    )
                )
        except Exception as exc:
            self.errorMessage("Refine Bbox (AI)", str(exc))
            return
        finally:
            self.canvas.releaseAiModel()
        if not refinements:
            self.status("The AI model returned no usable masks.", delay=8000)
            return
        for shape, top_left, bottom_right in refinements:
            shape.points[0] = top_left
            shape.points[1] = bottom_right
        self.canvas.storeShapes()
        self.canvas.update()
        self.setDirty()

    def startHostedSam2PointPrompt(self):
        if not self._hosted_sam2_client.is_configured():
            self.errorMessage(
                "Hosted SAM2",
                "Set hosted_sam2.url in the config or "
                "LABELME_HOSTED_SAM2_URL in the environment.",
            )
            return
        if self.image.isNull() or self.imageData is None:
            self.errorMessage("Hosted SAM2", "Open an image before point prompting.")
            return
        if self._hosted_sam2_request_active:
            self.status("Hosted SAM2 request already in progress.")
            return

        self.setEditMode()
        self.canvas.cancelPointPrompt(emit_signal=False)
        frame_key = self._hostedSam2FrameKey()
        cached_image = self._hostedSam2CachedImage(frame_key)
        if cached_image is not None:
            self._armHostedSam2PointPrompt()
            return

        image_data = bytes(self.imageData)
        self.status("Registering current frame with hosted SAM2...")
        self._runHostedSam2Request(
            self._hosted_sam2_client.register_image,
            self._hostedSam2ImageRegistered,
            image_data,
            client_frame_key=frame_key,
            _context={"frame_key": frame_key, "image_data": image_data},
        )

    def _hostedSam2FrameKey(self):
        if self.imageData is None:
            return None
        image_hash = hashlib.sha256(self.imageData).hexdigest()
        image_path = self.imagePath or self.filename or ""
        return "{}:{}x{}:{}".format(
            osp.abspath(str(image_path)),
            self.image.width(),
            self.image.height(),
            image_hash,
        )

    def _armHostedSam2PointPrompt(self):
        if self.canvas.armPointPrompt():
            self.status("SAM2 point prompt armed. Click an object.")
        else:
            self.errorMessage(
                "Hosted SAM2",
                "Cannot arm point prompt without an image.",
            )

    def _hostedSam2CachedImage(self, frame_key):
        image_info = self._hosted_sam2_image_cache.get(frame_key)
        if image_info is not None:
            self._hosted_sam2_image_cache.move_to_end(frame_key)
        return image_info

    def _rememberHostedSam2Image(self, frame_key, response):
        self._hosted_sam2_image_cache[frame_key] = response
        self._hosted_sam2_image_cache.move_to_end(frame_key)
        while len(self._hosted_sam2_image_cache) > self._hosted_sam2_max_cached_frames:
            self._hosted_sam2_image_cache.popitem(last=False)

    def _hostedSam2ImageRegistered(self, response, context):
        frame_key = context["frame_key"]
        if frame_key != self._hostedSam2FrameKey():
            self.status("Ignored hosted SAM2 registration for a stale frame.")
            return
        if (
            response["width"] != self.image.width()
            or response["height"] != self.image.height()
        ):
            self.errorMessage(
                "Hosted SAM2",
                "Backend image dimensions do not match the current frame.",
            )
            return
        self._rememberHostedSam2Image(frame_key, response)
        retry_point = context.get("retry_point")
        if retry_point is not None:
            self._sendHostedSam2PointPrompt(
                frame_key,
                retry_point[0],
                retry_point[1],
                retry_count=1,
                image_data=context.get("image_data"),
            )
            return
        self._armHostedSam2PointPrompt()

    def _hostedSam2PointPrompt(self, point):
        frame_key = self._hostedSam2FrameKey()
        image_info = self._hostedSam2CachedImage(frame_key)
        if image_info is None:
            self.errorMessage("Hosted SAM2", "Current frame is not registered.")
            return
        if self._hosted_sam2_request_active:
            self.status("Hosted SAM2 request already in progress.")
            return

        self._sendHostedSam2PointPrompt(frame_key, point.x(), point.y())

    def _sendHostedSam2PointPrompt(
        self, frame_key, x, y, retry_count=0, image_data=None
    ):
        image_info = self._hostedSam2CachedImage(frame_key)
        if image_info is None:
            self.errorMessage("Hosted SAM2", "Current frame is not registered.")
            return
        if frame_key != self._hostedSam2FrameKey():
            self.status("Ignored hosted SAM2 prompt for a stale frame.")
            return
        if image_data is None:
            image_data = bytes(self.imageData)
        self.status("Sending SAM2 point prompt...")
        self._runHostedSam2Request(
            self._hosted_sam2_client.point_prompt,
            self._hostedSam2PointPromptFinished,
            image_info["image_id"],
            x,
            y,
            1,
            _context={
                "kind": "point_prompt",
                "frame_key": frame_key,
                "point": (float(x), float(y)),
                "retry_count": retry_count,
                "image_data": image_data,
            },
        )

    def _hostedSam2PointPromptCancelled(self):
        self.status("SAM2 point prompt cancelled.")

    def _hostedSam2PointPromptFinished(self, response, context):
        if context["frame_key"] != self._hostedSam2FrameKey():
            self.status("Ignored hosted SAM2 bbox for a stale frame.")
            return

        bbox = response["bbox"]
        width = self.image.width()
        height = self.image.height()
        try:
            (x1, y1), (x2, y2) = intersect_xyxy_with_image(bbox, width, height)
        except ValueError:
            self.errorMessage("Hosted SAM2", "Hosted SAM2 returned an empty bbox.")
            return

        metadata = self._promptForNewShapeMetadata()
        if metadata is None:
            self.setEditMode()
            return

        text_label, flags, group_id, description, text_id = metadata
        shape = Shape(label=text_label, shape_type="rectangle", flags=flags)
        shape.addPoint(QtCore.QPointF(x1, y1))
        shape.addPoint(QtCore.QPointF(x2, y2))
        shape.close()
        self.canvas.shapes.append(shape)
        self.canvas.storeShapes()
        self.canvas.update()
        self._finishNewShape(shape, group_id, text_id, description)
        if not self._hosted_sam2_cache_frames:
            self._hosted_sam2_image_cache.pop(context["frame_key"], None)
        self.status("SAM2 bbox added.")

    def _runHostedSam2Request(self, function, on_success, *args, **kwargs):
        context = kwargs.pop("_context", {})
        worker = HostedSam2RequestWorker(function, *args, **kwargs)
        thread = QtCore.QThread(self)
        worker.moveToThread(thread)
        self._hosted_sam2_worker = worker
        self._hosted_sam2_thread = thread
        self._hosted_sam2_request_active = True
        self._hosted_sam2_request_context = context
        self._hosted_sam2_on_success = on_success

        thread.started.connect(worker.run)
        worker.finished.connect(self._hostedSam2WorkerFinished)
        worker.failed.connect(self._hostedSam2RequestFailed)
        worker.finished.connect(worker.deleteLater)
        worker.failed.connect(worker.deleteLater)
        worker.finished.connect(thread.quit)
        worker.failed.connect(thread.quit)
        thread.finished.connect(thread.deleteLater)
        thread.finished.connect(self._hostedSam2RequestCleanedUp)
        thread.start()

    @QtCore.Slot(object)
    def _hostedSam2WorkerFinished(self, result):
        on_success = self._hosted_sam2_on_success
        context = self._hosted_sam2_request_context or {}
        self._hosted_sam2_request_active = False
        if self._close_after_hosted_request:
            return
        try:
            on_success(result, context)
        except Exception as exc:
            logger.exception("Hosted SAM2 response handling failed")
            self.errorMessage("Hosted SAM2", str(exc))

    @QtCore.Slot(object)
    def _hostedSam2RequestFailed(self, error):
        self._hosted_sam2_request_active = False
        context = self._hosted_sam2_request_context or {}
        if self._close_after_hosted_request:
            return
        if (
            isinstance(error, HostedSam2Error)
            and error.status_code == 404
            and context.get("kind") == "point_prompt"
            and context.get("retry_count", 0) == 0
        ):
            frame_key = context["frame_key"]
            self._hosted_sam2_image_cache.pop(frame_key, None)
            if frame_key != self._hostedSam2FrameKey():
                self.status("Ignored hosted SAM2 retry for a stale frame.")
                return
            image_data = context.get("image_data")
            if image_data is None:
                self.errorMessage(
                    "Hosted SAM2", "Cannot retry because the frame data is unavailable."
                )
                return
            self.status("SAM2 cache expired; registering the frame again...")
            self._runHostedSam2Request(
                self._hosted_sam2_client.register_image,
                self._hostedSam2ImageRegistered,
                image_data,
                client_frame_key=frame_key,
                _context={
                    "frame_key": frame_key,
                    "retry_point": context["point"],
                    "image_data": image_data,
                },
            )
            return
        self.errorMessage("Hosted SAM2", str(error))

    @QtCore.Slot()
    def _hostedSam2RequestCleanedUp(self):
        thread = self.sender()
        if self._hosted_sam2_thread is thread:
            self._hosted_sam2_thread = None
            self._hosted_sam2_worker = None
            self._hosted_sam2_request_context = None
            self._hosted_sam2_on_success = None
        if self._close_after_hosted_request and not self._hosted_sam2_request_active:
            QtCore.QTimer.singleShot(0, self.close)

    def editID(self, item=None):
        if item and not isinstance(item, IDListWidgetItem):
            raise TypeError("item must be IDListWidgetItem type")

        if not self.canvas.editing():
            return
        if not item:
            _, item = self.currentItem()
        if item is None:
            return
        shape = item.shape()
        if shape is None:
            return
        current_id = "" if shape.track_id is None else shape.track_id
        id = self.IDDialog.popUp(text=current_id)
        if id is None:
            return
        if not str(id).strip():
            self.errorMessage(
                self.tr("Invalid ID"),
                self.tr("Invalid ID '{}' with validation type '{}'").format(
                    id, self._config["validate_label"]
                ),
            )
            return
        shape.track_id = id
        shape.group_id = int(id) if id.isdigit() else id

        item.setText(shape.track_id)
        self._update_shape_color(shape)
        self.canvas.storeShapes()
        self.setDirty()

    def editLabel(self, item=None):
        if item and not isinstance(item, LabelListWidgetItem):
            raise TypeError("item must be LabelListWidgetItem type")
        if not self.canvas.editing():
            return
        if not item:
            item, _ = self.currentItem()
        if item is None:
            return
        shape = item.shape()
        if shape is None:
            return
        previous_group_id = shape.group_id
        text, flags, group_id, description = self.labelDialog.popUp(
            text=shape.label,
            flags=shape.flags,
            group_id=shape.group_id,
            description=shape.description,
        )
        if text is None:
            return
        if not self.validateLabel(text):
            self.errorMessage(
                self.tr("Invalid label"),
                self.tr("Invalid label '{}' with validation type '{}'").format(
                    text, self._config["validate_label"]
                ),
            )
            return
        shape.label = text
        shape.flags = flags
        shape.group_id = group_id
        if group_id != previous_group_id:
            shape.track_id = group_id
        shape.description = description

        self._update_shape_color(shape)
        if shape.group_id is None:
            item.setText(
                '{} <font color="#{:02x}{:02x}{:02x}">●</font>'.format(
                    html.escape(shape.label), *shape.fill_color.getRgb()[:3]
                )
            )
        else:
            item.setText("{} ({})".format(shape.label, shape.group_id))
        try:
            id_item = self.IDList.findItemByShape(shape)
            id_item.setText(str(shape.track_id))
        except ValueError:
            pass
        self.canvas.storeShapes()
        self.setDirty()
        self._ensureLabelSelector(shape.label)

    def fileSearchChanged(self):
        pattern = self.fileSearch.text()
        try:
            matcher = re.compile(pattern, re.IGNORECASE) if pattern else None
        except re.error:
            matcher = re.compile(re.escape(pattern), re.IGNORECASE)
        for index in range(self.fileListWidget.count()):
            item = self.fileListWidget.item(index)
            item.setHidden(bool(matcher and not matcher.search(item.text())))

    def _restoreCurrentFileSelection(self):
        self.fileListWidget.blockSignals(True)
        try:
            if self.filename in self.imageList:
                self.fileListWidget.setCurrentRow(self.imageList.index(self.filename))
            else:
                self.fileListWidget.clearSelection()
        finally:
            self.fileListWidget.blockSignals(False)

    def fileSelectionChanged(self):
        items = self.fileListWidget.selectedItems()
        if not items:
            return
        item = items[0]

        if not self.mayContinue():
            self._restoreCurrentFileSelection()
            return

        if self.mode == "None":
            self.mode = "NORMAL"

        if self.mode == "TRACK INTERPOLATION":
            currIndex = self.imageList.index(str(item.text()))
            if currIndex < len(self.imageList):
                filename = self.imageList[currIndex]

            if filename in self.INTERPOLATION_list:
                getIndex = self.imageList.index(filename) + 1
                interpolationIndex = self.INTERPOLATION_list.index(self.filename) + 1
                self.navigation_list.statusBar.showMessage(
                    (
                        f"Status: {getIndex}/{len(self.imageList)} | "
                        f"Mode: {self.mode} - "
                        f"({interpolationIndex}/{len(self.INTERPOLATION_list)})"
                    )
                )

                if not self.loadFile(filename):
                    self._restoreCurrentFileSelection()
                    return
            else:
                self.errorMessage(
                    "Box Interpolation",
                    (
                        "You cannot select out-of-list frame. Use Previous (A) "
                        "and Next (D) buttons to move between the selected frames"
                    ),
                )
                self._restoreCurrentFileSelection()
                return

        else:
            currIndex = self.imageList.index(str(item.text()))
            if currIndex < len(self.imageList):
                filename = self.imageList[currIndex]
                if filename:
                    if not self.loadFile(filename):
                        self._restoreCurrentFileSelection()
                        return

            getIndex = self.imageList.index(self.filename) + 1
            self.navigation_list.statusBar.showMessage(
                f"Status: {getIndex}/{len(self.imageList)} | Mode: {self.mode}"
            )

    # React to canvas signals.
    def shapeSelectionChanged(self, selected_shapes):
        self._noSelectionSlot = True
        for shape in self.canvas.selectedShapes:
            shape.selected = False
        self.labelList.clearSelection()
        self.IDList.clearSelection()
        self.canvas.selectedShapes = selected_shapes
        for shape in self.canvas.selectedShapes:
            shape.selected = True
            try:
                item = self.labelList.findItemByShape(shape)
                self.labelList.selectItem(item)
                self.labelList.scrollToItem(item)
                id_item = self.IDList.findItemByShape(shape)
                self.IDList.selectItem(id_item)
                self.IDList.scrollToItem(id_item)
            except ValueError:
                pass
        self._noSelectionSlot = False
        n_selected = len(selected_shapes)
        self.actions.delete.setEnabled(n_selected)
        self.actions.duplicate.setEnabled(n_selected)
        self.actions.copy.setEnabled(n_selected)
        self.actions.edit.setEnabled(n_selected == 1)
        self.actions.edit_id.setEnabled(n_selected == 1)

    def addLabel(self, shape):
        if shape.group_id is None:
            text = shape.label
        else:
            text = "{} ({})".format(shape.label, shape.group_id)
        label_list_item = LabelListWidgetItem(text, shape)
        self.labelList.addItem(label_list_item)
        track_id_text = "" if shape.track_id is None else str(shape.track_id)
        id_list_item = IDListWidgetItem(track_id_text, shape)
        self.IDList.addItem(id_list_item)
        self._ensureLabelSelector(shape.label)
        self.labelDialog.addLabelHistory(shape.label)
        self.IDDialog.addIDHistory(shape.track_id)
        for action in self.actions.onShapesPresent:
            action.setEnabled(True)
        self._update_shape_color(shape)
        label_list_item.setText(
            '{} <font color="#{:02x}{:02x}{:02x}">●</font>'.format(
                html.escape(text), *shape.fill_color.getRgb()[:3]
            )
        )

    def _update_shape_color(self, shape):
        unique_name = shape.label + "_" + str(shape.track_id)
        r, g, b = self._get_rgb_by_label(unique_name, add_to_selector=False)
        shape.line_color = QtGui.QColor(r, g, b)
        shape.vertex_fill_color = QtGui.QColor(r, g, b)
        shape.hvertex_fill_color = QtGui.QColor(255, 255, 255)
        shape.fill_color = QtGui.QColor(r, g, b, 128)
        shape.select_line_color = QtGui.QColor(255, 255, 255)
        shape.select_fill_color = QtGui.QColor(r, g, b, 155)

    def _ensureLabelSelector(self, label):
        if self.uniqLabelList.findItemByLabel(label) is None:
            item = self.uniqLabelList.createItemFromLabel(label)
            self.uniqLabelList.addItem(item)
            rgb = self._get_rgb_by_label(label)
            self.uniqLabelList.setItemLabel(item, label, rgb)

    def _get_rgb_by_label(self, label, add_to_selector=True):
        if self._config["shape_color"] == "auto":
            if not add_to_selector:
                digest = hashlib.sha256(label.encode("utf-8")).digest()
                label_id = int.from_bytes(digest[:4], "big")
                label_id += self._config["shift_auto_shape_color"]
                return LABEL_COLORMAP[label_id % len(LABEL_COLORMAP)]
            item = self.uniqLabelList.findItemByLabel(label)
            if item is None:
                item = self.uniqLabelList.createItemFromLabel(label)
                self.uniqLabelList.addItem(item)
                rgb = self._get_rgb_by_label(label)
                self.uniqLabelList.setItemLabel(item, label, rgb)
            label_id = self.uniqLabelList.indexFromItem(item).row() + 1
            label_id += self._config["shift_auto_shape_color"]
            return LABEL_COLORMAP[label_id % len(LABEL_COLORMAP)]
        elif (
            self._config["shape_color"] == "manual" and self._config["label_colors"]
            # and label in self._config["label_colors"]
        ):
            # return self._config["label_colors"][label]
            suffix = label.split("_")[-1]
            if suffix == "None":
                return [224, 224, 0]
            try:
                idx = int(suffix)
                return self._config["label_colors"][idx][idx]
            except (ValueError, IndexError, KeyError):
                return (0, 255, 0)
        elif self._config["default_shape_color"]:
            return self._config["default_shape_color"]
        return (0, 255, 0)

    def remLabels(self, shapes):
        for shape in shapes:
            try:
                item = self.labelList.findItemByShape(shape)
                self.labelList.removeItem(item)
            except ValueError:
                pass
            try:
                item = self.IDList.findItemByShape(shape)
                self.IDList.removeItem(item)
            except ValueError:
                pass

    def loadShapes(self, shapes, replace=True):
        self._noSelectionSlot = True
        self.labelList.blockSignals(True)
        self.IDList.blockSignals(True)
        for shape in shapes:
            self.addLabel(shape)
        self.labelList.blockSignals(False)
        self.IDList.blockSignals(False)
        self.labelList.clearSelection()
        self.IDList.clearSelection()
        self._noSelectionSlot = False
        self.canvas.loadShapes(shapes, replace=replace)

    def _deserializeShapes(self, shapes):
        def validate_geometry(shape_type, points, shape_number):
            exact_counts = {
                "rectangle": 2,
                "circle": 2,
                "line": 2,
                "point": 1,
                "mask": 2,
            }
            minimum_counts = {"polygon": 3, "linestrip": 2, "points": 1}
            if shape_type in exact_counts and len(points) != exact_counts[shape_type]:
                raise ValueError(
                    "shape {} of type {} must have exactly {} points".format(
                        shape_number, shape_type, exact_counts[shape_type]
                    )
                )
            if (
                shape_type in minimum_counts
                and len(points) < minimum_counts[shape_type]
            ):
                raise ValueError(
                    "shape {} of type {} must have at least {} points".format(
                        shape_number, shape_type, minimum_counts[shape_type]
                    )
                )
            if shape_type in {"rectangle", "mask"}:
                try:
                    normalized_rectangle_points(points)
                except ValueError as exc:
                    raise ValueError(
                        "shape {} has invalid {} geometry: {}".format(
                            shape_number, shape_type, exc
                        )
                    ) from exc
            elif shape_type == "circle" and points[0] == points[1]:
                raise ValueError(
                    "shape {} circle must have a positive radius".format(shape_number)
                )

        deserialized = []
        for index, shape_data in enumerate(shapes):
            if not isinstance(shape_data, dict):
                raise ValueError("shape {} must be an object".format(index + 1))
            label = shape_data["label"]
            if not isinstance(label, str) or not label:
                raise ValueError(
                    "shape {} must have a non-empty string label".format(index + 1)
                )
            points = shape_data["points"]
            shape_type = shape_data["shape_type"]
            if shape_type not in {
                "polygon",
                "rectangle",
                "point",
                "line",
                "circle",
                "linestrip",
                "points",
                "mask",
            }:
                raise ValueError(
                    "shape {} has unsupported type: {}".format(index + 1, shape_type)
                )
            flags = shape_data["flags"]
            if not isinstance(flags, dict):
                raise ValueError("shape {} flags must be an object".format(index + 1))
            description = shape_data.get("description", "")
            group_id = shape_data.get("group_id")
            track_id = shape_data.get("track_id")
            other_data_value = shape_data.get("other_data", {})
            if not isinstance(other_data_value, dict):
                raise ValueError(
                    "shape {} metadata must be an object".format(index + 1)
                )
            other_data = dict(other_data_value)
            if track_id is None or track_id == "":
                track_id = group_id

            if not isinstance(points, (list, tuple)):
                raise ValueError("shape {} points must be a sequence".format(index + 1))
            validated_points = []
            for point in points:
                if not isinstance(point, (list, tuple)) or len(point) != 2:
                    raise ValueError(
                        "shape {} points must be coordinate pairs".format(index + 1)
                    )
                if any(isinstance(value, bool) for value in point):
                    raise ValueError(
                        "shape {} coordinates must be finite numbers".format(index + 1)
                    )
                try:
                    x, y = (float(value) for value in point)
                except (TypeError, ValueError) as exc:
                    raise ValueError(
                        "shape {} coordinates must be finite numbers".format(index + 1)
                    ) from exc
                if not math.isfinite(x) or not math.isfinite(y):
                    raise ValueError(
                        "shape {} coordinates must be finite numbers".format(index + 1)
                    )
                validated_points.append([x, y])
            points = validated_points
            validate_geometry(shape_type, points, index + 1)

            mask = shape_data["mask"]
            if shape_type == "mask":
                if mask is None:
                    raise ValueError("shape {} mask data is required".format(index + 1))
                try:
                    mask = np.asarray(mask, dtype=bool)
                except (TypeError, ValueError) as exc:
                    raise ValueError(
                        "shape {} mask must be a non-empty 2D array".format(index + 1)
                    ) from exc
                if mask.ndim != 2 or mask.size == 0 or not mask.any():
                    raise ValueError(
                        "shape {} mask must be a non-empty 2D array".format(index + 1)
                    )

            if (
                self.ir_activated
                and label == self.ir_name
                and str(track_id) == str(self.ir_id)
            ):
                if len(points) != 2:
                    raise ValueError(
                        "shape {} cannot be refined because it is not a box".format(
                            index + 1
                        )
                    )
                deltas = [
                    [
                        self.ir_mod_shape[0][0] - self.ir_old_shape[0][0],
                        self.ir_mod_shape[0][1] - self.ir_old_shape[0][1],
                    ],
                    [
                        self.ir_mod_shape[1][0] - self.ir_old_shape[1][0],
                        self.ir_mod_shape[1][1] - self.ir_old_shape[1][1],
                    ],
                ]

                points = [
                    [
                        points[0][0] + deltas[0][0],
                        points[0][1] + deltas[0][1],
                    ],
                    [
                        points[1][0] + deltas[1][0],
                        points[1][1] + deltas[1][1],
                    ],
                ]
                validate_geometry(shape_type, points, index + 1)

            shape = Shape(
                label=label,
                shape_type=shape_type,
                group_id=group_id,
                track_id=track_id,
                description=description,
                mask=mask,
            )
            shape.other_data = other_data
            for x, y in points:
                shape.addPoint(QtCore.QPointF(x, y))
            shape.close()

            default_flags = {}
            if self._config["label_flags"]:
                for pattern, keys in self._config["label_flags"].items():
                    if re.match(pattern, label):
                        for key in keys:
                            default_flags[key] = False
            shape.flags = default_flags
            shape.flags.update(flags)

            deserialized.append(shape)
        return deserialized

    def loadLabels(self, shapes):
        self.loadShapes(self._deserializeShapes(shapes))

    def loadFlags(self, flags):
        self.flag_widget.clear()
        for key, flag in flags.items():
            item = QtWidgets.QListWidgetItem(key)
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            item.setCheckState(Qt.Checked if flag else Qt.Unchecked)
            self.flag_widget.addItem(item)

    def saveLabels(self, filename):
        def format_shape(s):
            return dict(
                label=s.label.encode("utf-8") if PY2 else s.label,
                points=[(p.x(), p.y()) for p in s.points],
                group_id=s.group_id,
                track_id=s.track_id,
                description=s.description,
                shape_type=s.shape_type,
                flags=s.flags,
                mask=None if s.mask is None else utils.img_arr_to_b64(s.mask),
                other_data=s.other_data.copy(),
            )

        shapes = [format_shape(item.shape()) for item in self.labelList]
        flags = {}
        for i in range(self.flag_widget.count()):
            item = self.flag_widget.item(i)
            key = item.text()
            flag = item.checkState() == Qt.Checked
            flags[key] = flag
        try:
            label_dir = osp.dirname(filename)
            imagePath = osp.abspath(self.imagePath)
            try:
                relative_image_path = osp.relpath(imagePath, label_dir)
            except ValueError:
                relative_image_path = None
            if relative_image_path is not None and osp.exists(
                osp.join(label_dir, relative_image_path)
            ):
                imagePath = relative_image_path
            preserve_embedded_data = bool(
                self.labelFile is not None and self.labelFile.imageDataEmbedded
            )
            imageData = (
                self.imageData
                if self._config["store_data"] or preserve_embedded_data
                else None
            )
            request = dict(
                filename=filename,
                shapes=shapes,
                imagePath=imagePath,
                imageData=imageData,
                imageHeight=self.image.height(),
                imageWidth=self.image.width(),
                otherData=self.otherData or {},
                flags=flags,
            )
            legacy_sources = self._legacyAnnotationSources(
                self.filename, self.labelFile, filename
            )
            if legacy_sources:
                request["_retire_sources"] = legacy_sources
            save_label_files_atomically([request])
            self.labelFile = LabelFile(filename)
            items = self.fileListWidget.findItems(self.filename, Qt.MatchExactly)
            if len(items) > 0:
                if len(items) != 1:
                    raise RuntimeError("There are duplicate files.")
                items[0].setCheckState(Qt.Checked)
            # disable allows next and previous image to proceed
            # self.filename = filename
            return True
        except (LabelFileError, OSError, RuntimeError, ValueError) as e:
            self.errorMessage(
                self.tr("Error saving label data"), self.tr("<b>%s</b>") % e
            )
            return False

    def duplicateSelectedShape(self):
        added_shapes = self.canvas.duplicateSelectedShapes()
        for shape in added_shapes:
            self.addLabel(shape)
        self.setDirty()

    def pasteSelectedShape(self):
        self.loadShapes(self._copied_shapes, replace=False)
        self.setDirty()

    def copySelectedShape(self):
        self._copied_shapes = [s.copy() for s in self.canvas.selectedShapes]
        self.actions.paste.setEnabled(len(self._copied_shapes) > 0)

    def labelSelectionChanged(self):
        if self._noSelectionSlot:
            return
        if self.canvas.editing():
            selected_shapes = []
            for item in self.labelList.selectedItems():
                selected_shapes.append(item.shape())
            if selected_shapes:
                self.canvas.selectShapes(selected_shapes)
            else:
                self.canvas.deSelectShape()

    def IDSelectionChanged(self):
        if self._noSelectionSlot:
            return
        if self.canvas.editing():
            selected_shapes = []
            for item in self.IDList.selectedItems():
                selected_shapes.append(item.shape())
            if selected_shapes:
                self.canvas.selectShapes(selected_shapes)
            else:
                self.canvas.deSelectShape()

    def labelItemChanged(self, item):
        self._setShapeVisibility(item.shape(), item.checkState(), self.IDList)

    def IDItemChanged(self, item):
        self._setShapeVisibility(item.shape(), item.checkState(), self.labelList)

    def _setShapeVisibility(self, shape, check_state, counterpart_list):
        if self._syncing_visibility:
            return
        self._syncing_visibility = True
        try:
            try:
                counterpart = counterpart_list.findItemByShape(shape)
            except ValueError:
                counterpart = None
            if counterpart is not None and counterpart.checkState() != check_state:
                counterpart.setCheckState(check_state)
            self.canvas.setShapeVisible(shape, check_state == Qt.Checked)
        finally:
            self._syncing_visibility = False

    def labelOrderChanged(self):
        self._shapeOrderChanged(self.labelList)

    def IDOrderChanged(self):
        self._shapeOrderChanged(self.IDList)

    def _shapeOrderChanged(self, source_list):
        shapes = [item.shape() for item in source_list]
        source_ids = [id(shape) for shape in shapes if shape is not None]
        canvas_ids = [id(shape) for shape in self.canvas.shapes]
        if (
            len(source_ids) != len(shapes)
            or len(set(source_ids)) != len(source_ids)
            or len(source_ids) != len(canvas_ids)
            or set(source_ids) != set(canvas_ids)
        ):
            logger.error("Ignoring an invalid shape-list reorder")
            self._rebuildShapeLists(self.canvas.shapes)
            self.status(self.tr("Could not apply the requested shape order."))
            return

        # The canvas order is the authoritative order serialized by the app.
        # Preserve its undo history, visibility, and selection while rebuilding
        # both views because Qt's internal move detaches and clones list items.
        self.canvas.shapes = list(shapes)
        self.canvas.storeShapes()
        self.canvas.update()
        self._rebuildShapeLists(shapes)
        self.setDirty()

    def _rebuildShapeLists(self, shapes):
        shape_ids = {id(shape) for shape in shapes}
        selected_shapes = [
            shape for shape in self.canvas.selectedShapes if id(shape) in shape_ids
        ]
        visibility = {
            id(shape): self.canvas.isVisible(shape) for shape in self.canvas.shapes
        }
        signal_objects = (
            self.labelList,
            self.labelList.model(),
            self.IDList,
            self.IDList.model(),
        )
        previous_signal_states = [
            (obj, obj.blockSignals(True)) for obj in signal_objects
        ]
        previous_selection_guard = self._noSelectionSlot
        self._noSelectionSlot = True
        try:
            self.labelList.clear()
            self.IDList.clear()
            for shape in shapes:
                self.addLabel(shape)
                check_state = (
                    Qt.Checked if visibility.get(id(shape), True) else Qt.Unchecked
                )
                self.labelList.findItemByShape(shape).setCheckState(check_state)
                self.IDList.findItemByShape(shape).setCheckState(check_state)
        finally:
            self._noSelectionSlot = previous_selection_guard
            for obj, was_blocked in reversed(previous_signal_states):
                obj.blockSignals(was_blocked)

        if not previous_selection_guard:
            self.shapeSelectionChanged(selected_shapes)

    # Callback functions:

    def _nextTrackId(self):
        used_track_ids = set()
        for shape in self.canvas.shapes:
            for value in (shape.track_id, shape.group_id):
                if value is None:
                    continue
                value = str(value).strip()
                if value.isdigit():
                    used_track_ids.add(int(value))

        track_id = 1
        while track_id in used_track_ids:
            track_id += 1
        return str(track_id)

    def _promptForNewShapeMetadata(self):
        items = self.uniqLabelList.selectedItems()
        text_label = None
        text_id = None
        if items:
            text_label = items[0].data(Qt.UserRole)
        flags = {}
        group_id = None
        description = ""
        if self.mode not in ["NORMAL", "None"]:
            text_label = self.label_INPO
            flags = {}
            group_id = None
            description = ""
            text_id = self.ID_INPO
        elif self._config["display_label_popup"] or not text_label:
            previous_text_label = self.labelDialog.edit.text()
            text_label, flags, group_id, description = self.labelDialog.popUp(
                text_label
            )
            if group_id is not None:
                text_id = str(group_id)
            if not text_label:
                self.labelDialog.edit.setText(previous_text_label)

        if text_label and not self.validateLabel(text_label):
            self.errorMessage(
                self.tr("Invalid label"),
                self.tr("Invalid label '{}' with validation type '{}'").format(
                    text_label, self._config["validate_label"]
                ),
            )
            text_label = ""
        if not text_label:
            return None
        if not text_id:
            text_id = self._nextTrackId()
        return text_label, flags, group_id, description, text_id

    def _finishNewShape(self, shape, group_id, track_id, description):
        self.labelList.clearSelection()
        self.IDList.clearSelection()
        shape.group_id = group_id
        shape.track_id = track_id
        shape.description = description
        self.addLabel(shape)
        self.actions.undoLastPoint.setEnabled(False)
        self.actions.undo.setEnabled(True)
        self.setDirty()
        self.setEditMode()

    def newShape(self):
        """Pop-up and give focus to the label editor."""
        metadata = self._promptForNewShapeMetadata()
        if metadata is None:
            self.canvas.undoLastLine()
            self.canvas.shapesBackups.pop()
            return

        text_label, flags, group_id, description, text_id = metadata
        shape = self.canvas.setLastLabel(text_label, flags)
        self._finishNewShape(shape, group_id, text_id, description)

    def scrollRequest(self, delta, orientation):
        units = -delta * 0.1  # natural scroll
        bar = self.scrollBars[orientation]
        value = bar.value() + bar.singleStep() * units
        self.setScroll(orientation, value)

    def setScroll(self, orientation, value):
        self.scrollBars[orientation].setValue(int(value))
        self.scroll_values[orientation][self.filename] = value

    def setZoom(self, value):
        self.actions.fitWidth.setChecked(False)
        self.actions.fitWindow.setChecked(False)
        self.zoomMode = self.MANUAL_ZOOM
        self.zoomWidget.setValue(value)
        self.zoom_values[self.filename] = (self.zoomMode, value)

    def addZoom(self, increment=1.1):
        zoom_value = self.zoomWidget.value() * increment
        if increment > 1:
            zoom_value = math.ceil(zoom_value)
        else:
            zoom_value = math.floor(zoom_value)
        self.setZoom(zoom_value)

    def zoomRequest(self, delta, pos):
        canvas_width_old = self.canvas.width()
        units = 1.1
        if delta < 0:
            units = 0.9
        self.addZoom(units)

        canvas_width_new = self.canvas.width()
        if canvas_width_old != canvas_width_new:
            canvas_scale_factor = canvas_width_new / canvas_width_old

            x_shift = round(pos.x() * canvas_scale_factor) - pos.x()
            y_shift = round(pos.y() * canvas_scale_factor) - pos.y()

            self.setScroll(
                Qt.Horizontal,
                self.scrollBars[Qt.Horizontal].value() + x_shift,
            )
            self.setScroll(
                Qt.Vertical,
                self.scrollBars[Qt.Vertical].value() + y_shift,
            )

    def setFitWindow(self, value=True):
        if value:
            self.actions.fitWidth.setChecked(False)
        self.zoomMode = self.FIT_WINDOW if value else self.MANUAL_ZOOM
        self.adjustScale()

    def setFitWidth(self, value=True):
        if value:
            self.actions.fitWindow.setChecked(False)
        self.zoomMode = self.FIT_WIDTH if value else self.MANUAL_ZOOM
        self.adjustScale()

    def enableKeepPrevScale(self, enabled):
        self._config["keep_prev_scale"] = enabled
        self.actions.keepPrevScale.setChecked(enabled)

    def onNewBrightnessContrast(self, qimage):
        self.canvas.loadPixmap(QtGui.QPixmap.fromImage(qimage), clear_shapes=False)

    def brightnessContrast(self, value):
        dialog = BrightnessContrastDialog(
            utils.img_data_to_pil(self.imageData),
            self.onNewBrightnessContrast,
            parent=self,
        )
        key = self._brightnessContrastKey()
        brightness, contrast = self.brightnessContrast_values.get(key, (None, None))
        if brightness is not None:
            dialog.slider_brightness.setValue(brightness)
        if contrast is not None:
            dialog.slider_contrast.setValue(contrast)
        dialog.exec_()

        brightness = dialog.slider_brightness.value()
        contrast = dialog.slider_contrast.value()
        if key is not None:
            self.brightnessContrast_values[key] = (brightness, contrast)

    def togglePolygons(self, value):
        flag = value
        for item in self.labelList:
            if value is None:
                flag = item.checkState() == Qt.Unchecked
            item.setCheckState(Qt.Checked if flag else Qt.Unchecked)

    def hideSelectedShape(self):
        for shape in self.canvas.selectedShapes:
            for item in self.labelList:
                if item.shape() is shape:
                    item.setCheckState(Qt.Unchecked)
                    break

    def loadFile(self, filename=None):
        """Load the specified file, or the last opened file if None."""
        if not self._flushPendingAutoSave():
            return False

        if filename is None:  # image file name .jpg
            filename = self.settings.value("filename", "")
        filename = osp.abspath(str(filename))
        if not QtCore.QFile.exists(filename):
            self.errorMessage(
                self.tr("Error opening file"),
                self.tr("No such file: <b>%s</b>") % filename,
            )
            return False
        self.status(str(self.tr("Loading %s...")) % osp.basename(str(filename)))
        explicit_label_path = filename if LabelFile.is_label_file(filename) else None
        if explicit_label_path is None and filename == self.filename:
            explicit_label_path = self._explicit_label_path
        label_file = explicit_label_path or self._resolveJsonPath(
            image_path=filename, for_write=False
        )
        loaded_label_file = None
        loaded_image_data = None
        loaded_image_path = filename
        loaded_other_data = None
        actual_filename = filename
        if QtCore.QFile.exists(label_file) and LabelFile.is_label_file(
            label_file
        ):  # check if label_file exists and has correct type
            try:
                loaded_label_file = LabelFile(label_file)
            except LabelFileError as e:
                self.errorMessage(
                    self.tr("Error opening file"),
                    self.tr(
                        "<p><b>%s</b></p><p>Make sure <i>%s</i> is a valid label file."
                    )
                    % (e, label_file),
                )
                self.status(self.tr("Error reading %s") % label_file)
                return False
            loaded_image_data = loaded_label_file.imageData
            referenced_image_path = loaded_label_file.imagePath
            if not isinstance(referenced_image_path, str) or not referenced_image_path:
                self.errorMessage(
                    self.tr("Error opening file"),
                    self.tr("Label file has no valid imagePath: %s") % label_file,
                )
                return False
            if not osp.isabs(referenced_image_path):
                referenced_image_path = osp.join(
                    osp.dirname(label_file), referenced_image_path
                )
            referenced_image_path = osp.abspath(osp.normpath(referenced_image_path))
            loaded_image_path = (
                referenced_image_path if explicit_label_path else filename
            )
            loaded_other_data = loaded_label_file.otherData
            if loaded_image_data is None:
                loaded_image_data = LabelFile.load_image_file(loaded_image_path)
            if explicit_label_path and osp.isfile(referenced_image_path):
                actual_filename = referenced_image_path
        else:
            loaded_image_data = LabelFile.load_image_file(filename)
        if loaded_image_data is None:
            self.errorMessage(
                self.tr("Error opening file"),
                self.tr("Cannot read image: %s") % actual_filename,
            )
            self.status(self.tr("Error reading %s") % actual_filename)
            return False
        image = QtGui.QImage.fromData(loaded_image_data)

        if image.isNull():
            formats = [
                "*.{}".format(fmt.data().decode())
                for fmt in QtGui.QImageReader.supportedImageFormats()
            ]
            self.errorMessage(
                self.tr("Error opening file"),
                self.tr(
                    "<p>Make sure <i>{0}</i> is a valid image file.<br/>"
                    "Supported image formats: {1}</p>"
                ).format(actual_filename, ",".join(formats)),
            )
            self.status(self.tr("Error reading %s") % actual_filename)
            return False

        loaded_shapes = []
        flags = {k: False for k in self._config["flags"] or []}
        try:
            if loaded_label_file is not None:
                loaded_shapes = self._deserializeShapes(loaded_label_file.shapes)
                if not isinstance(loaded_label_file.flags, dict):
                    raise ValueError("top-level flags must be an object")
                flags.update(loaded_label_file.flags)
        except (IndexError, KeyError, TypeError, ValueError) as exc:
            self.errorMessage(
                self.tr("Error opening file"),
                self.tr("Invalid annotation data in %s: %s") % (label_file, exc),
            )
            self.status(self.tr("Error reading %s") % label_file)
            return False

        prev_shapes = list(self.canvas.shapes) if self._config["keep_prev"] else []
        self.resetState(release_ai_model=False)
        self.canvas.setEnabled(False)
        self.labelFile = loaded_label_file
        self.ir_old_shapes = (
            list(loaded_label_file.shapes) if loaded_label_file is not None else []
        )
        self.imageData = loaded_image_data
        self.imagePath = loaded_image_path
        self.otherData = loaded_other_data
        self.image = image  # image data
        self.filename = actual_filename
        self._explicit_label_path = explicit_label_path
        self.canvas.loadPixmap(QtGui.QPixmap.fromImage(image))
        if loaded_shapes:
            self.loadShapes(loaded_shapes)
        self.loadFlags(flags)
        if (
            self._config["keep_prev"] and self.noShapes()
        ):  # check noShapes() /// Shapes are annotations
            self.loadShapes(
                prev_shapes, replace=False
            )  # load annotation from prev image
            self.setDirty()
        elif self.ir_activated:
            self.setDirty()
        else:
            self.setClean()
        self.canvas.setEnabled(True)
        # set zoom values
        is_initial_load = not self.zoom_values
        if self.filename in self.zoom_values:
            self.zoomMode = self.zoom_values[self.filename][0]
            self.setZoom(self.zoom_values[self.filename][1])
        elif is_initial_load or not self._config["keep_prev_scale"]:
            self.adjustScale(initial=True)
        # set scroll values
        for orientation in self.scroll_values:
            if self.filename in self.scroll_values[orientation]:
                self.setScroll(
                    orientation, self.scroll_values[orientation][self.filename]
                )
        # set brightness contrast values
        brightness_contrast_key = self._brightnessContrastKey(self.filename)
        brightness, contrast = self.brightnessContrast_values.get(
            brightness_contrast_key, (None, None)
        )
        prev_brightness, prev_contrast = self._previousBrightnessContrastValues(
            brightness_contrast_key
        )
        if self._config["keep_prev_brightness"] and brightness is None:
            brightness = prev_brightness
        if self._config["keep_prev_contrast"] and contrast is None:
            contrast = prev_contrast
        if brightness_contrast_key is not None:
            self.brightnessContrast_values[brightness_contrast_key] = (
                brightness,
                contrast,
            )
        if brightness is not None or contrast is not None:
            dialog = BrightnessContrastDialog(
                utils.img_data_to_pil(self.imageData),
                self.onNewBrightnessContrast,
                parent=self,
            )
            if brightness is not None:
                dialog.slider_brightness.setValue(brightness)
            if contrast is not None:
                dialog.slider_contrast.setValue(contrast)
            dialog.onNewValue(None)
        self.paintCanvas()
        self.addRecentFile(self.filename)
        self.toggleActions(True)
        self.canvas.setFocus()
        # Selection is committed only after image and annotation validation succeeds.
        if self.filename in self.imageList and (
            self.fileListWidget.currentRow() != self.imageList.index(self.filename)
        ):
            self.fileListWidget.blockSignals(True)
            try:
                self.fileListWidget.setCurrentRow(self.imageList.index(self.filename))
            finally:
                self.fileListWidget.blockSignals(False)
        self.status(str(self.tr("Loaded %s")) % osp.basename(str(self.filename)))
        return True

    def resizeEvent(self, event):
        if (
            self.canvas
            and not self.image.isNull()
            and self.zoomMode != self.MANUAL_ZOOM
        ):
            self.adjustScale()
        super(MainWindow, self).resizeEvent(event)

    def paintCanvas(self):
        assert not self.image.isNull(), "cannot paint null image"
        self.canvas.scale = 0.01 * self.zoomWidget.value()
        self.canvas.adjustSize()
        self.canvas.update()

    def adjustScale(self, initial=False):
        value = self.scalers[self.FIT_WINDOW if initial else self.zoomMode]()
        value = int(100 * value)
        self.zoomWidget.setValue(value)
        self.zoom_values[self.filename] = (self.zoomMode, value)

    def scaleFitWindow(self):
        """Figure out the size of the pixmap to fit the main widget."""
        e = 2.0  # So that no scrollbars are generated.
        w1 = self.centralWidget().width() - e
        h1 = self.centralWidget().height() - e
        a1 = w1 / h1
        # Calculate a new scale value based on the pixmap's aspect ratio.
        w2 = self.canvas.pixmap.width() - 0.0
        h2 = self.canvas.pixmap.height() - 0.0
        a2 = w2 / h2
        return w1 / w2 if a2 >= a1 else h1 / h2

    def scaleFitWidth(self):
        # The epsilon does not seem to work too well here.
        w = self.centralWidget().width() - 2.0
        return w / self.canvas.pixmap.width()

    def enableSaveImageWithData(self, enabled):
        self._config["store_data"] = enabled
        self.actions.saveWithImageData.setChecked(enabled)

    def enableAutoSave(self, enabled):
        self._config["auto_save"] = bool(enabled)
        self.actions.saveAuto.setChecked(bool(enabled))
        if enabled and self.dirty and self.filename:
            self._pending_auto_save_filename = self.filename
            self._pending_auto_save_target = self._resolveJsonPath(for_write=True)
            self._save_timer.start()
        elif not enabled:
            self._save_timer.stop()
            self._pending_auto_save_filename = None
            self._pending_auto_save_target = None

    def closeEvent(self, event):
        if self._tracking_thread is not None and self._tracking_thread.isRunning():
            self._close_after_tracking = True
            self._cancelTrackingWorker()
            self.status("Canceling the active tracking operation before closing...")
            event.ignore()
            return
        if (
            self._hosted_sam2_thread is not None
            and self._hosted_sam2_thread.isRunning()
        ):
            self._close_after_hosted_request = True
            self.status("Waiting for the active hosted SAM2 request to finish...")
            event.ignore()
            return
        self._close_after_tracking = False
        self._close_after_hosted_request = False
        if not self.mayContinue():
            event.ignore()
            return
        self.settings.setValue("filename", self.filename if self.filename else "")
        self.settings.setValue("window/size", self.size())
        self.settings.setValue("window/position", self.pos())
        self.settings.setValue("window/state", self.saveState())
        self.settings.setValue("recentFiles", self.recentFiles)
        self._hosted_sam2_image_cache.clear()
        close_client = getattr(self._hosted_sam2_client, "close", None)
        if close_client is not None:
            close_client()
        self.canvas.releaseAiModel()
        event.accept()
        # ask the use for where to save the labels
        # self.settings.setValue('window/geometry', self.saveGeometry())

    def dragEnterEvent(self, event):
        extensions = [
            ".%s" % fmt.data().decode().lower()
            for fmt in QtGui.QImageReader.supportedImageFormats()
        ]
        if event.mimeData().hasUrls():
            items = [i.toLocalFile() for i in event.mimeData().urls()]
            if any([i.lower().endswith(tuple(extensions)) for i in items]):
                event.accept()
        else:
            event.ignore()

    def dropEvent(self, event):
        if not self.mayContinue():
            event.ignore()
            return
        items = [i.toLocalFile() for i in event.mimeData().urls()]
        self.importDroppedImageFiles(items)

    # User Dialogs #

    def loadRecent(self, filename):
        if self.mayContinue():
            self.loadFile(filename)

    def openPrevImg(self, _value=False):
        keep_prev = self._config["keep_prev"]
        if QtWidgets.QApplication.keyboardModifiers() == (
            Qt.ControlModifier | Qt.ShiftModifier
        ):
            self._config["keep_prev"] = True
        try:
            return self._openPrevImg()
        finally:
            self._config["keep_prev"] = keep_prev

    def _openPrevImg(self):
        if not self.mayContinue():
            return

        if len(self.imageList) <= 0:
            return

        self._flushPendingAutoSave()

        if self.mode == "NORMAL" or self.mode == "None":
            self.ir_activated = False
            if self.filename is None:
                return

            currIndex = self.imageList.index(self.filename)
            if currIndex - 1 >= 0:
                filename = self.imageList[currIndex - 1]
                if filename:
                    self.loadFile(filename)
        else:
            currIndex = self.INTERPOLATION_list.index(self.filename)
            if currIndex - 1 >= 0:
                filename = self.INTERPOLATION_list[currIndex - 1]
                if filename:
                    self.loadFile(filename)

    def openNextImg(self, _value=False, load=True):
        keep_prev = self._config["keep_prev"]
        if QtWidgets.QApplication.keyboardModifiers() == (
            Qt.ControlModifier | Qt.ShiftModifier
        ):
            self._config["keep_prev"] = True
        try:
            return self._openNextImg(load=load)
        finally:
            self._config["keep_prev"] = keep_prev

    def _openNextImg(self, load=True):
        # Refinement deltas are valid for one forward frame load only.
        self.ir_activated = False
        if not self.mayContinue():
            return

        if len(self.imageList) <= 0:
            return

        self._flushPendingAutoSave()

        if self.mode == "NORMAL" or self.mode == "None":
            if self.filename is not None:
                if self.filename not in self.imageList:
                    return
                if self.imageList.index(self.filename) + 1 >= len(self.imageList):
                    return
            if (
                self.interpolationrefine_list.checkBox.isChecked()
                and self.ir_name != "None"
                and self.ir_id != "None"
            ):
                old_found = False
                modified_found = False
                # original
                for item in self.ir_old_shapes:
                    if item["label"] == self.ir_name and str(
                        shape_track_id(item)
                    ) == str(self.ir_id):
                        self.ir_old_shape = item["points"]
                        old_found = True
                # modified
                for item in self.labelList:
                    if item.shape().label == self.ir_name and str(
                        item.shape().track_id
                    ) == str(self.ir_id):
                        modified_found = True
                        self.ir_mod_shape = [
                            [p.x(), p.y()] for p in item.shape().points
                        ]
                if not (old_found and modified_found):
                    self.ir_old_shape = "None"
                    self.ir_mod_shape = "None"
                self.ir_activated = old_found and modified_found
            else:
                self.ir_activated = False

            filename = None
            if self.filename is None:
                filename = self.imageList[0]
            else:
                currIndex = self.imageList.index(self.filename)
                filename = self.imageList[currIndex + 1]
            if filename and load:
                try:
                    self.loadFile(filename)
                finally:
                    self.ir_activated = False
            else:
                self.ir_activated = False
        else:
            if (
                not self.INTERPOLATION_list
                or self.filename not in self.INTERPOLATION_list
            ):
                return
            currIndex = self.INTERPOLATION_list.index(self.filename)
            if currIndex + 1 >= len(self.INTERPOLATION_list):
                return
            filename = self.INTERPOLATION_list[currIndex + 1]
            if filename and load:
                self.loadFile(filename)

    def openFile(self, _value=False):
        if not self.mayContinue():
            return
        path = osp.dirname(str(self.filename)) if self.filename else "."
        formats = [
            "*.{}".format(fmt.data().decode())
            for fmt in QtGui.QImageReader.supportedImageFormats()
        ]
        filters = self.tr("Image & Label files (%s)") % " ".join(
            formats + ["*%s" % LabelFile.suffix]
        )
        fileDialog = FileDialogPreview(self)
        fileDialog.setFileMode(FileDialogPreview.ExistingFile)
        fileDialog.setNameFilter(filters)
        fileDialog.setWindowTitle(
            self.tr("%s - Choose Image or Label file") % __appname__,
        )
        fileDialog.setWindowFilePath(path)
        fileDialog.setViewMode(FileDialogPreview.Detail)
        if fileDialog.exec_():
            fileName = fileDialog.selectedFiles()[0]
            if fileName:
                self.loadFile(fileName)

    def changeOutputDirDialog(self, _value=False):
        if not self.mayContinue():
            return
        default_output_dir = self.output_dir
        if default_output_dir is None and self.filename:
            default_output_dir = osp.dirname(self.filename)
        if default_output_dir is None:
            default_output_dir = self.currentPath()

        output_dir = QtWidgets.QFileDialog.getExistingDirectory(
            self,
            self.tr("%s - Save/Load Annotations in Directory") % __appname__,
            default_output_dir,
            QtWidgets.QFileDialog.ShowDirsOnly
            | QtWidgets.QFileDialog.DontResolveSymlinks,
        )
        output_dir = str(output_dir)

        if not output_dir:
            return

        previous_output_dir = self.output_dir
        previous_explicit_label_path = self._explicit_label_path
        current_filename = self.filename
        self.output_dir = output_dir
        self._explicit_label_path = None
        if self.lastOpenDir and not self.importDirImages(
            self.lastOpenDir, load=False, check_continue=False
        ):
            self.output_dir = previous_output_dir
            self._explicit_label_path = previous_explicit_label_path
            self.status(
                self.tr("Could not scan images for the selected annotation directory.")
            )
            return
        keep_prev = self._config["keep_prev"]
        ir_activated = self.ir_activated
        try:
            self._config["keep_prev"] = False
            self.ir_activated = False
            loaded = not current_filename or self.loadFile(current_filename)
        finally:
            self._config["keep_prev"] = keep_prev
            self.ir_activated = ir_activated
        if not loaded:
            self.output_dir = previous_output_dir
            self._explicit_label_path = previous_explicit_label_path
            if self.lastOpenDir:
                self.importDirImages(self.lastOpenDir, load=False, check_continue=False)
            self._restoreCurrentFileSelection()
            self.status(
                self.tr("Could not load annotations from the selected directory.")
            )
            return

        self.statusBar().showMessage(
            self.tr("%s . Annotations will be saved/loaded in %s")
            % ("Change Annotations Dir", self.output_dir)
        )
        self.statusBar().show()

    def saveFile(self, _value=False):
        assert not self.image.isNull(), "cannot save empty image"
        if self.output_file:
            saved = self._saveFile(self.output_file)
            if saved:
                self.close()
            return saved
        return self._saveFile(self._resolveJsonPath(for_write=True))

    def saveFileAs(self, _value=False):
        assert not self.image.isNull(), "cannot save empty image"
        return self._saveFile(self.saveFileDialog(), remember_explicit_path=True)

    def saveFileDialog(self):
        caption = self.tr("%s - Choose File") % __appname__
        filters = self.tr("Label files (*%s)") % LabelFile.suffix
        if self.output_dir:
            dlg = QtWidgets.QFileDialog(self, caption, self.output_dir, filters)
        else:
            dlg = QtWidgets.QFileDialog(self, caption, self.currentPath(), filters)
        dlg.setDefaultSuffix(LabelFile.suffix[1:])
        dlg.setAcceptMode(QtWidgets.QFileDialog.AcceptSave)
        dlg.setOption(QtWidgets.QFileDialog.DontConfirmOverwrite, False)
        dlg.setOption(QtWidgets.QFileDialog.DontUseNativeDialog, False)
        if self.output_dir:
            default_labelfile_name = self._resolveJsonPath(for_write=True)
        else:
            basename = osp.basename(osp.splitext(self.filename)[0])
            default_labelfile_name = osp.join(
                self.currentPath(), basename + LabelFile.suffix
            )
        filename = dlg.getSaveFileName(
            self,
            self.tr("Choose File"),
            default_labelfile_name,
            self.tr("Label files (*%s)") % LabelFile.suffix,
        )
        if isinstance(filename, tuple):
            filename, _ = filename
        return filename

    def _saveFile(self, filename, remember_explicit_path=False):
        if filename and self.saveLabels(filename):
            if remember_explicit_path:
                self._explicit_label_path = osp.abspath(filename)
            self.addRecentFile(filename)
            self.setClean()
            return True
        return False

    def closeFile(self, _value=False):
        if not self.mayContinue():
            return
        self.resetState()
        self.setClean()
        self.toggleActions(False)
        self.canvas.setEnabled(False)
        self.actions.saveAs.setEnabled(False)

    def getLabelFile(self):
        return self._resolveJsonPath(for_write=False)

    def deleteFile(self):
        mb = QtWidgets.QMessageBox
        msg = self.tr(
            "You are about to permanently delete this label file, proceed anyway?"
        )
        answer = mb.warning(self, self.tr("Attention"), msg, mb.Yes | mb.No)
        if answer != mb.Yes:
            return

        label_file = self.getLabelFile()
        if osp.exists(label_file):
            try:
                os.remove(label_file)
            except OSError as exc:
                self.errorMessage(self.tr("Error deleting label data"), str(exc))
                return
            logger.info("Label file is removed: {}".format(label_file))

            item = self.fileListWidget.currentItem()
            if item is not None:
                item.setCheckState(Qt.Unchecked)

            self._save_timer.stop()
            self._pending_auto_save_filename = None
            self._pending_auto_save_target = None
            self.resetState()
            self.setClean()
            self.toggleActions(False)
            self.canvas.setEnabled(False)

    # Message Dialogs. #
    def hasLabels(self):
        if self.noShapes():
            self.errorMessage(
                "No objects labeled",
                "You must label at least one object to save the file.",
            )
            return False
        return True

    def hasLabelFile(self):
        if self.filename is None:
            return False

        label_file = self.getLabelFile()
        return osp.exists(label_file)

    def mayContinue(self):
        if getattr(self, "filename", None) is None:
            return True
        if getattr(self, "_save_timer", None) is not None:
            if self._save_timer.isActive() and self._flushPendingAutoSave():
                if not self.dirty:
                    return True
        if not self.dirty:
            return True
        mb = QtWidgets.QMessageBox
        msg = self.tr('Save annotations to "{}" before closing?').format(self.filename)
        answer = mb.question(
            self,
            self.tr("Save annotations?"),
            msg,
            mb.Save | mb.Discard | mb.Cancel,
            mb.Save,
        )
        if answer == mb.Discard:
            self._save_timer.stop()
            self._pending_auto_save_filename = None
            self._pending_auto_save_target = None
            return True
        elif answer == mb.Save:
            return bool(self.saveFile())
        else:  # answer == mb.Cancel
            return False

    def errorMessage(self, title, message):
        return QtWidgets.QMessageBox.critical(
            self, title, "<p><b>%s</b></p>%s" % (title, message)
        )

    def informationMessage(self, title, message):
        return QtWidgets.QMessageBox.information(
            self, title, "<p><b>%s</b></p>%s" % (title, message)
        )

    def currentPath(self):
        return osp.dirname(str(self.filename)) if self.filename else "."

    def toggleKeepPrevMode(self):
        self._config["keep_prev"] = not self._config["keep_prev"]

    def removeSelectedPoint(self):
        self.canvas.removeSelectedPoint()
        self.canvas.update()
        if not self.canvas.hShape.points:
            self.canvas.deleteShape(self.canvas.hShape)
            self.remLabels([self.canvas.hShape])
            if self.noShapes():
                for action in self.actions.onShapesPresent:
                    action.setEnabled(False)
        self.setDirty()

    def deleteSelectedShape(self):
        # yes, no = QtWidgets.QMessageBox.Yes, QtWidgets.QMessageBox.No
        # msg = self.tr(
        #     "You are about to permanently delete {} polygons, " "proceed anyway?"
        # ).format(len(self.canvas.selectedShapes))
        # if yes == QtWidgets.QMessageBox.warning(
        #     self, self.tr("Attention"), msg, yes | no, yes
        # ):
        #     self.remLabels(self.canvas.deleteSelected())
        #     self.setDirty()
        #     if self.noShapes():
        #         for action in self.actions.onShapesPresent:
        #             action.setEnabled(False)

        self.remLabels(self.canvas.deleteSelected())
        self.setDirty()
        if self.noShapes():
            for action in self.actions.onShapesPresent:
                action.setEnabled(False)

    def copyShape(self):
        self.canvas.endMove(copy=True)
        for shape in self.canvas.selectedShapes:
            self.addLabel(shape)
        self.labelList.clearSelection()
        self.IDList.clearSelection()
        self.setDirty()

    def moveShape(self):
        self.canvas.endMove(copy=False)
        self.setDirty()

    def openDirDialog(self, _value=False, dirpath=None):
        if not self.mayContinue():
            return

        defaultOpenDirPath = dirpath if dirpath else "."
        if self.lastOpenDir and osp.exists(self.lastOpenDir):
            defaultOpenDirPath = self.lastOpenDir
        else:
            defaultOpenDirPath = osp.dirname(self.filename) if self.filename else "."

        targetDirPath = str(
            QtWidgets.QFileDialog.getExistingDirectory(
                self,
                self.tr("%s - Open Directory") % __appname__,
                defaultOpenDirPath,
                QtWidgets.QFileDialog.ShowDirsOnly
                | QtWidgets.QFileDialog.DontResolveSymlinks
                | QtWidgets.QFileDialog.DontUseNativeDialog,
            )
        )
        self.importDirImages(targetDirPath, check_continue=False)

    @property
    def imageList(self):
        if self._imageListCache is None:
            lst = []
            for i in range(self.fileListWidget.count()):
                item = self.fileListWidget.item(i)
                lst.append(item.text())
            self._imageListCache = lst
        return self._imageListCache

    def importDroppedImageFiles(self, imageFiles):
        extensions = [
            ".%s" % fmt.data().decode().lower()
            for fmt in QtGui.QImageReader.supportedImageFormats()
        ]

        imageFiles = [
            osp.abspath(file)
            for file in imageFiles
            if file.lower().endswith(tuple(extensions))
        ]
        if not self.lastOpenDir and imageFiles:
            try:
                self.lastOpenDir = osp.commonpath(
                    [osp.dirname(file) for file in imageFiles]
                )
            except ValueError:
                self.lastOpenDir = None
        all_image_files = list(self.imageList) + imageFiles
        added_files = []
        for file in imageFiles:
            if file in self.imageList:
                continue
            label_file = self._resolveJsonPath(
                image_path=file,
                for_write=False,
                image_paths=all_image_files,
            )
            item = QtWidgets.QListWidgetItem(file)
            item.setFlags(
                Qt.ItemIsEnabled | Qt.ItemIsSelectable | Qt.ItemIsUserCheckable
            )
            if QtCore.QFile.exists(label_file) and LabelFile.is_label_file(label_file):
                item.setCheckState(Qt.Checked)
            else:
                item.setCheckState(Qt.Unchecked)
            self.fileListWidget.addItem(item)
            self._imageListCache = None
            added_files.append(file)

        if len(self.imageList) > 1:
            self.actions.openNextImg.setEnabled(True)
            self.actions.openPrevImg.setEnabled(True)

        if added_files:
            self.loadFile(added_files[0])

    def importDirImages(self, dirpath, pattern=None, load=True, check_continue=True):
        if not dirpath or (check_continue and not self.mayContinue()):
            return False

        current_filename = getattr(self, "filename", None)
        previous_root = self.lastOpenDir
        previous_items = [
            (
                self.fileListWidget.item(index).text(),
                self.fileListWidget.item(index).checkState(),
                self.fileListWidget.item(index).isHidden(),
            )
            for index in range(self.fileListWidget.count())
        ]
        previous_row = self.fileListWidget.currentRow()
        previous_next_enabled = self.actions.openNextImg.isEnabled()
        previous_prev_enabled = self.actions.openPrevImg.isEnabled()
        new_root = osp.abspath(dirpath)
        filenames = self.scanAllImages(dirpath)
        if pattern:
            try:
                filenames = [f for f in filenames if re.search(pattern, f)]
            except re.error:
                pass

        def populate(items, root, preserve_states=False):
            self.fileListWidget.blockSignals(True)
            try:
                self.fileListWidget.clear()
                self.lastOpenDir = root
                paths = [item[0] if preserve_states else item for item in items]
                for index, filename in enumerate(paths):
                    item = QtWidgets.QListWidgetItem(filename)
                    item.setFlags(
                        Qt.ItemIsEnabled | Qt.ItemIsSelectable | Qt.ItemIsUserCheckable
                    )
                    if preserve_states:
                        item.setCheckState(items[index][1])
                        item.setHidden(items[index][2])
                    else:
                        label_file = self._resolveJsonPath(
                            image_path=filename,
                            for_write=False,
                            image_paths=paths,
                        )
                        item.setCheckState(
                            Qt.Checked
                            if QtCore.QFile.exists(label_file)
                            and LabelFile.is_label_file(label_file)
                            else Qt.Unchecked
                        )
                    self.fileListWidget.addItem(item)
                self._imageListCache = list(paths)
            finally:
                self.fileListWidget.blockSignals(False)

        if not filenames:
            populate([], new_root)
            self.resetState()
            self.setClean()
            self.toggleActions(False)
            self.canvas.setEnabled(False)
            self.actions.saveAs.setEnabled(False)
            self.actions.openNextImg.setEnabled(False)
            self.actions.openPrevImg.setEnabled(False)
            self.status(self.tr("No images found in %s") % new_root)
            return True

        populate(filenames, new_root)
        if load and not self.loadFile(filenames[0]):
            populate(previous_items, previous_root, preserve_states=True)
            self.filename = current_filename
            if 0 <= previous_row < self.fileListWidget.count():
                self.fileListWidget.blockSignals(True)
                try:
                    self.fileListWidget.setCurrentRow(previous_row)
                finally:
                    self.fileListWidget.blockSignals(False)
            self.actions.openNextImg.setEnabled(previous_next_enabled)
            self.actions.openPrevImg.setEnabled(previous_prev_enabled)
            return False
        if not load:
            self.filename = current_filename
        navigation_enabled = len(filenames) > 1
        self.actions.openNextImg.setEnabled(navigation_enabled)
        self.actions.openPrevImg.setEnabled(navigation_enabled)
        return True

    def scanAllImages(self, folderPath):
        extensions = [
            ".%s" % fmt.data().decode().lower()
            for fmt in QtGui.QImageReader.supportedImageFormats()
        ]

        images = []
        for root, dirs, files in os.walk(folderPath):
            for file in files:
                if file.lower().endswith(tuple(extensions)):
                    relativePath = os.path.normpath(osp.join(root, file))
                    images.append(relativePath)
        images = natsort.os_sorted(images)
        return images
