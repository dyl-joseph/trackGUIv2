<h4 align="center">
  TrackMe (v2)
</h4>

<h4 align="center">
  Simple Tracking Annotation Tool based on LabelMe
</h4>

## Description
This tool is built to integrate tracking visualization and annotation capabilities into LabelMe. <br>
Please feel free to use if your project/work includes visualization of multi-object tracking and editing of object information. <br>
TrackMe annotation format is compatible with LabelMe annotation format (.json) without conversion.

<img src="examples/trackgui/TrackMe_overview.png" width="100%"/> <br>
<i>TrackMe saves and displays the tracking information of multiple objects on the right. It generates unique colors for different combinations of object label and ID  </i>

## Features
- Add/remove tracking ID.
- Associate boxes (assign IDs) for existing non-ID detection boxes in the video folder (SORT).
- Interpolate boxes in long video range in case no pre-defined detection boxes.
- Modify/Delete boxes of same info throughout a list of continuous frames.
- Display homogeneous color for same object info (for the sake of multi-view object tracking).
- **Track ID rendered on canvas** next to each bounding box for quick visual identification.
- **Backward-compatible with standard LabelMe annotations** — automatically uses `group_id` as `track_id` when `track_id` is missing from JSON files.
- **Track Forward (CSRT)** — Select a bounding box and press **T** to automatically propagate it across subsequent frames using OpenCV CSRT tracker.
- **Track modification modes** — Remove Box, Swap ID, and Swap Label modes for bulk editing annotations across frame ranges.

## Changes from Original TrackGUI

### Bug Fixes

1. **Config crash on empty `~/.labelmerc`** — `yaml.safe_load()` returns `None` when the config file is empty or contains only comments. The config loader (`labelme/config/__init__.py`) now guards against this and falls back to default configuration instead of crashing with `AttributeError: 'NoneType' object has no attribute 'items'`.

2. **Image path resolution for annotation loading** — When loading a JSON annotation file, the `imagePath` field (e.g., `video0007/frame000000.jpg`) was incorrectly tested as a path relative to the current working directory instead of relative to the JSON file's own directory (`labelme/label_file.py`). This caused images to fail to load when the working directory didn't match the JSON file's location.

3. **Annotations not appearing when JSON files are in a parent directory** — When annotation `.json` files live in the opened directory but the images are in a subdirectory (a common layout for video annotation), the app only looked for JSON files alongside each image. Now it also checks the opened directory (`lastOpenDir`) as a fallback (`labelme/app.py`).

4. **Cannot delete shapes after drawing** — After drawing a polygon or rectangle and labeling it, the canvas stayed in CREATE mode. This meant shapes could not be selected or deleted without manually clicking the "Edit" button first. Now the app automatically switches to edit mode after finishing a shape (`labelme/app.py`).

5. **macOS file dialog freeze on cloud-synced directories** — The native macOS file picker would freeze/hang when opening directories on OneDrive or other cloud-synced paths. Switched to Qt's built-in file dialog (`QFileDialog.DontUseNativeDialog`) which handles these paths reliably (`labelme/app.py`).

6. **Double dialog prompt when drawing rectangles** — The label and track ID dialogs each appeared twice when creating a new rectangle. Caused by the autocompleter's `activated` signal being connected to `popUp()`, which reopened the modal dialog. Fixed by connecting to `edit.setText()` instead (`labelme/widgets/label_dialog.py`, `labelme/widgets/id_dialog.py`).

7. **Deletion dialog only deleted, never swapped** — The track modification dialog didn't expose its mode selection. Rewrote the dialog and the `DELETION` method to properly support all three modes: Remove Box, Swap ID, and Swap Label (`labelme/widgets/deletetrack_dialog.py`, `labelme/app.py`).

8. **Save path resolution inconsistencies** — `saveFile()` and `setDirty()` used different logic to determine where to write JSON files, sometimes falling through to a save dialog or writing to the wrong directory. Consolidated into a single `_resolveJsonPath()` helper that prefers `lastOpenDir` (`labelme/app.py`).

9. **Deletions not persisting** — `deleteSelectedShape()` called `setDirty()` but not `saveFile()`, so deleted shapes reappeared on reload. Added explicit save after deletion (`labelme/app.py`).

10. **Crash on non-numeric track_id** — `_get_rgb_by_label()` called `int()` on track IDs like `"defect"`, causing a `ValueError`. Added try/except fallback (`labelme/app.py`).

### Enhancements

11. **Track Forward (CSRT)** — Select a rectangle bounding box and press **T** (or Track > Track Forward) to propagate it across subsequent frames using OpenCV's CSRT tracker. The tracker writes annotations with the same label and track_id to each frame's JSON file, stopping automatically if tracking confidence drops (`labelme/app.py`).

12. **Track ID displayed on bounding boxes** — Each bounding box now renders its `track_id` (or `group_id` as fallback) in white text above the top-left corner of the shape on the canvas (`labelme/shape.py`). This makes it easy to visually identify which track each annotation belongs to without having to select it.

13. **Backward compatibility with standard LabelMe annotations** — When loading JSON files that have `group_id` but no `track_id` (i.e., annotations created with standard LabelMe rather than TrackGUI), the loader now automatically falls back to using `group_id` as the `track_id` (`labelme/label_file.py`). This means pre-existing LabelMe annotations work seamlessly in TrackGUI without manual conversion.

### Code Quality

14. **Ruff formatting applied** — The entire codebase has been reformatted with `ruff format` (line length 88, double quotes, 4-space indent, isort-sorted imports) for consistent code style (`labelme/app.py` and other files).

## Installation

### Option A: From environment file (recommended)

```bash
conda env create -f environment.yml
conda activate trackGUI
pip install -e .
```

### Option B: Fresh install

```bash
conda create --name=trackGUI python=3.8
conda activate trackGUI
pip install -e .
```

## Usage

### Starting the application

```bash
conda activate trackGUI
labelme
```

### Opening a directory of images

1. Click **Open Dir** (or use the keyboard shortcut) to open a directory.
2. If your images are in a subdirectory (e.g., `my_project/video0007/`) and your JSON annotations are in the parent directory (`my_project/`), open the **parent directory**. The app will recursively find images in subdirectories and match them with JSON files in the opened directory.

### Drawing and editing annotations

1. Select a drawing tool from the toolbar (e.g., **Create Rectangle**, **Create Polygon**).
2. Click on the image to draw your shape. For rectangles, click the top-left corner and then the bottom-right corner.
3. After completing the shape, a dialog will prompt you for a **label** (object class) and a **track ID** (unique identifier for tracking the object across frames).
4. The app automatically switches to **Edit mode** after drawing, so you can immediately select, move, resize, or delete the shape you just drew.
5. To draw another shape, select the drawing tool again from the toolbar.

### Selecting and deleting annotations

1. In **Edit mode**, click on a shape to select it (it will be highlighted).
2. Hold **Ctrl** and click to select multiple shapes.
3. Press **Delete** (or use the menu: Edit > Delete Polygons) to remove selected shapes.
4. The track ID number is displayed in white text above each bounding box for easy identification.

### Keyboard shortcuts

- **D** — Next image
- **A** — Previous image
- **Ctrl+S** — Save annotations
- **Shift+D** — Delete selected polygons
- **Ctrl+Z** — Undo
- **E** — Edit mode (select/move shapes)
- **H** — Hide selected shape
- **T** — Toggle all polygons visibility
- **Ctrl+T** — Track Forward (propagate selected bbox across frames)

### Tracking modes

- **Track from scratch**: Tracks from the first frame to the end frame with automatic ID assignment.
- **Track from Current Frame w/ Annotation**: Tracks from the current frame using the modified or manually assigned ID.
- **Track from Current Frame w/o Annotation**: Tracks from the current frame with automatic ID assignment.
- **Track Forward (CSRT)**: Select a single rectangle, press **T**, enter the end frame. The OpenCV CSRT tracker propagates the bbox forward, writing annotations with the same label and track_id to each frame's JSON file. Stops automatically if tracking fails.

### Annotation format

Annotations are saved as `.json` files (one per image) in LabelMe format. Each shape includes:
- `label` — object class name
- `track_id` — unique tracking identifier across frames
- `points` — bounding box or polygon coordinates
- `shape_type` — "rectangle", "polygon", etc.
- `group_id` — group identifier (used as fallback for `track_id` if missing)

## Note
All frames must have labeled boxes for interpolation and tracking features to work correctly.

## Original Paper Citation
If you find our work helpful, please consider citing our paper:
```
@article{phan2024trackme,
  title={TrackMe: A Simple and Effective Multiple Object Tracking Annotation Tool},
  author={Phan, Thinh and Phillips, Isaac and Lockett, Andrew and Kidd, Michael T and Le, Ngan},
  journal={arXiv preprint arXiv:2410.15518},
  year={2024}
}
```
