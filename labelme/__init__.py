# flake8: noqa

import logging
import os
import os.path as osp
import sys
import tempfile

from qtpy import QT_VERSION

if "MPLCONFIGDIR" not in os.environ:
    mpl_config_dir = osp.join(tempfile.gettempdir(), "labelme-matplotlib")
    os.makedirs(mpl_config_dir, exist_ok=True)
    os.environ["MPLCONFIGDIR"] = mpl_config_dir


__appname__ = "TrackMe"

# Semantic Versioning 2.0.0: https://semver.org/
# 1. MAJOR version when you make incompatible API changes;
# 2. MINOR version when you add functionality in a backwards-compatible manner;
# 3. PATCH version when you make backwards-compatible bug fixes.
# e.g., 1.0.0a0, 1.0.0a1, 1.0.0b0, 1.0.0rc0, 1.0.0, 1.0.0.post0
__version__ = "5.4.1"

QT4 = QT_VERSION[0] == "4"
QT5 = QT_VERSION[0] == "5"
del QT_VERSION

PY2 = sys.version[0] == "2"
PY3 = sys.version[0] == "3"
del sys

from labelme.label_file import LabelFile
from labelme import testing
from labelme import utils
