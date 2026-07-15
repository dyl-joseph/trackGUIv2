import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ["LABELME_CONFIG_FILE"] = os.devnull

import pytest
from qtpy import QtCore


@pytest.fixture(scope="session", autouse=True)
def isolate_qsettings(tmp_path_factory):
    settings_dir = tmp_path_factory.mktemp("qsettings")
    QtCore.QSettings.setDefaultFormat(QtCore.QSettings.IniFormat)
    QtCore.QSettings.setPath(
        QtCore.QSettings.IniFormat,
        QtCore.QSettings.UserScope,
        str(settings_dir),
    )
    yield
    QtCore.QSettings().clear()
