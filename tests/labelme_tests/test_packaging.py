import re
from pathlib import Path


def test_pyinstaller_icon_paths_exist():
    repository = Path(__file__).parents[2]
    spec_text = (repository / "labelme.spec").read_text(encoding="utf-8")
    icon_paths = re.findall(r"icon=['\"]([^'\"]+)", spec_text)

    assert icon_paths
    assert all((repository / icon_path).is_file() for icon_path in icon_paths)
