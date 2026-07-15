import os
import re

from setuptools import find_packages
from setuptools import setup


def get_version():
    filename = "labelme/__init__.py"
    with open(filename) as f:
        match = re.search(r"""^__version__ = ['"]([^'"]*)['"]""", f.read(), re.M)
    if not match:
        raise RuntimeError("{} doesn't contain __version__".format(filename))
    version = match.groups()[0]
    return version


def get_install_requires():
    install_requires = [
        "gdown",
        "imgviz>=1.7.5",
        "natsort>=7.1.0",
        "numpy",
        "onnxruntime>=1.14.1,!=1.16.0",
        "opencv-contrib-python>=4.5",
        "Pillow>=2.8",
        "PyYAML",
        "qtpy!=1.11.2",
        "requests",
        "scikit-image",
        "termcolor",
        "filterpy",
        "scipy",
        "scikit-learn",
        "ultralytics>=8.3.0",
    ]

    # Find python binding for qt with priority:
    # PyQt5 -> PySide2
    # and PyQt5 is automatically installed on Python3.
    QT_BINDING = None

    try:
        import PyQt5  # NOQA

        QT_BINDING = "pyqt5"
    except ImportError:
        pass

    if QT_BINDING is None:
        try:
            import PySide2  # NOQA

            QT_BINDING = "pyside2"
        except ImportError:
            pass

    if QT_BINDING is None:
        # PyQt5 can be installed via pip for Python3
        # 5.15.3, 5.15.4 won't work with PyInstaller
        install_requires.append("PyQt5!=5.15.3,!=5.15.4")
        QT_BINDING = "pyqt5"

    del QT_BINDING

    if os.name == "nt":  # Windows
        install_requires.append("colorama")

    return install_requires


def get_long_description():
    with open("README.md") as f:
        long_description = f.read()
    try:
        # when this package is being released
        import github2pypi

        return github2pypi.replace_url(
            slug="wkentaro/labelme", content=long_description, branch="main"
        )
    except ImportError:
        # when this package is being installed
        return long_description


def main():
    version = get_version()

    setup(
        name="labelme",
        version=version,
        packages=find_packages(),
        description="Image Polygonal Annotation with Python",
        long_description=get_long_description(),
        long_description_content_type="text/markdown",
        author="Kentaro Wada",
        author_email="www.kentaro.wada@gmail.com",
        url="https://github.com/wkentaro/labelme",
        install_requires=get_install_requires(),
        python_requires=">=3.9",
        license="GPLv3",
        keywords="Image Annotation, Machine Learning",
        classifiers=[
            "Development Status :: 5 - Production/Stable",
            "Intended Audience :: Developers",
            "Intended Audience :: Science/Research",
            "Natural Language :: English",
            "Operating System :: OS Independent",
            "Programming Language :: Python",
            "Programming Language :: Python :: 3.9",
            "Programming Language :: Python :: 3 :: Only",
        ],
        package_data={"labelme": ["icons/*", "config/*.yaml", "translate/*"]},
        exclude_package_data={"labelme": ["icons/*.pt"]},
        entry_points={
            "console_scripts": [
                "labelme=labelme.__main__:main",
                "labelme_draw_json=labelme.cli.draw_json:main",
                "labelme_draw_label_png=labelme.cli.draw_label_png:main",
                "labelme_json_to_dataset=labelme.cli.json_to_dataset:main",
                "labelme_export_json=labelme.cli.export_json:main",
                "labelme_on_docker=labelme.cli.on_docker:main",
            ],
        },
    )


if __name__ == "__main__":
    main()
