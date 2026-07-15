#!/usr/bin/env bash

set -euo pipefail

python_bin="${PYTHON:-python3}"

export MPLBACKEND="${MPLBACKEND:-agg}"
export QT_QPA_PLATFORM="${QT_QPA_PLATFORM:-offscreen}"

"${python_bin}" -m ruff format --check .
"${python_bin}" -m ruff check .
"${python_bin}" -m compileall -q labelme server tests test_interpolation.py conftest.py
"${python_bin}" -m pytest
"${python_bin}" -m pip check
"${python_bin}" setup.py check
