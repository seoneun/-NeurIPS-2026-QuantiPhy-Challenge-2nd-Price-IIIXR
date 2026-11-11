#!/bin/bash
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${QP_ROOT:-$HERE/..}" && pwd)"
E="${1:-$ROOT/envs/quantiphy}"
CON=$HERE/env_constraints.txt
PYBASE=${PYTHON_BASE:-python3.10}
$PYBASE -c "import sys; assert sys.version_info[:2] == (3, 10), sys.version" || { echo "needs python 3.10 (PYTHON_BASE=...)"; exit 1; }
[ -x "$E/bin/python" ] || $PYBASE -m venv "$E"
PY=$E/bin/python
$PY -m pip install --progress-bar off -c "$CON" pip setuptools wheel
$PY -m pip install --progress-bar off -c "$CON" vllm pandas pyarrow opencv-python-headless av qwen-vl-utils
$PY -m pip install --progress-bar off --no-deps --only-binary=:all: scipy==1.15.3 timm==1.0.30
echo "activate: source $E/bin/activate"
