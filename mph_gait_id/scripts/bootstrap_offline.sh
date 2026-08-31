#!/usr/bin/env bash
set -euo pipefail

PACKAGE_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
PROJECT_ROOT=$(cd -- "${PACKAGE_ROOT}/.." && pwd)
VENV="${PROJECT_ROOT}/.venv"

python3 -m venv "${VENV}"
"${VENV}/bin/python" -m pip install --upgrade pip
"${VENV}/bin/python" -m pip install -r "${PROJECT_ROOT}/requirements.txt"
"${VENV}/bin/python" "${PACKAGE_ROOT}/scripts/download_assets.py" --profile offline-demo
"${VENV}/bin/python" "${PACKAGE_ROOT}/scripts/doctor.py"

printf 'Environment ready. Start with:\n  %s -m mph_gait_id.ui_app\n'   "${VENV}/bin/python"
