#!/usr/bin/env bash
set -euo pipefail

PACKAGE_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
PROJECT_ROOT=$(cd -- "${PACKAGE_ROOT}/.." && pwd)
cd "${PROJECT_ROOT}"

if [[ -n "${PYTHON_CMD:-}" ]]; then
  read -r -a PYTHON_ARR <<< "${PYTHON_CMD}"
  exec "${PYTHON_ARR[@]}" -m mph_gait_id.ui_app "$@"
fi

if [[ -x "${PROJECT_ROOT}/.venv/bin/python" ]]; then
  exec "${PROJECT_ROOT}/.venv/bin/python" -m mph_gait_id.ui_app "$@"
fi

CONDA_ENV=${CONDA_ENV:-pytorch}
exec conda run --no-capture-output -n "${CONDA_ENV}"   python -m mph_gait_id.ui_app "$@"
