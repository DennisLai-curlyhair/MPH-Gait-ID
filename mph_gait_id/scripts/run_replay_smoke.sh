#!/usr/bin/env bash
set -euo pipefail

SYSTEM_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORKSPACE_ROOT="$(cd "${SYSTEM_ROOT}/.." && pwd)"
cd "${WORKSPACE_ROOT}"

conda run -n "${CONDA_ENV:-pytorch}" python \
  mph_gait_id/scripts/replay_smoke.py "$@"
