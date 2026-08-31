#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT}"

CONDA_ENV="${CONDA_ENV:-pytorch}"
exec conda run --no-capture-output -n "${CONDA_ENV}" \
  python mph_gait_id/scripts/replay_enrollment_smoke.py "$@"
