#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT}"

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 /path/to/replay_sequence" >&2
  exit 2
fi

REPLAY="$1"
CONDA_ENV="${CONDA_ENV:-pytorch}"
FPS="${FPS:-120}"
TIMEOUT="${TIMEOUT:-240}"

BUNDLES=(
  pointnet_tmax_fixed_special5_seed0_split0
  mph_gait_fixed_special5_seed0_split0
  lidargaitpp_fixed_special5_seed0_split0
)

for bundle in "${BUNDLES[@]}"; do
  printf '\n=== Realtime enrollment smoke: %s ===\n' "${bundle}"
  conda run --no-capture-output -n "${CONDA_ENV}" \
    python mph_gait_id/scripts/replay_enrollment_smoke.py \
      --replay "${REPLAY}" \
      --bundle "${bundle}" \
      --fps "${FPS}" \
      --timeout "${TIMEOUT}"
done
