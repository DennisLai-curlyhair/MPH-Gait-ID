#!/usr/bin/env bash
set -euo pipefail

SYSTEM_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORKSPACE_ROOT="$(cd "${SYSTEM_ROOT}/.." && pwd)"
REPLAY_PATH="${1:-${SYSTEM_ROOT}/data/replay/P3_C4_V001}"
FPS="${FPS:-120}"
TIMEOUT="${TIMEOUT:-240}"

cd "${WORKSPACE_ROOT}"

BUNDLES=(
  pointnet_tmax_fixed_special5_seed0_split0
  mph_gait_fixed_special5_seed0_split0
  lidargaitpp_fixed_special5_seed0_split0
)

for bundle in "${BUNDLES[@]}"; do
  printf '\n===== %s =====\n' "${bundle}"
  bash mph_gait_id/scripts/run_replay_smoke.sh \
    --replay "${REPLAY_PATH}" \
    --bundle "${bundle}" \
    --fps "${FPS}" \
    --timeout "${TIMEOUT}"
done
