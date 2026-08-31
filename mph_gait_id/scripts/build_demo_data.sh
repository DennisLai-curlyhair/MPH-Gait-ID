#!/usr/bin/env bash
set -euo pipefail

SYSTEM_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ROOT_DIR="$(dirname "${SYSTEM_DIR}")"
DEST="${DEMO_DEST:-${ROOT_DIR}/demo_data}"
RAW_ROOT="${ROOT_DIR}/dataset/PersonRecognitionWalking_29"

if [[ -e "${DEST}" ]]; then
  echo "Destination already exists: ${DEST}" >&2
  echo "Move or remove it explicitly before rebuilding." >&2
  exit 1
fi

copy_sequence() {
  local scope="$1"
  local person_number="$2"
  local sequence="$3"
  local person_dir
  local person_label
  local raw_source
  local raw_target

  person_dir="$(printf 'person_%03d' "${person_number}")"
  person_label="$(printf 'P%03d' "${person_number}")"
  raw_source="${RAW_ROOT}/${person_dir}/${sequence}"
  raw_target="${DEST}/${scope}/pointcloud/${person_label}/${sequence}"

  if [[ ! -d "${raw_source}" ]]; then
    echo "Missing raw point-cloud sequence: ${raw_source}" >&2
    exit 1
  fi
  mkdir -p "$(dirname "${raw_target}")"
  cp -a "${raw_source}" "${raw_target}"
}

for person_number in 1 5 26; do
  person_short="P${person_number}"
  copy_sequence "gallery" "${person_number}" "${person_short}_C2_V019"
  copy_sequence "gallery" "${person_number}" "${person_short}_C3_V020"
  copy_sequence "probe/personal" "${person_number}" "${person_short}_C1_V001"
done

copy_sequence "probe/special" 26 "P26_C4_V001"
copy_sequence "probe/special" 26 "P26_C5_V001"
copy_sequence "probe/special" 26 "P26_C7_V001"

echo "Demo data copied to ${DEST}"
