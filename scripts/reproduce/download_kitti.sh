#!/usr/bin/env bash

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
# shellcheck source=common.sh
source "${script_dir}/common.sh"

usage() {
    cat <<'EOF'
Usage: scripts/reproduce/download_kitti.sh

Download the 13 KITTI validation drives used by the paper and prepare the
first-110-frame VideoDepth layout.

Environment variables:
  KITTI_ROOT        Default: data/eval/kitti.
  PREPARE           Run the layout preparation after download. Default: 1.
  VALIDATE          Validate all prepared RGB/depth pairs. Default: 1.
  FRAMES_PER_DRIVE  Frozen at 110; other values are rejected.
  MATERIALIZE_MODE  hardlink (default), symlink or copy.
  DRY_RUN           Print download/preparation commands only.
  PYTHON_BIN        Python executable override.
EOF
}

only_help_or_no_args usage "$@"
validate_switch PREPARE "${PREPARE:-1}"
validate_switch VALIDATE "${VALIDATE:-1}"
resolve_repo_path "${KITTI_ROOT:-data/eval/kitti}" kitti_root
frames_per_drive="${FRAMES_PER_DRIVE:-110}"
[[ "${frames_per_drive}" == "110" ]] || {
    die "the frozen KITTI protocol requires FRAMES_PER_DRIVE=110"
}

note "downloading KITTI protocol sources"
run_in_dir "${REPO_ROOT}" env \
    "STREAMVGGT_KITTI_ROOT=${kitti_root}" \
    bash "${REPO_ROOT}/scripts/download_stage4a_kitti.sh"

if [[ "${PREPARE:-1}" == "1" || "${VALIDATE:-1}" == "1" ]]; then
    init_runtime 0
fi
if [[ "${PREPARE:-1}" == "1" ]]; then
    note "preparing the KITTI first-110-frame layout"
    run_in_dir "${REPO_ROOT}" "${PYTHON_BIN}" \
        "${REPO_ROOT}/scripts/prepare_stage4a_kitti.py" \
        --root "${kitti_root}" \
        --frames-per-drive "${frames_per_drive}" \
        --mode "${MATERIALIZE_MODE:-hardlink}"
fi

if [[ "${VALIDATE:-1}" == "1" ]]; then
    note "validating the prepared KITTI protocol"
    run_in_dir "${REPO_ROOT}" "${PYTHON_BIN}" \
        "${REPO_ROOT}/scripts/check_stage4a_kitti.py" \
        --root "${kitti_root}"
fi
