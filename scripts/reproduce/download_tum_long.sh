#!/usr/bin/env bash

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
# shellcheck source=common.sh
source "${script_dir}/common.sh"

usage() {
    cat <<'EOF'
Usage: scripts/reproduce/download_tum_long.sh

Download the three raw TUM RGB-D sequences used by the held-out long-sequence
matrix. Only RGB images, timestamps and pose ground truth are extracted.

Environment variables:
  LONG_DATA_ROOT   Default: data/eval/stage4c_tum.
  DELETE_ARCHIVES  Delete downloaded .tgz files after extraction. Default: 0.
  VALIDATE         Validate 1000 associated frames per sequence. Default: 1.
  MIN_FRAMES       Validation threshold. Default: 1000.
  DRY_RUN          Print the download command only.
EOF
}

only_help_or_no_args usage "$@"
validate_switch DELETE_ARCHIVES "${DELETE_ARCHIVES:-0}"
validate_switch VALIDATE "${VALIDATE:-1}"
resolve_repo_path "${LONG_DATA_ROOT:-data/eval/stage4c_tum}" data_root

note "downloading held-out raw-TUM long sequences"
run_in_dir "${REPO_ROOT}" env \
    "STREAMVGGT_STAGE4C_DATA_ROOT=${data_root}" \
    "STREAMVGGT_STAGE4C_DELETE_ARCHIVES=${DELETE_ARCHIVES:-0}" \
    bash "${REPO_ROOT}/scripts/download_stage4c_tum.sh"

if [[ "${VALIDATE:-1}" == "1" ]]; then
    init_runtime 0
    note "validating held-out raw-TUM coverage"
    run_in_dir "${REPO_ROOT}" "${PYTHON_BIN}" \
        "${REPO_ROOT}/scripts/check_stage4c_data.py" \
        --root "${data_root}" \
        --min-frames "${MIN_FRAMES:-1000}"
fi
