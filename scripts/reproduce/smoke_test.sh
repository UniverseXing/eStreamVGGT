#!/usr/bin/env bash

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
# shellcheck source=common.sh
source "${script_dir}/common.sh"

usage() {
    cat <<'EOF'
Usage: scripts/reproduce/smoke_test.sh

Run one real bounded-cache inference on examples/example_building. This checks
the checkpoint, CUDA environment, model loading, DINO selection and prediction
heads without downloading an evaluation dataset.

Environment variables:
  METHOD       One bounded method (K4 by default; K6 and K8 are accepted).
  IMAGES_DIR   Image directory. Default: examples/example_building.
  WEIGHTS      Checkpoint. Default: ckpt/checkpoints.pth.
  MAX_FRAMES   Optional input cap; it must exceed the selected cache window.
  DRY_RUN      Set to 1 to print the command without checking GPU/weights.
  PYTHON_BIN   Python executable override.
EOF
    common_usage
}

only_help_or_no_args usage "$@"
init_runtime 0

method="${METHOD:-anchor_recent_dino_diverse_k4}"
case "${method}" in
    anchor_recent_dino_diverse_k4|anchor_recent_dino_diverse_k6|anchor_recent_dino_diverse_k8) ;;
    *) die "METHOD must be one of the three bounded public methods" ;;
esac

resolve_repo_path "${WEIGHTS:-ckpt/checkpoints.pth}" weights
resolve_repo_path "${IMAGES_DIR:-examples/example_building}" images_dir
require_file "${weights}" "checkpoint"
if [[ "${DRY_RUN}" != "1" && ! -d "${images_dir}" ]]; then
    die "missing smoke-test image directory: ${images_dir}"
fi
runtime_preflight

args=(
    "${PYTHON_BIN}" "${script_dir}/smoke_inference.py"
    --repo-root "${REPO_ROOT}"
    --weights "${weights}"
    --images-dir "${images_dir}"
    --method "${method}"
)
if [[ -n "${MAX_FRAMES:-}" ]]; then
    args+=(--max-frames "${MAX_FRAMES}")
fi

note "bounded smoke test: ${method}"
run_command "${args[@]}"
