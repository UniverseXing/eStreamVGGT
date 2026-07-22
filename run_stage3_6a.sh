#!/usr/bin/env bash

# Stage 3.6A: bounded overlapping-window pose stitching feasibility test.
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
results_root="${STREAMVGGT_STAGE3_6A_RESULTS_ROOT:-${repo_root}/eval_results/stage3_6a}"
weights="${STREAMVGGT_STAGE3_6A_WEIGHTS:-${repo_root}/ckpt/checkpoints.pth}"
data_root="${STREAMVGGT_STAGE3_6A_BONN_ROOT:-${repo_root}/data/eval/bonn/rgbd_bonn_dataset}"
sequence="${STREAMVGGT_STAGE3_6A_SEQUENCE:-person_tracking2}"
prefixes="${STREAMVGGT_STAGE3_6A_PREFIX_FRAMES:-10 20 30 40 50 60 70 80 90 100 110}"

if [[ -z "${CONDA_PREFIX:-}" || ! -x "${CONDA_PREFIX}/bin/python" ]]; then
    echo "Stage 3.6A requires an activated Conda environment with CONDA_PREFIX/bin/python" >&2
    echo "CONDA_DEFAULT_ENV=${CONDA_DEFAULT_ENV:-unset}, CONDA_PREFIX=${CONDA_PREFIX:-unset}" >&2
    exit 2
fi
python_bin="${CONDA_PREFIX}/bin/python"
"${python_bin}" -c 'import numpy, sys; print("Stage 3.6A Python:", sys.executable, "NumPy:", numpy.__version__)'

run_eval() {
    local label="$1"
    shift
    local -a command prefix_args
    read -r -a prefix_args <<< "${prefixes}"
    command=(
        "${python_bin}" "${repo_root}/src/eval/pose_evaluation/eval_stage3_6a_window_pose.py"
        --weights "${weights}"
        --data-root "${data_root}"
        --sequence "${sequence}"
        --output-dir "${results_root}/${label}"
        --method "${label}"
        --size "${STREAMVGGT_STAGE3_6A_SIZE:-518}"
        --prefix-frames "${prefix_args[@]}"
        "$@"
    )
    if [[ -n "${STREAMVGGT_STAGE3_6A_MAX_FRAMES:-}" ]]; then
        command+=(--max-frames "${STREAMVGGT_STAGE3_6A_MAX_FRAMES}")
    fi
    echo "===== Stage 3.6A ${sequence}: ${label} ====="
    "${command[@]}"
}

run_eval full_cache --mode stream
run_eval temporal_binned_dino_k8 \
    --mode stream --cache-window 8 --cache-policy temporal_binned_dino_k8
run_eval window16_overlap4 --mode window_stitch --window-size 16 --overlap 4
run_eval window32_overlap8 --mode window_stitch --window-size 32 --overlap 8

"${python_bin}" "${repo_root}/scripts/summarize_stage3_6a.py" \
    --results-root "${results_root}" \
    --output "${repo_root}/stage3_6a_results.csv"

"${python_bin}" "${repo_root}/scripts/check_stage3_6a_gate.py" \
    --input "${repo_root}/stage3_6a_results.csv" \
    --output "${repo_root}/stage3_6a_gate.csv"
