#!/usr/bin/env bash

# Stage 3.5C-1: cross aggregator recency with camera history and test DINO K8.
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
results_root="${STREAMVGGT_STAGE3_5C1_RESULTS_ROOT:-${repo_root}/eval_results/stage3_5c1}"
weights="${STREAMVGGT_STAGE3_5C1_WEIGHTS:-${repo_root}/ckpt/checkpoints.pth}"
data_root="${STREAMVGGT_STAGE3_5C1_BONN_ROOT:-${repo_root}/data/eval/bonn/rgbd_bonn_dataset}"
sequence="${STREAMVGGT_STAGE3_5C1_SEQUENCE:-person_tracking2}"
prefixes="${STREAMVGGT_STAGE3_5C1_PREFIX_FRAMES:-10 20 30 40 50 60 70 80 90 100 110}"

if [[ -z "${CONDA_PREFIX:-}" || ! -x "${CONDA_PREFIX}/bin/python" ]]; then
    echo "Stage 3.5C-1 requires an activated Conda environment with CONDA_PREFIX/bin/python" >&2
    echo "CONDA_DEFAULT_ENV=${CONDA_DEFAULT_ENV:-unset}, CONDA_PREFIX=${CONDA_PREFIX:-unset}" >&2
    exit 2
fi
python_bin="${CONDA_PREFIX}/bin/python"
"${python_bin}" -c 'import numpy, sys; print("Stage 3.5C-1 Python:", sys.executable, "NumPy:", numpy.__version__)'

run_eval() {
    local label="$1"
    local aggregator_window="$2"
    local aggregator_policy="$3"
    local camera_window="$4"
    local camera_policy="$5"
    local -a command prefix_args

    read -r -a prefix_args <<< "${prefixes}"
    command=(
        "${python_bin}" "${repo_root}/src/eval/long_sequence/eval_stage3_4_long.py"
        --weights "${weights}"
        --dataset bonn
        --data-root "${data_root}"
        --output-dir "${results_root}/bonn_${label}"
        --seq-list "${sequence}"
        --size "${STREAMVGGT_STAGE3_5C1_SIZE:-518}"
        --prefix-frames "${prefix_args[@]}"
        --trace-memory
    )
    if [[ -n "${STREAMVGGT_STAGE3_5C1_MAX_FRAMES:-}" ]]; then
        command+=(--max-frames "${STREAMVGGT_STAGE3_5C1_MAX_FRAMES}")
    fi
    if [[ -n "${aggregator_window}" ]]; then
        command+=(
            --cache-window "${aggregator_window}"
            --cache-policy "${aggregator_policy}"
        )
    fi
    if [[ -n "${camera_policy}" ]]; then
        command+=(--camera-cache-policy "${camera_policy}")
        if [[ -n "${camera_window}" ]]; then
            command+=(--camera-cache-window "${camera_window}")
        fi
    fi

    echo "===== Stage 3.5C-1 ${sequence}: ${label} ====="
    "${command[@]}"
}

# Self-contained controls for quality, recency, and the current best K6 tradeoff.
run_eval full_cache "" "" "" ""
run_eval stage3_2_k4 4 anchor_recent_dino_diverse_2old_1recent "" ""
run_eval fifo_k4 4 fifo "" ""
run_eval old_dino_k6 6 anchor_recent_dino_diverse "" ""

# Reverse the Stage 3.5B crossover: give the aggregator only recent frames and
# let the camera head retain bounded or complete global history.
run_eval fifo_k4_camera16 4 fifo 16 anchor_recent
run_eval fifo_k4_camera_full 4 fifo "" full
run_eval anchor_recent_k4_camera_full 4 anchor_recent "" full

# Capacity control: anchor + three DINO historical slots + three recent
# historical frames + current. This uses the existing general DINO policy.
run_eval standard_dino_k8 8 anchor_recent_dino_diverse "" ""

"${python_bin}" "${repo_root}/scripts/summarize_stage3_4.py" \
    --results-root "${results_root}" \
    --output "${repo_root}/stage3_5c1_results.csv" \
    --sequence-output "${repo_root}/stage3_5c1_sequence_results.csv"

"${python_bin}" "${repo_root}/scripts/check_stage3_5b_gate.py" \
    --input "${repo_root}/stage3_5c1_results.csv" \
    --output "${repo_root}/stage3_5c1_gate.csv" \
    --candidates \
        fifo_k4_camera16 \
        fifo_k4_camera_full \
        anchor_recent_k4_camera_full \
        standard_dino_k8
