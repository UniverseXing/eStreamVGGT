#!/usr/bin/env bash

# Stage 3.5B: decouple geometry/camera caches and test a recency-heavy DINO K6.
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
results_root="${STREAMVGGT_STAGE3_5B_RESULTS_ROOT:-${repo_root}/eval_results/stage3_5b}"
weights="${STREAMVGGT_STAGE3_5B_WEIGHTS:-${repo_root}/ckpt/checkpoints.pth}"
data_root="${STREAMVGGT_STAGE3_5B_BONN_ROOT:-${repo_root}/data/eval/bonn/rgbd_bonn_dataset}"
sequence="${STREAMVGGT_STAGE3_5B_SEQUENCE:-person_tracking2}"
prefixes="${STREAMVGGT_STAGE3_5B_PREFIX_FRAMES:-10 20 30 40 50 60 70 80 90 100 110}"

if [[ -z "${CONDA_PREFIX:-}" || ! -x "${CONDA_PREFIX}/bin/python" ]]; then
    echo "Stage 3.5B requires an activated Conda environment with CONDA_PREFIX/bin/python" >&2
    echo "CONDA_DEFAULT_ENV=${CONDA_DEFAULT_ENV:-unset}, CONDA_PREFIX=${CONDA_PREFIX:-unset}" >&2
    exit 2
fi
python_bin="${CONDA_PREFIX}/bin/python"
"${python_bin}" -c 'import numpy, sys; print("Stage 3.5B Python:", sys.executable, "NumPy:", numpy.__version__)'

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
        --size "${STREAMVGGT_STAGE3_5B_SIZE:-518}"
        --prefix-frames "${prefix_args[@]}"
        --trace-memory
    )
    if [[ -n "${STREAMVGGT_STAGE3_5B_MAX_FRAMES:-}" ]]; then
        command+=(--max-frames "${STREAMVGGT_STAGE3_5B_MAX_FRAMES}")
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

    echo "===== Stage 3.5B ${sequence}: ${label} ====="
    "${command[@]}"
}

# Self-contained controls. Their Stage 3.5A results already exist, but rerunning
# them here protects the gate from code/environment drift.
run_eval full_cache "" "" "" ""
run_eval stage3_2_k4 4 anchor_recent_dino_diverse_2old_1recent "" ""
run_eval old_dino_k6 6 anchor_recent_dino_diverse "" ""

# Same DINO K4 aggregator, independently selected camera history. K8/K16 mean
# one fixed anchor plus the most recent 7/15 frames, including the current frame.
run_eval split_k4_camera4 4 anchor_recent_dino_diverse_2old_1recent 4 anchor_recent
run_eval split_k4_camera8 4 anchor_recent_dino_diverse_2old_1recent 8 anchor_recent
run_eval split_k4_camera16 4 anchor_recent_dino_diverse_2old_1recent 16 anchor_recent
run_eval split_k4_camera_full 4 anchor_recent_dino_diverse_2old_1recent "" full

# New K6: anchor + one DINO historical slot + three recent historical frames
# + current. It trades one old-DINO slot from old_dino_k6 for one recent slot.
run_eval recent_dino_k6 6 anchor_recent_dino_diverse_1old_3recent "" ""

"${python_bin}" "${repo_root}/scripts/summarize_stage3_4.py" \
    --results-root "${results_root}" \
    --output "${repo_root}/stage3_5b_results.csv" \
    --sequence-output "${repo_root}/stage3_5b_sequence_results.csv"

# This writes a decision table but intentionally does not fail the SLURM job
# when no method passes: a negative diagnostic result is still a valid result.
"${python_bin}" "${repo_root}/scripts/check_stage3_5b_gate.py" \
    --input "${repo_root}/stage3_5b_results.csv" \
    --output "${repo_root}/stage3_5b_gate.csv"
