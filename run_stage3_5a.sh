#!/usr/bin/env bash

# Stage 3.5A: diagnose recent-frame continuity on Bonn person_tracking2.
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
results_root="${STREAMVGGT_STAGE3_5A_RESULTS_ROOT:-${repo_root}/eval_results/stage3_5a}"
weights="${STREAMVGGT_STAGE3_5A_WEIGHTS:-${repo_root}/ckpt/checkpoints.pth}"
data_root="${STREAMVGGT_STAGE3_5A_BONN_ROOT:-${repo_root}/data/eval/bonn/rgbd_bonn_dataset}"
sequence="${STREAMVGGT_STAGE3_5A_SEQUENCE:-person_tracking2}"
prefixes="${STREAMVGGT_STAGE3_5A_PREFIX_FRAMES:-10 20 30 40 50 60 70 80 90 100 110}"

run_eval() {
    local label="$1"
    local window="$2"
    local policy="$3"
    local -a command prefix_args

    read -r -a prefix_args <<< "${prefixes}"
    command=(
        python "${repo_root}/src/eval/long_sequence/eval_stage3_4_long.py"
        --weights "${weights}"
        --dataset bonn
        --data-root "${data_root}"
        --output-dir "${results_root}/bonn_${label}"
        --seq-list "${sequence}"
        --size "${STREAMVGGT_STAGE3_5A_SIZE:-518}"
        --prefix-frames "${prefix_args[@]}"
        --trace-memory
    )
    if [[ -n "${STREAMVGGT_STAGE3_5A_MAX_FRAMES:-}" ]]; then
        command+=(--max-frames "${STREAMVGGT_STAGE3_5A_MAX_FRAMES}")
    fi
    if [[ -n "${window}" ]]; then
        command+=(--cache-window "${window}" --cache-policy "${policy}")
    fi

    echo "===== Stage 3.5A ${sequence}: ${label} ====="
    "${command[@]}"
}

# The matrix is deliberately small. uniform_k6 is retired from future runs;
# FIFO K4 remains only as a causal diagnostic for temporal recency.
run_eval full_cache "" ""
run_eval stage3_2_k4 4 anchor_recent_dino_diverse_2old_1recent
run_eval old_k4 4 anchor_recent_dino_diverse
run_eval fifo_k4 4 fifo
run_eval old_dino_k6 6 anchor_recent_dino_diverse

python "${repo_root}/scripts/summarize_stage3_4.py" \
    --results-root "${results_root}" \
    --output "${repo_root}/stage3_5a_results.csv" \
    --sequence-output "${repo_root}/stage3_5a_sequence_results.csv"

