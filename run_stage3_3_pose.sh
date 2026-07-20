#!/usr/bin/env bash

# Stage 3.3 initial pose matrix. Defaults to the already-prepared Sintel set.
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
datasets="${STREAMVGGT_STAGE3_3_POSE_DATASETS:-sintel}"

run_eval() {
    local label="$1"
    local window="$2"
    local policy="$3"
    echo "===== Stage 3.3 pose: ${label} on ${datasets} ====="
    (
        cd "${repo_root}/src"
        unset STREAMVGGT_CACHE_WINDOW STREAMVGGT_CACHE_POLICY
        if [[ -n "${window}" ]]; then
            export STREAMVGGT_CACHE_WINDOW="${window}"
            export STREAMVGGT_CACHE_POLICY="${policy}"
        fi
        STREAMVGGT_POSE_DATASETS="${datasets}" \
        STREAMVGGT_RUN_TAG="stage3_3_${label}" \
        bash eval/pose_evaluation/run_streaming_pose.sh
    )
}

run_eval full_cache "" ""
run_eval stage3_2_k4 4 anchor_recent_dino_diverse_2old_1recent
run_eval fifo_k6 6 fifo
run_eval uniform_k6 6 anchor_recent_uniform
run_eval old_dino_k6 6 anchor_recent_dino_diverse

python "${repo_root}/scripts/summarize_stage3_3_pose.py" \
    --results-root "${repo_root}/eval_results/pose" \
    --name-filter stage3_3 \
    --output "${repo_root}/stage3_3_pose_results.csv"
