#!/usr/bin/env bash

# Trace cache, input, output, and CUDA memory for the four Stage 3 methods.
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
dataset="${STREAMVGGT_MEMORY_DATASET:-bonn}"
sequence="${STREAMVGGT_MEMORY_SEQUENCE:-balloon2}"
max_frames="${STREAMVGGT_MEMORY_FRAMES:-110}"
old_policy="anchor_recent_dino_diverse"
new_policy="anchor_recent_dino_diverse_2old_1recent"

run_eval() {
    local label="$1"
    local cache_window="$2"
    local cache_policy="$3"
    echo "===== Stage 3 memory breakdown: ${label}, ${sequence}, N=${max_frames} ====="
    (
        cd "${repo_root}/src"
        unset STREAMVGGT_CACHE_WINDOW STREAMVGGT_CACHE_POLICY
        if [[ -n "${cache_window}" ]]; then
            export STREAMVGGT_CACHE_WINDOW="${cache_window}"
            export STREAMVGGT_CACHE_POLICY="${cache_policy}"
        fi
        STREAMVGGT_EVAL_DATASETS="${dataset}" \
        STREAMVGGT_SEQ_LIST="${sequence}" \
        STREAMVGGT_MAX_FRAMES="${max_frames}" \
        STREAMVGGT_RUN_TAG="memory_${label}_${sequence}" \
        STREAMVGGT_LOG_SELECTIONS=0 \
        STREAMVGGT_TRACE_MEMORY=1 \
        bash eval/video_depth/run.sh
    )
}

run_eval full_cache "" ""
run_eval old_k4 4 "${old_policy}"
run_eval old_k6 6 "${old_policy}"
run_eval new_k4 4 "${new_policy}"

python3 "${repo_root}/scripts/summarize_stage3_memory_trace.py" \
    --results-root "${repo_root}/eval_results/video_depth" \
    --dataset "${dataset}" \
    --sequence "${sequence}" \
    --frames "${max_frames}"
