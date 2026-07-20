#!/usr/bin/env bash

# Compare cache scaling on identical prefixes of one aligned video sequence.
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
dataset="${STREAMVGGT_SCALING_DATASET:-bonn}"
sequence="${STREAMVGGT_SCALING_SEQUENCE:-balloon2}"
prefixes="${STREAMVGGT_SCALING_PREFIXES:-25 50 75 100 110}"
old_policy="anchor_recent_dino_diverse"
new_policy="anchor_recent_dino_diverse_2old_1recent"

run_eval() {
    local label="$1"
    local max_frames="$2"
    local cache_window="$3"
    local cache_policy="$4"
    echo "===== Stage 3 scaling: ${label}, ${sequence}, N=${max_frames} ====="
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
        STREAMVGGT_RUN_TAG="scaling_${label}_${sequence}" \
        STREAMVGGT_LOG_SELECTIONS=0 \
        bash eval/video_depth/run.sh
    )
}

# Interleave methods at each prefix so temporal GPU variation affects them similarly.
for max_frames in ${prefixes}; do
    run_eval full_cache "${max_frames}" "" ""
    run_eval old_k4 "${max_frames}" 4 "${old_policy}"
    run_eval old_k6 "${max_frames}" 6 "${old_policy}"
    run_eval new_k4 "${max_frames}" 4 "${new_policy}"
done

python3 "${repo_root}/scripts/summarize_stage3_scaling.py" \
    --results-root "${repo_root}/eval_results/video_depth" \
    --dataset "${dataset}" \
    --sequence "${sequence}" \
    --prefixes ${prefixes}
