#!/usr/bin/env bash

# Stage 3.2: full-cache and three K=4 policies on identical Bonn prefixes.
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
dataset="${STREAMVGGT_STAGE3_2_DATASET:-bonn}"
sequence="${STREAMVGGT_STAGE3_2_SEQUENCE:-balloon2}"
prefixes="${STREAMVGGT_STAGE3_2_PREFIXES:-25 50 75 100 110}"
adaptive_threshold="${STREAMVGGT_ADAPTIVE_SIM_THRESHOLD:-0.99}"
adaptive_min_gap="${STREAMVGGT_ADAPTIVE_MIN_GAP:-8}"
threshold_tag="${adaptive_threshold//./p}"

old_policy="anchor_recent_dino_diverse"
fixed_policy="anchor_recent_dino_diverse_2old_1recent"
adaptive_policy="anchor_stable_adaptive_recent"

run_eval() {
    local label="$1"
    local max_frames="$2"
    local cache_policy="$3"
    echo "===== Stage 3.2: ${label}, ${sequence}, N=${max_frames} ====="
    (
        cd "${repo_root}/src"
        unset STREAMVGGT_CACHE_WINDOW STREAMVGGT_CACHE_POLICY
        if [[ -n "${cache_policy}" ]]; then
            export STREAMVGGT_CACHE_WINDOW=4
            export STREAMVGGT_CACHE_POLICY="${cache_policy}"
        fi
        STREAMVGGT_EVAL_DATASETS="${dataset}" \
        STREAMVGGT_SEQ_LIST="${sequence}" \
        STREAMVGGT_MAX_FRAMES="${max_frames}" \
        STREAMVGGT_RUN_TAG="stage3_2_${label}_${sequence}_tau${threshold_tag}_gap${adaptive_min_gap}" \
        STREAMVGGT_LOG_SELECTIONS=0 \
        STREAMVGGT_ADAPTIVE_SIM_THRESHOLD="${adaptive_threshold}" \
        STREAMVGGT_ADAPTIVE_MIN_GAP="${adaptive_min_gap}" \
        bash eval/video_depth/run.sh
    )
}

for max_frames in ${prefixes}; do
    run_eval full_cache "${max_frames}" ""
    run_eval old_k4 "${max_frames}" "${old_policy}"
    run_eval fixed_k4 "${max_frames}" "${fixed_policy}"
    run_eval adaptive_k4 "${max_frames}" "${adaptive_policy}"
done

python3 "${repo_root}/scripts/summarize_stage3_2_scaling.py" \
    --results-root "${repo_root}/eval_results/video_depth" \
    --dataset "${dataset}" \
    --sequence "${sequence}" \
    --prefixes ${prefixes} \
    --adaptive-threshold "${adaptive_threshold}" \
    --adaptive-min-gap "${adaptive_min_gap}" \
    --threshold-tag "${threshold_tag}"

# Keep formal timing free of logging synchronization, then run one diagnostic
# pass to expose adaptive replacement steps and similarity scores.
if [[ "${STREAMVGGT_STAGE3_2_SELECTION_TRACE:-1}" == "1" ]]; then
    (
        cd "${repo_root}/src"
        STREAMVGGT_EVAL_DATASETS="${dataset}" \
        STREAMVGGT_SEQ_LIST="${sequence}" \
        STREAMVGGT_MAX_FRAMES=110 \
        STREAMVGGT_CACHE_WINDOW=4 \
        STREAMVGGT_CACHE_POLICY="${adaptive_policy}" \
        STREAMVGGT_RUN_TAG="stage3_2_adaptive_selection_${sequence}_tau${threshold_tag}_gap${adaptive_min_gap}" \
        STREAMVGGT_LOG_SELECTIONS=1 \
        STREAMVGGT_ADAPTIVE_SIM_THRESHOLD="${adaptive_threshold}" \
        STREAMVGGT_ADAPTIVE_MIN_GAP="${adaptive_min_gap}" \
        bash eval/video_depth/run.sh
    )
fi
