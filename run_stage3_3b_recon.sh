#!/usr/bin/env bash

# Stage 3.3B formal matrix: full, Stage 3.2 K4, uniform K6, old-DINO K6.
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
datasets="${STREAMVGGT_STAGE3_3B_DATASETS:-7scenes nrgbd eth3d}"

run_eval() {
    local label="$1"
    local protocol="$2"
    local window="$3"
    local policy="$4"
    local prefix_frames=""
    if [[ "${protocol}" == "dense" ]]; then
        prefix_frames="4 6 8 10"
    fi
    echo "===== Stage 3.3B ${protocol}: ${label} on ${datasets} ====="
    (
        cd "${repo_root}/src"
        unset STREAMVGGT_CACHE_WINDOW STREAMVGGT_CACHE_POLICY
        if [[ -n "${window}" ]]; then
            export STREAMVGGT_CACHE_WINDOW="${window}"
            export STREAMVGGT_CACHE_POLICY="${policy}"
        fi
        STREAMVGGT_MV_DATASETS="${datasets}" \
        STREAMVGGT_MV_PROTOCOL="${protocol}" \
        STREAMVGGT_MV_PREFIX_FRAMES="${prefix_frames}" \
        STREAMVGGT_MV_SAVE_ARTIFACTS="${STREAMVGGT_STAGE3_3B_SAVE_ARTIFACTS:-0}" \
        STREAMVGGT_MV_RUN_TAG="stage3_3b_${label}" \
        bash eval/mv_recon/run_streaming_recon.sh
    )
}

run_eval paper_full_cache paper "" ""
run_eval dense_full_cache dense "" ""
run_eval dense_stage3_2_k4 dense 4 anchor_recent_dino_diverse_2old_1recent
run_eval dense_uniform_k6 dense 6 anchor_recent_uniform
run_eval dense_old_dino_k6 dense 6 anchor_recent_dino_diverse

if [[ "${STREAMVGGT_STAGE3_3B_RUN_FIFO:-0}" == "1" ]]; then
    (
        export STREAMVGGT_MV_MAX_SCENES=1
        run_eval dense_fifo_k6 dense 6 fifo
    )
fi

python "${repo_root}/scripts/summarize_stage3_3b_recon.py" \
    --results-root "${repo_root}/eval_results/mv_recon" \
    --name-filter stage3_3b \
    --output "${repo_root}/stage3_3b_recon_results.csv"
