#!/usr/bin/env bash

# Stage 3.3C: 50-frame TUM-dynamics reconstruction/cache-policy matrix.
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
sequences="${STREAMVGGT_STAGE3_3C_SEQUENCES:-}"

run_eval() {
    local label="$1"
    local window="$2"
    local policy="$3"
    echo "===== Stage 3.3C: ${label} on TUM-dynamics (50 frames) ====="
    (
        cd "${repo_root}/src"
        unset STREAMVGGT_CACHE_WINDOW STREAMVGGT_CACHE_POLICY
        if [[ -n "${window}" ]]; then
            export STREAMVGGT_CACHE_WINDOW="${window}"
            export STREAMVGGT_CACHE_POLICY="${policy}"
        fi
        STREAMVGGT_MV_DATASETS=tum \
        STREAMVGGT_MV_PROTOCOL=paper \
        STREAMVGGT_MV_TUM_FRAMES="${STREAMVGGT_STAGE3_3C_TUM_FRAMES:-50}" \
        STREAMVGGT_MV_TUM_SAMPLING="${STREAMVGGT_STAGE3_3C_TUM_SAMPLING:-first}" \
        STREAMVGGT_MV_PREFIX_FRAMES="${STREAMVGGT_STAGE3_3C_PREFIX_FRAMES:-10 20 30 40 50}" \
        STREAMVGGT_MV_SEQ_LIST="${sequences}" \
        STREAMVGGT_MV_MAX_SCENES="${STREAMVGGT_STAGE3_3C_MAX_SCENES:-}" \
        STREAMVGGT_MV_MAX_FRAMES="${STREAMVGGT_STAGE3_3C_MAX_FRAMES:-}" \
        STREAMVGGT_MV_SAVE_ARTIFACTS="${STREAMVGGT_STAGE3_3C_SAVE_ARTIFACTS:-0}" \
        STREAMVGGT_MV_RUN_TAG="stage3_3c_${label}" \
        bash eval/mv_recon/run_streaming_recon.sh
    )
}

run_eval full_cache "" ""
run_eval stage3_2_k4 4 anchor_recent_dino_diverse_2old_1recent
run_eval uniform_k6 6 anchor_recent_uniform
run_eval old_dino_k6 6 anchor_recent_dino_diverse

if [[ "${STREAMVGGT_STAGE3_3C_RUN_FIFO:-0}" == "1" ]]; then
    (
        export STREAMVGGT_STAGE3_3C_MAX_SCENES=1
        run_eval fifo_k6 6 fifo
    )
fi

python "${repo_root}/scripts/summarize_stage3_3b_recon.py" \
    --results-root "${repo_root}/eval_results/mv_recon" \
    --name-filter stage3_3c \
    --output "${repo_root}/stage3_3c_recon_results.csv"
