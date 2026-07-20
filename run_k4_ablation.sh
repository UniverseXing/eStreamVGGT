#!/usr/bin/env bash

# K=4 composition ablation: anchor + 2 DINO-diverse old frames + 1 recent frame.
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
policy="anchor_recent_dino_diverse_2old_1recent"

for run in 1 2 3; do
    echo "===== K=4 composition ablation, repeat ${run}/3 ====="
    (
        cd "${repo_root}/src"
        STREAMVGGT_EVAL_DATASETS="sintel bonn" \
        STREAMVGGT_CACHE_WINDOW=4 \
        STREAMVGGT_CACHE_POLICY="${policy}" \
        STREAMVGGT_RUN_TAG="ablation_run${run}" \
        STREAMVGGT_LOG_SELECTIONS=0 \
        bash eval/video_depth/run.sh
    )
done
