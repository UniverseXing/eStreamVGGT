#!/usr/bin/env bash

# Stage 2 timing repeat 2. Run this from the repository root.
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

run_eval() {
    local policy="$1"
    local window="$2"
    echo "===== run2: ${policy}, K=${window} ====="
    (
        cd "${repo_root}/src"
        STREAMVGGT_EVAL_DATASETS="sintel bonn" \
        STREAMVGGT_CACHE_WINDOW="${window}" \
        STREAMVGGT_CACHE_POLICY="${policy}" \
        STREAMVGGT_RUN_TAG="run2" \
        STREAMVGGT_LOG_SELECTIONS=0 \
        bash eval/video_depth/run.sh
    )
}

run_eval anchor_recent_image_diff 6
run_eval anchor_recent_oldest_valid 4
run_eval anchor_recent_uniform 6
run_eval anchor_recent_dino_diverse 4
run_eval anchor_recent_oldest_valid 6
run_eval anchor_recent_image_diff 4
run_eval anchor_recent_dino_diverse 6
run_eval anchor_recent_uniform 4
