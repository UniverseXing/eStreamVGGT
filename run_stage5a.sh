#!/usr/bin/env bash

# Stage 5A conference experiment: same-budget K4 baselines on VideoDepth.
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Full and K4 reuse the frozen same-GPU Stage 4A/4B tables by default. Only
# the two missing controls require new inference.
core_methods="${STREAMVGGT_STAGE5A_METHODS:-recent4 anchor_recent4}"
if [[ "${STREAMVGGT_STAGE5A_RERUN_REFERENCES:-0}" == "1" ]]; then
    core_methods="full_cache ${core_methods} proposed_k4"
fi
methods="${core_methods}"
if [[ "${STREAMVGGT_STAGE5A_INCLUDE_OPTIONAL:-0}" == "1" ]]; then
    methods="${methods} anchor_uniform4 random4_seed0 random4_seed1 random4_seed2"
fi
datasets="${STREAMVGGT_STAGE5A_DATASETS:-bonn sintel kitti}"

method_config() {
    case "$1" in
        full_cache)       METHOD_WINDOW=""; METHOD_POLICY=""; METHOD_SEED=0 ;;
        recent4)          METHOD_WINDOW=4; METHOD_POLICY=fifo; METHOD_SEED=0 ;;
        anchor_recent4)   METHOD_WINDOW=4; METHOD_POLICY=anchor_recent; METHOD_SEED=0 ;;
        anchor_uniform4)  METHOD_WINDOW=4; METHOD_POLICY=anchor_uniform_k4; METHOD_SEED=0 ;;
        random4_seed0)    METHOD_WINDOW=4; METHOD_POLICY=random_reservoir_k4; METHOD_SEED=0 ;;
        random4_seed1)    METHOD_WINDOW=4; METHOD_POLICY=random_reservoir_k4; METHOD_SEED=1 ;;
        random4_seed2)    METHOD_WINDOW=4; METHOD_POLICY=random_reservoir_k4; METHOD_SEED=2 ;;
        proposed_k4)      METHOD_WINDOW=4; METHOD_POLICY=anchor_recent_dino_diverse_k4; METHOD_SEED=0 ;;
        *) echo "Unknown Stage 5A method: $1" >&2; exit 2 ;;
    esac
}

for method in ${methods}; do
    method_config "${method}"
    echo "===== Stage 5A ${method}: ${datasets} ====="
    (
        cd "${repo_root}/src"
        export STREAMVGGT_EVAL_DATASETS="${datasets}"
        export STREAMVGGT_RUN_TAG="stage5a_${method}"
        export STREAMVGGT_MAX_FRAMES="${STREAMVGGT_STAGE5A_MAX_FRAMES:-}"
        export STREAMVGGT_LOG_SELECTIONS="${STREAMVGGT_STAGE5A_LOG_SELECTIONS:-1}"
        export STREAMVGGT_CACHE_RANDOM_SEED="${METHOD_SEED}"
        if [[ -n "${METHOD_WINDOW}" ]]; then
            export STREAMVGGT_CACHE_WINDOW="${METHOD_WINDOW}"
            export STREAMVGGT_CACHE_POLICY="${METHOD_POLICY}"
        else
            unset STREAMVGGT_CACHE_WINDOW STREAMVGGT_CACHE_POLICY
        fi
        bash eval/video_depth/run.sh
    )
done
