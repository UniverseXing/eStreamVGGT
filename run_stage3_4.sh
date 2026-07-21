#!/usr/bin/env bash

# Stage 3.4: long-sequence scaling and 7-Scenes synthetic loop evaluation.
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
results_root="${STREAMVGGT_STAGE3_4_RESULTS_ROOT:-${repo_root}/eval_results/stage3_4}"
weights="${STREAMVGGT_STAGE3_4_WEIGHTS:-${repo_root}/ckpt/checkpoints.pth}"
parts="${STREAMVGGT_STAGE3_4_PARTS:-bonn 7scenes_loop}"

run_eval() {
    local dataset="$1"
    local label="$2"
    local window="$3"
    local policy="$4"
    local root prefixes sequence_string
    local -a command sequence_args=() max_args=()

    if [[ "${dataset}" == "bonn" ]]; then
        root="${STREAMVGGT_STAGE3_4_BONN_ROOT:-${repo_root}/data/eval/bonn/rgbd_bonn_dataset}"
        prefixes="${STREAMVGGT_STAGE3_4_BONN_PREFIX_FRAMES:-10 20 30 40 50 60 70 80 90 100 110}"
        sequence_string="${STREAMVGGT_STAGE3_4_BONN_SEQUENCES:-}"
    else
        root="${STREAMVGGT_STAGE3_4_7SCENES_ROOT:-${repo_root}/data/eval/7scenes}"
        prefixes="${STREAMVGGT_STAGE3_4_LOOP_PREFIX_FRAMES:-10 20 30 40 50 60 70 80 90 100}"
        sequence_string="${STREAMVGGT_STAGE3_4_7SCENES_SEQUENCES:-}"
    fi
    if [[ -n "${sequence_string}" ]]; then
        read -r -a sequence_args <<< "${sequence_string}"
        sequence_args=(--seq-list "${sequence_args[@]}")
    fi
    if [[ -n "${STREAMVGGT_STAGE3_4_MAX_SEQUENCES:-}" ]]; then
        max_args+=(--max-sequences "${STREAMVGGT_STAGE3_4_MAX_SEQUENCES}")
    fi
    if [[ -n "${STREAMVGGT_STAGE3_4_MAX_FRAMES:-}" ]]; then
        max_args+=(--max-frames "${STREAMVGGT_STAGE3_4_MAX_FRAMES}")
    fi

    read -r -a prefix_args <<< "${prefixes}"
    command=(
        python "${repo_root}/src/eval/long_sequence/eval_stage3_4_long.py"
        --weights "${weights}"
        --dataset "${dataset}"
        --data-root "${root}"
        --output-dir "${results_root}/${dataset}_${label}"
        --size "${STREAMVGGT_STAGE3_4_SIZE:-518}"
        --prefix-frames "${prefix_args[@]}"
        --loop-forward-frames "${STREAMVGGT_STAGE3_4_LOOP_FORWARD_FRAMES:-50}"
        "${sequence_args[@]}"
        "${max_args[@]}"
    )
    if [[ -n "${window}" ]]; then
        command+=(--cache-window "${window}" --cache-policy "${policy}")
    fi
    if [[ "${STREAMVGGT_STAGE3_4_TRACE_MEMORY:-1}" == "1" ]]; then
        command+=(--trace-memory)
    else
        command+=(--no-trace-memory)
    fi

    echo "===== Stage 3.4 ${dataset}: ${label} ====="
    "${command[@]}"
}

for dataset in ${parts}; do
    case "${dataset}" in
        bonn|7scenes_loop) ;;
        *) echo "Unknown Stage 3.4 part: ${dataset}" >&2; exit 2 ;;
    esac
    run_eval "${dataset}" full_cache "" ""
    run_eval "${dataset}" stage3_2_k4 4 anchor_recent_dino_diverse_2old_1recent
    run_eval "${dataset}" uniform_k6 6 anchor_recent_uniform
    run_eval "${dataset}" old_dino_k6 6 anchor_recent_dino_diverse
done

python "${repo_root}/scripts/summarize_stage3_4.py" \
    --results-root "${results_root}" \
    --output "${repo_root}/stage3_4_results.csv" \
    --sequence-output "${repo_root}/stage3_4_sequence_results.csv"

