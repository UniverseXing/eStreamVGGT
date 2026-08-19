#!/usr/bin/env bash

set -euo pipefail

workdir=".."
weights="${workdir}/ckpt/checkpoints.pth"
read -r -a datasets <<< "${STREAMVGGT_POSE_DATASETS:-sintel}"

cache_suffix="full_cache"
cache_args=()
if [[ -n "${STREAMVGGT_CACHE_WINDOW:-}" ]]; then
    cache_policy="${STREAMVGGT_CACHE_POLICY:-fifo}"
    cache_suffix="${cache_policy}_k${STREAMVGGT_CACHE_WINDOW}"
    cache_args+=(--cache-window "${STREAMVGGT_CACHE_WINDOW}" --cache-policy "${cache_policy}")
fi
cache_args+=(--cache-random-seed "${STREAMVGGT_CACHE_RANDOM_SEED:-0}")
if [[ -n "${STREAMVGGT_RUN_TAG:-}" ]]; then
    cache_suffix="${cache_suffix}_${STREAMVGGT_RUN_TAG}"
fi

sequence_args=()
if [[ -n "${STREAMVGGT_POSE_SEQ_LIST:-}" ]]; then
    read -r -a seq_list <<< "${STREAMVGGT_POSE_SEQ_LIST}"
    sequence_args+=(--seq-list "${seq_list[@]}")
fi
if [[ -n "${STREAMVGGT_POSE_MAX_FRAMES:-}" ]]; then
    sequence_args+=(--max-frames "${STREAMVGGT_POSE_MAX_FRAMES}")
fi
if [[ -n "${STREAMVGGT_POSE_STRIDE:-}" ]]; then
    sequence_args+=(--stride "${STREAMVGGT_POSE_STRIDE}")
fi
if [[ "${STREAMVGGT_LOG_SELECTIONS:-0}" == "1" ]]; then
    sequence_args+=(--log-selections)
fi
if [[ "${STREAMVGGT_POSE_RESUME:-0}" == "1" ]]; then
    sequence_args+=(--resume)
fi

for dataset in "${datasets[@]}"; do
    output_dir="${workdir}/eval_results/pose/${dataset}_streamvggt_${cache_suffix}"
    extra_roots=()
    root_var="STREAMVGGT_POSE_${dataset^^}_ROOT"
    if [[ -n "${!root_var:-}" ]]; then
        extra_roots+=(--data-root "${!root_var}")
    fi
    echo "${output_dir}"
    accelerate launch --num_processes 1 ../src/eval/pose_evaluation/eval_streaming_pose.py \
        --weights "${weights}" \
        --dataset "${dataset}" \
        --output-dir "${output_dir}" \
        --size 518 \
        "${cache_args[@]}" \
        "${sequence_args[@]}" \
        "${extra_roots[@]}"
done
