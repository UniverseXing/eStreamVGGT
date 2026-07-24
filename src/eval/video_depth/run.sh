#!/bin/bash

set -e

workdir='..'
model_name='streamvggt'
ckpt_name='checkpoints'
model_weights="${workdir}/ckpt/${ckpt_name}.pth"
results_root="${STREAMVGGT_VIDEO_DEPTH_OUTPUT_ROOT:-${workdir}/eval_results/video_depth}"
read -r -a datasets <<< "${STREAMVGGT_EVAL_DATASETS:-sintel bonn kitti}"

cache_suffix=""
if [[ -n "${STREAMVGGT_CACHE_WINDOW:-}" ]]; then
    cache_policy="${STREAMVGGT_CACHE_POLICY:-fifo}"
    cache_suffix="_${cache_policy}_k${STREAMVGGT_CACHE_WINDOW}"
fi
if [[ -n "${STREAMVGGT_RUN_TAG:-}" ]]; then
    cache_suffix="${cache_suffix}_${STREAMVGGT_RUN_TAG}"
fi
if [[ -n "${STREAMVGGT_MAX_FRAMES:-}" ]]; then
    cache_suffix="${cache_suffix}_n${STREAMVGGT_MAX_FRAMES}"
fi

sequence_args=()
if [[ -n "${STREAMVGGT_SEQ_LIST:-}" ]]; then
    read -r -a seq_list <<< "${STREAMVGGT_SEQ_LIST}"
    sequence_args+=(--seq_list "${seq_list[@]}")
fi
if [[ -n "${STREAMVGGT_MAX_FRAMES:-}" ]]; then
    sequence_args+=(--max_frames "${STREAMVGGT_MAX_FRAMES}")
fi

for data in "${datasets[@]}"; do
    output_dir="${results_root}/${data}_${model_name}${cache_suffix}"
    echo "$output_dir"
    CUDA_LAUNCH_BLOCKING=1 accelerate launch --num_processes 1  ../src/eval/video_depth/launch.py \
        --weights "$model_weights" \
        --output_dir "$output_dir" \
        --eval_dataset "$data" \
        --size 518 \
        "${sequence_args[@]}"
    python ../src/eval/video_depth/eval_depth.py \
    --output_dir "$output_dir" \
    --eval_dataset "$data" \
    --align "scale" \
    "${sequence_args[@]}"
done
