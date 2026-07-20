#!/bin/bash
set -e

workdir='..'
model_name='StreamVGGT'
ckpt_name='checkpoints'
model_weights="${workdir}/ckpt/${ckpt_name}.pth"
read -r -a datasets <<< "${STREAMVGGT_EVAL_DATASETS:-sintel bonn kitti nyu}"

cache_suffix=""
if [[ -n "${STREAMVGGT_CACHE_WINDOW:-}" ]]; then
    cache_policy="${STREAMVGGT_CACHE_POLICY:-fifo}"
    cache_suffix="_${cache_policy}_k${STREAMVGGT_CACHE_WINDOW}"
fi

for data in "${datasets[@]}"; do
    output_dir="${workdir}/eval_results/monodepth/${data}_${model_name}${cache_suffix}"
    echo "$output_dir"
    CUDA_LAUNCH_BLOCKING=1 python ./eval/monodepth/launch.py \
        --weights "$model_weights" \
        --output_dir "$output_dir" \
        --eval_dataset "$data"
done

for data in "${datasets[@]}"; do
    output_dir="${workdir}/eval_results/monodepth/${data}_${model_name}${cache_suffix}"
    CUDA_LAUNCH_BLOCKING=1 python ./eval/monodepth/eval_metrics.py \
        --output_dir "$output_dir" \
        --eval_dataset "$data"
done
