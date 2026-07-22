#!/usr/bin/env bash

# One cache-aware reconstruction run. Invoke from the repository's src/.
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
weights="${STREAMVGGT_WEIGHTS:-${repo_root}/ckpt/checkpoints.pth}"
datasets="${STREAMVGGT_MV_DATASETS:-7scenes nrgbd eth3d}"
run_tag="${STREAMVGGT_MV_RUN_TAG:-manual}"
output_dir="${STREAMVGGT_MV_OUTPUT_DIR:-${repo_root}/eval_results/mv_recon/streamvggt_${run_tag}}"

read -r -a dataset_args <<< "${datasets}"
args=(
    eval/mv_recon/launch.py
    --weights "${weights}"
    --model_name StreamVGGT
    --output_dir "${output_dir}"
    --size "${STREAMVGGT_MV_SIZE:-518}"
    --seed "${STREAMVGGT_MV_SEED:-0}"
    --protocol "${STREAMVGGT_MV_PROTOCOL:-paper}"
    --seven-scenes-kf-every "${STREAMVGGT_MV_7SCENES_KF_EVERY:-50}"
    --nrgbd-kf-every "${STREAMVGGT_MV_NRGBD_KF_EVERY:-100}"
    --tum-frames "${STREAMVGGT_MV_TUM_FRAMES:-50}"
    --tum-sampling "${STREAMVGGT_MV_TUM_SAMPLING:-first}"
    --datasets "${dataset_args[@]}"
)

if [[ -n "${STREAMVGGT_CACHE_WINDOW:-}" ]]; then
    args+=(--cache-window "${STREAMVGGT_CACHE_WINDOW}")
    args+=(--cache-policy "${STREAMVGGT_CACHE_POLICY:-fifo}")
fi
if [[ -n "${STREAMVGGT_MV_MAX_SCENES:-}" ]]; then
    args+=(--max-scenes "${STREAMVGGT_MV_MAX_SCENES}")
fi
if [[ -n "${STREAMVGGT_MV_MAX_FRAMES:-}" ]]; then
    args+=(--max-frames "${STREAMVGGT_MV_MAX_FRAMES}")
fi
if [[ -n "${STREAMVGGT_MV_PREFIX_FRAMES:-}" ]]; then
    read -r -a prefix_args <<< "${STREAMVGGT_MV_PREFIX_FRAMES}"
    args+=(--prefix-frames "${prefix_args[@]}")
fi
if [[ -n "${STREAMVGGT_MV_SEQ_LIST:-}" ]]; then
    read -r -a sequence_args <<< "${STREAMVGGT_MV_SEQ_LIST}"
    args+=(--seq-list "${sequence_args[@]}")
fi
for dataset_name in 7scenes nrgbd eth3d tum; do
    env_suffix="${dataset_name^^}"
    env_suffix="${env_suffix//-/_}"
    env_name="STREAMVGGT_MV_${env_suffix}_SEQ_LIST"
    if [[ -n "${!env_name:-}" ]]; then
        args+=(--dataset-seq-list "${dataset_name}=${!env_name}")
    fi
done
if [[ -n "${STREAMVGGT_MV_7SCENES_ROOT:-}" ]]; then
    args+=(--data-root "7scenes=${STREAMVGGT_MV_7SCENES_ROOT}")
fi
if [[ -n "${STREAMVGGT_MV_NRGBD_ROOT:-}" ]]; then
    args+=(--data-root "nrgbd=${STREAMVGGT_MV_NRGBD_ROOT}")
fi
if [[ -n "${STREAMVGGT_MV_ETH3D_ROOT:-}" ]]; then
    args+=(--data-root "eth3d=${STREAMVGGT_MV_ETH3D_ROOT}")
fi
if [[ -n "${STREAMVGGT_MV_TUM_ROOT:-}" ]]; then
    args+=(--data-root "tum=${STREAMVGGT_MV_TUM_ROOT}")
fi
if [[ "${STREAMVGGT_MV_LOG_SELECTIONS:-1}" == "1" ]]; then
    args+=(--log-selections)
fi
if [[ "${STREAMVGGT_MV_USE_PROJ:-0}" == "1" ]]; then
    args+=(--use_proj)
fi
if [[ "${STREAMVGGT_MV_SAVE_ARTIFACTS:-1}" == "0" ]]; then
    args+=(--no-save-artifacts)
fi

echo "Reconstruction datasets: ${datasets}"
echo "Protocol: ${STREAMVGGT_MV_PROTOCOL:-paper}"
echo "Output: ${output_dir}"
accelerate launch --num_processes 1 "${args[@]}"
