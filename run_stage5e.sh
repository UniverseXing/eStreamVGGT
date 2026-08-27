#!/usr/bin/env bash

# Emergency Stage 5E: K4 versus official OVGGT.
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
project_python="${CONDA_PREFIX:-}/bin/python"
ovggt_root="${STREAMVGGT_STAGE5E_OVGGT_ROOT:-${repo_root}/external/OVGGT}"
weights="${STREAMVGGT_STAGE5E_WEIGHTS:-${repo_root}/ckpt/checkpoints.pth}"
output_root="${STREAMVGGT_STAGE5E_OUTPUT_ROOT:-${repo_root}/eval_results/stage5e_ovggt}"
datasets="${STREAMVGGT_STAGE5E_DATASETS:-bonn sintel kitti}"
parts="${STREAMVGGT_STAGE5E_PARTS:-parity inference finalize}"
expected_ovggt_commit="${STREAMVGGT_STAGE5E_OVGGT_COMMIT:-b582391f3dc6ec734aaa3a8fde3b4baadaf7800a}"

if [[ ! -x "${project_python}" ]]; then
    echo "Stage 5E requires the activated StreamVGGT environment" >&2
    exit 2
fi
if [[ ! -f "${ovggt_root}/src/ovggt/models/ovggt.py" ]]; then
    echo "Missing official OVGGT checkout: ${ovggt_root}" >&2
    echo "Run the Stage 5E setup commands in STAGE5_CONFERENCE_EXPERIMENT_PLAN.md first." >&2
    exit 2
fi
if [[ ! -f "${weights}" ]]; then
    echo "Missing StreamVGGT weights: ${weights}" >&2
    exit 2
fi
actual_ovggt_commit="$(git -C "${ovggt_root}" rev-parse HEAD)"
if [[ "${actual_ovggt_commit}" != "${expected_ovggt_commit}" ]]; then
    echo "OVGGT commit mismatch: ${actual_ovggt_commit} != ${expected_ovggt_commit}" >&2
    echo "Run: git -C ${ovggt_root} checkout ${expected_ovggt_commit}" >&2
    exit 2
fi

"${project_python}" -c 'import torch, sys; assert torch.cuda.is_available(); print("OVGGT Python:", sys.executable, "Torch:", torch.__version__, "CUDA:", torch.version.cuda, "GPU:", torch.cuda.get_device_name(0))'

eval_predictions() {
    local dataset="$1"
    local directory="$2"
    shift 2
    (
        cd "${repo_root}/src"
        "${project_python}" eval/video_depth/eval_depth.py \
            --output_dir "${directory}" \
            --eval_dataset "${dataset}" \
            --align scale "$@"
    )
}

for part in ${parts}; do
    case "${part}" in
        parity)
            echo "===== Stage 5E parity: project Full, Bonn balloon2/10 ====="
            (
                cd "${repo_root}/src"
                export STREAMVGGT_VIDEO_DEPTH_OUTPUT_ROOT="${output_root}/parity_ours"
                export STREAMVGGT_EVAL_DATASETS=bonn
                export STREAMVGGT_SEQ_LIST=balloon2
                export STREAMVGGT_MAX_FRAMES=10
                export STREAMVGGT_RUN_TAG=stage5e_parity_ours_full
                unset STREAMVGGT_CACHE_WINDOW STREAMVGGT_CACHE_POLICY
                bash eval/video_depth/run.sh
            )
            echo "===== Stage 5E parity: OVGGT implementation Full, Bonn balloon2/10 ====="
            "${project_python}" "${repo_root}/scripts/stage5e_ovggt_video_depth.py" \
                --repo-root "${repo_root}" --ovggt-root "${ovggt_root}" \
                --weights "${weights}" --dataset bonn --seq-list balloon2 \
                --max-frames 10 --mode full \
                --output-dir "${output_root}/parity_ovggt/bonn"
            eval_predictions bonn "${output_root}/parity_ovggt/bonn" \
                --seq_list balloon2 --max_frames 10
            "${project_python}" "${repo_root}/scripts/summarize_stage5e.py" \
                --repo-root "${repo_root}" --output-root "${output_root}" \
                --parity-only
            ;;
        inference)
            for dataset in ${datasets}; do
                echo "===== Stage 5E OVGGT: ${dataset} ====="
                "${project_python}" "${repo_root}/scripts/stage5e_ovggt_video_depth.py" \
                    --repo-root "${repo_root}" --ovggt-root "${ovggt_root}" \
                    --weights "${weights}" --dataset "${dataset}" \
                    --mode ovggt \
                    --output-dir "${output_root}/${dataset}"
                eval_predictions "${dataset}" "${output_root}/${dataset}"
            done
            ;;
        finalize)
            read -r -a dataset_array <<< "${datasets}"
            "${project_python}" "${repo_root}/scripts/summarize_stage5e.py" \
                --repo-root "${repo_root}" --output-root "${output_root}" \
                --datasets "${dataset_array[@]}"
            ;;
        *)
            echo "Unknown Stage 5E part: ${part}" >&2
            exit 2
            ;;
    esac
done
