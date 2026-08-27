#!/usr/bin/env bash

# Emergency Stage 5E: K4 versus official StreamVGGT-STAC.
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
project_python="${CONDA_PREFIX:-}/bin/python"
stac_root="${STREAMVGGT_STAGE5E_STAC_ROOT:-${repo_root}/../STAC}"
stac_env="${STREAMVGGT_STAGE5E_STAC_ENV:-stac}"
weights="${STREAMVGGT_STAGE5E_WEIGHTS:-${repo_root}/ckpt/checkpoints.pth}"
output_root="${STREAMVGGT_STAGE5E_OUTPUT_ROOT:-${repo_root}/eval_results/stage5e_stac}"
datasets="${STREAMVGGT_STAGE5E_DATASETS:-bonn sintel kitti}"
parts="${STREAMVGGT_STAGE5E_PARTS:-parity inference finalize}"
backend="${STREAMVGGT_STAGE5E_BACKEND:-cuda}"
expected_stac_commit="${STREAMVGGT_STAGE5E_STAC_COMMIT:-fd7e718597cf9963de85c8fffae32a698e8619f5}"

if [[ ! -x "${project_python}" ]]; then
    echo "Stage 5E requires the activated StreamVGGT environment" >&2
    exit 2
fi
if [[ ! -f "${stac_root}/model_wrapper.py" ]]; then
    echo "Missing official STAC checkout: ${stac_root}" >&2
    echo "Run the Stage 5E setup commands in STAGE5_CONFERENCE_EXPERIMENT_PLAN.md first." >&2
    exit 2
fi
if [[ ! -f "${weights}" ]]; then
    echo "Missing StreamVGGT weights: ${weights}" >&2
    exit 2
fi
actual_stac_commit="$(git -C "${stac_root}" rev-parse HEAD)"
if [[ "${actual_stac_commit}" != "${expected_stac_commit}" ]]; then
    echo "STAC commit mismatch: ${actual_stac_commit} != ${expected_stac_commit}" >&2
    echo "Run: git -C ${stac_root} checkout ${expected_stac_commit}" >&2
    exit 2
fi

stac_python=(conda run --no-capture-output -n "${stac_env}" python)
"${stac_python[@]}" -c 'import torch, sys; assert torch.cuda.is_available(); print("STAC Python:", sys.executable, "Torch:", torch.__version__, "CUDA:", torch.version.cuda, "GPU:", torch.cuda.get_device_name(0))'

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
            echo "===== Stage 5E parity: STAC adapter Full, Bonn balloon2/10 ====="
            "${stac_python[@]}" "${repo_root}/scripts/stage5e_stac_video_depth.py" \
                --repo-root "${repo_root}" --stac-root "${stac_root}" \
                --weights "${weights}" --dataset bonn --seq-list balloon2 \
                --max-frames 10 --mode full --backend "${backend}" \
                --output-dir "${output_root}/parity_stac/bonn"
            eval_predictions bonn "${output_root}/parity_stac/bonn" \
                --seq_list balloon2 --max_frames 10
            ;;
        inference)
            for dataset in ${datasets}; do
                echo "===== Stage 5E StreamVGGT-STAC: ${dataset} ====="
                "${stac_python[@]}" "${repo_root}/scripts/stage5e_stac_video_depth.py" \
                    --repo-root "${repo_root}" --stac-root "${stac_root}" \
                    --weights "${weights}" --dataset "${dataset}" \
                    --mode stac --backend "${backend}" \
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
