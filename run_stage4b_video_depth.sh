#!/usr/bin/env bash

# Stage 4B: per-sequence VideoDepth evaluation and paired statistics.
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
python_bin="${CONDA_PREFIX:-}/bin/python"
results_root="${STREAMVGGT_STAGE4B_RESULTS_ROOT:-${repo_root}/eval_results/video_depth}"
kitti_root="${STREAMVGGT_KITTI_ROOT:-${repo_root}/data/eval/kitti}"
datasets="${STREAMVGGT_STAGE4B_DATASETS:-bonn kitti sintel}"
methods="${STREAMVGGT_STAGE4B_METHODS:-full_cache stage3_2_k4 old_dino_k6 temporal_binned_dino_k8}"

if [[ "${results_root}" != /* ]]; then
    results_root="${repo_root}/${results_root}"
fi
if [[ "${kitti_root}" != /* ]]; then
    kitti_root="${repo_root}/${kitti_root}"
fi
if [[ -z "${CONDA_PREFIX:-}" || ! -x "${python_bin}" ]]; then
    echo "Stage 4B requires the activated StreamVGGT Conda environment" >&2
    echo "CONDA_DEFAULT_ENV=${CONDA_DEFAULT_ENV:-unset}, CONDA_PREFIX=${CONDA_PREFIX:-unset}" >&2
    exit 2
fi

"${python_bin}" -c 'import cv2, numpy, torch, sys; print("Stage 4B Python:", sys.executable, "NumPy:", numpy.__version__, "Torch:", torch.__version__)'

result_directory() {
    local dataset="$1"
    local method="$2"
    case "${method}" in
        full_cache)
            echo "${results_root}/${dataset}_streamvggt_stage4a_full_cache"
            ;;
        stage3_2_k4)
            echo "${results_root}/${dataset}_streamvggt_anchor_recent_dino_diverse_2old_1recent_k4_stage4a_stage3_2_k4"
            ;;
        old_dino_k6)
            echo "${results_root}/${dataset}_streamvggt_anchor_recent_dino_diverse_k6_stage4a_old_dino_k6"
            ;;
        temporal_binned_dino_k8)
            echo "${results_root}/${dataset}_streamvggt_temporal_binned_dino_k8_k8_stage4a_temporal_binned_dino_k8"
            ;;
        *)
            echo "Unknown Stage 4B method: ${method}" >&2
            return 2
            ;;
    esac
}

if [[ "${STREAMVGGT_STAGE4B_SKIP_EVAL:-0}" != "1" ]]; then
    for dataset in ${datasets}; do
        for method in ${methods}; do
            output_dir="$(result_directory "${dataset}" "${method}")"
            if [[ ! -f "${output_dir}/runtime_memory_rank0.json" ]]; then
                echo "Missing frozen Stage 4A runtime result: ${output_dir}" >&2
                exit 2
            fi
            echo "===== Stage 4B eval ${dataset}/${method} ====="
            (
                cd "${repo_root}/src"
                export STREAMVGGT_KITTI_ROOT="${kitti_root}"
                "${python_bin}" ../src/eval/video_depth/eval_depth.py \
                    --output_dir "${output_dir}" \
                    --eval_dataset "${dataset}" \
                    --align scale
            )
        done
    done
fi

if [[ "${STREAMVGGT_STAGE4B_SKIP_FINALIZE:-0}" == "1" ]]; then
    echo "Skipping Stage 4B statistics as requested."
    exit 0
fi

"${python_bin}" "${repo_root}/scripts/summarize_stage4b_video_depth.py" \
    --results-root "${results_root}" \
    --sequence-output "${repo_root}/stage4b_video_depth_sequence_results.csv" \
    --statistics-output "${repo_root}/stage4b_video_depth_statistics.csv" \
    --paired-output "${repo_root}/stage4b_video_depth_paired_comparison.csv" \
    --regret-output "${repo_root}/stage4b_video_depth_regret.csv" \
    --pareto-output "${repo_root}/stage4b_pareto.csv" \
    --bootstrap-samples 10000 \
    --seed 0
