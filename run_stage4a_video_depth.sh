#!/usr/bin/env bash

# Stage 4A: KITTI outdoor VideoDepth plus temporal-K8 Bonn/Sintel completion.
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
python_bin="${CONDA_PREFIX:-}/bin/python"
methods="${STREAMVGGT_STAGE4A_METHODS:-stage3_2_k4 old_dino_k6 temporal_binned_dino_k8 full_cache}"
formal_results_root="${STREAMVGGT_STAGE4A_RESULTS_ROOT:-${repo_root}/eval_results/video_depth}"
kitti_root="${STREAMVGGT_KITTI_ROOT:-${repo_root}/data/eval/kitti}"
if [[ "${formal_results_root}" != /* ]]; then
    formal_results_root="${repo_root}/${formal_results_root}"
fi
if [[ "${kitti_root}" != /* ]]; then
    kitti_root="${repo_root}/${kitti_root}"
fi

if [[ -z "${CONDA_PREFIX:-}" || ! -x "${python_bin}" ]]; then
    echo "Stage 4A requires the activated StreamVGGT Conda environment" >&2
    echo "CONDA_DEFAULT_ENV=${CONDA_DEFAULT_ENV:-unset}, CONDA_PREFIX=${CONDA_PREFIX:-unset}" >&2
    exit 2
fi
"${python_bin}" -c 'import numpy, torch, sys; print("Stage 4A Python:", sys.executable, "NumPy:", numpy.__version__, "Torch:", torch.__version__)'

"${python_bin}" "${repo_root}/scripts/check_stage4a_kitti.py" \
    --root "${kitti_root}"

run_method() {
    local method="$1"
    local datasets window policy
    case "${method}" in
        full_cache)
            datasets="kitti"
            window=""
            policy=""
            ;;
        stage3_2_k4)
            datasets="kitti"
            window="4"
            policy="anchor_recent_dino_diverse_2old_1recent"
            ;;
        old_dino_k6)
            datasets="kitti"
            window="6"
            policy="anchor_recent_dino_diverse"
            ;;
        temporal_binned_dino_k8)
            datasets="${STREAMVGGT_STAGE4A_K8_DATASETS:-kitti bonn sintel}"
            window="8"
            policy="temporal_binned_dino_k8"
            ;;
        *)
            echo "Unknown Stage 4A method: ${method}" >&2
            exit 2
            ;;
    esac

    echo "===== Stage 4A ${method}: ${datasets} ====="
    (
        cd "${repo_root}/src"
        export STREAMVGGT_EVAL_DATASETS="${datasets}"
        export STREAMVGGT_RUN_TAG="stage4a_${method}"
        export STREAMVGGT_MAX_FRAMES="${STREAMVGGT_STAGE4A_MAX_FRAMES:-}"
        export STREAMVGGT_VIDEO_DEPTH_OUTPUT_ROOT="${formal_results_root}"
        export STREAMVGGT_KITTI_ROOT="${kitti_root}"
        if [[ -n "${window}" ]]; then
            export STREAMVGGT_CACHE_WINDOW="${window}"
            export STREAMVGGT_CACHE_POLICY="${policy}"
        else
            unset STREAMVGGT_CACHE_WINDOW STREAMVGGT_CACHE_POLICY
        fi
        bash eval/video_depth/run.sh
    )
}

for method in ${methods}; do
    run_method "${method}"
done

if [[ "${STREAMVGGT_STAGE4A_SKIP_FINALIZE:-0}" == "1" ]]; then
    echo "Skipping Stage 4A summary/gate as requested."
    exit 0
fi

"${python_bin}" "${repo_root}/scripts/summarize_stage4a_video_depth.py" \
    --results-root "${formal_results_root}" \
    --baselines "${repo_root}/stage4a_video_depth_baselines.json" \
    --output "${repo_root}/stage4a_video_depth_results.csv"
"${python_bin}" "${repo_root}/scripts/check_stage4a_gate.py" \
    --input "${repo_root}/stage4a_video_depth_results.csv" \
    --output "${repo_root}/stage4a_gate.csv"
