#!/usr/bin/env bash

# Stage 6A: journal same-budget controls and component ablations.
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
python_bin="${CONDA_PREFIX:-}/bin/python"
parts="${STREAMVGGT_STAGE6A_PARTS:-video_depth pose reconstruction finalize}"
same_budget_methods="${STREAMVGGT_STAGE6A_METHODS:-full_cache recent4 anchor_recent4 anchor_uniform4 random4_seed0 random4_seed1 random4_seed2 dino_only4 proposed_k4}"
component_methods="${STREAMVGGT_STAGE6A_COMPONENT_METHODS:-proposed_k6 no_recent_k6}"
tum_sequences="${STREAMVGGT_STAGE6A_TUM_SEQUENCES:-rgbd_dataset_freiburg3_sitting_halfsphere rgbd_dataset_freiburg3_sitting_rpy rgbd_dataset_freiburg3_sitting_static rgbd_dataset_freiburg3_sitting_xyz rgbd_dataset_freiburg3_walking_halfsphere rgbd_dataset_freiburg3_walking_rpy rgbd_dataset_freiburg3_walking_static rgbd_dataset_freiburg3_walking_xyz}"

if [[ -z "${CONDA_PREFIX:-}" || ! -x "${python_bin}" ]]; then
    echo "Stage 6A requires the activated StreamVGGT Conda environment" >&2
    echo "CONDA_DEFAULT_ENV=${CONDA_DEFAULT_ENV:-unset}, CONDA_PREFIX=${CONDA_PREFIX:-unset}" >&2
    exit 2
fi
"${python_bin}" -c 'import numpy, torch, sys; print("Stage 6A Python:", sys.executable, "NumPy:", numpy.__version__, "Torch:", torch.__version__, "GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none")'

method_config() {
    local method="$1"
    case "${method}" in
        full_cache)       METHOD_WINDOW=""; METHOD_POLICY=""; METHOD_SEED=0 ;;
        recent4)          METHOD_WINDOW=4; METHOD_POLICY=fifo; METHOD_SEED=0 ;;
        anchor_recent4)   METHOD_WINDOW=4; METHOD_POLICY=anchor_recent; METHOD_SEED=0 ;;
        anchor_uniform4)  METHOD_WINDOW=4; METHOD_POLICY=anchor_uniform_k4; METHOD_SEED=0 ;;
        random4_seed0)    METHOD_WINDOW=4; METHOD_POLICY=random_reservoir_k4; METHOD_SEED=0 ;;
        random4_seed1)    METHOD_WINDOW=4; METHOD_POLICY=random_reservoir_k4; METHOD_SEED=1 ;;
        random4_seed2)    METHOD_WINDOW=4; METHOD_POLICY=random_reservoir_k4; METHOD_SEED=2 ;;
        dino_only4)       METHOD_WINDOW=4; METHOD_POLICY=dino_diverse_no_anchor_k4; METHOD_SEED=0 ;;
        proposed_k4)      METHOD_WINDOW=4; METHOD_POLICY=anchor_recent_dino_diverse_k4; METHOD_SEED=0 ;;
        proposed_k6)      METHOD_WINDOW=6; METHOD_POLICY=anchor_recent_dino_diverse_k6; METHOD_SEED=0 ;;
        no_recent_k6)     METHOD_WINDOW=6; METHOD_POLICY=anchor_dino_diverse_no_recent_k6; METHOD_SEED=0 ;;
        *) echo "Unknown Stage 6A method: ${method}" >&2; exit 2 ;;
    esac
}

with_cache_env() {
    local method="$1"
    shift
    method_config "${method}"
    export STREAMVGGT_CACHE_RANDOM_SEED="${METHOD_SEED}"
    if [[ -n "${METHOD_WINDOW}" ]]; then
        export STREAMVGGT_CACHE_WINDOW="${METHOD_WINDOW}"
        export STREAMVGGT_CACHE_POLICY="${METHOD_POLICY}"
    else
        unset STREAMVGGT_CACHE_WINDOW STREAMVGGT_CACHE_POLICY
    fi
    "$@"
}

run_video_depth_one() {
    local method="$1"
    local datasets="$2"
    echo "===== Stage 6A VideoDepth: ${method} ====="
    (
        cd "${repo_root}/src"
        export STREAMVGGT_EVAL_DATASETS="${datasets}"
        export STREAMVGGT_RUN_TAG="stage6a_${method}"
        export STREAMVGGT_MAX_FRAMES="${STREAMVGGT_STAGE6A_MAX_FRAMES:-}"
        export STREAMVGGT_LOG_SELECTIONS="${STREAMVGGT_STAGE6A_LOG_SELECTIONS:-1}"
        export STREAMVGGT_VIDEO_DEPTH_OUTPUT_ROOT="${STREAMVGGT_STAGE6A_VIDEO_RESULTS_ROOT:-${repo_root}/eval_results/video_depth}"
        with_cache_env "${method}" bash eval/video_depth/run.sh
    )
}

run_pose_one() {
    local method="$1"
    echo "===== Stage 6A TUM pose: ${method} ====="
    (
        cd "${repo_root}/src"
        export STREAMVGGT_POSE_DATASETS=tum
        export STREAMVGGT_POSE_SEQ_LIST="${tum_sequences}"
        export STREAMVGGT_POSE_MAX_FRAMES="${STREAMVGGT_STAGE6A_POSE_MAX_FRAMES:-}"
        export STREAMVGGT_LOG_SELECTIONS="${STREAMVGGT_STAGE6A_LOG_SELECTIONS:-1}"
        export STREAMVGGT_RUN_TAG="stage6a_${method}"
        with_cache_env "${method}" bash eval/pose_evaluation/run_streaming_pose.sh
    )
}

run_reconstruction_one() {
    local method="$1"
    echo "===== Stage 6A TUM Dynamics reconstruction: ${method} ====="
    (
        cd "${repo_root}/src"
        export STREAMVGGT_MV_DATASETS=tum
        export STREAMVGGT_MV_PROTOCOL=paper
        export STREAMVGGT_MV_TUM_FRAMES="${STREAMVGGT_STAGE6A_RECON_FRAMES:-50}"
        export STREAMVGGT_MV_TUM_SAMPLING=first
        export STREAMVGGT_MV_PREFIX_FRAMES="${STREAMVGGT_STAGE6A_RECON_PREFIX_FRAMES:-}"
        export STREAMVGGT_MV_TUM_SEQ_LIST="${tum_sequences}"
        export STREAMVGGT_MV_SAVE_ARTIFACTS="${STREAMVGGT_STAGE6A_SAVE_ARTIFACTS:-0}"
        export STREAMVGGT_MV_LOG_SELECTIONS="${STREAMVGGT_STAGE6A_LOG_SELECTIONS:-1}"
        export STREAMVGGT_MV_RUN_TAG="stage6a_${method}"
        with_cache_env "${method}" bash eval/mv_recon/run_streaming_recon.sh
    )
}

all_methods="${same_budget_methods} ${component_methods}"
for part in ${parts}; do
    case "${part}" in
        video_depth)
            for method in ${same_budget_methods}; do
                run_video_depth_one "${method}" "${STREAMVGGT_STAGE6A_VIDEO_DATASETS:-bonn sintel kitti}"
            done
            for method in ${component_methods}; do
                run_video_depth_one "${method}" "${STREAMVGGT_STAGE6A_COMPONENT_VIDEO_DATASETS:-kitti}"
            done
            ;;
        pose)
            for method in ${all_methods}; do
                run_pose_one "${method}"
            done
            ;;
        reconstruction)
            for method in ${all_methods}; do
                run_reconstruction_one "${method}"
            done
            ;;
        finalize)
            finalize_args=(
                --repo-root "${repo_root}"
                --bootstrap-samples "${STREAMVGGT_STAGE6A_BOOTSTRAP_SAMPLES:-10000}"
            )
            if [[ "${STREAMVGGT_STAGE6A_ALLOW_INCOMPLETE:-0}" == "1" ]]; then
                finalize_args+=(--allow-incomplete)
            fi
            "${python_bin}" "${repo_root}/scripts/summarize_stage6a.py" "${finalize_args[@]}"
            ;;
        *) echo "Unknown Stage 6A part: ${part}" >&2; exit 2 ;;
    esac
done
