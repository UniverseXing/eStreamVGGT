#!/usr/bin/env bash

# Stage 3.7: incremental Stage 3.3 backtest for temporal-binned DINO K8 only.
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
parts="${STREAMVGGT_STAGE3_7_PARTS:-pose static dynamic}"
seven_scenes_successful="${STREAMVGGT_STAGE3_7_7SCENES_SEQUENCES:-chess/seq-03 chess/seq-05 fire/seq-03 fire/seq-04 heads/seq-01 office/seq-02 pumpkin/seq-01 pumpkin/seq-07 redkitchen/seq-03 redkitchen/seq-04 stairs/seq-01 stairs/seq-04}"

if [[ -z "${CONDA_PREFIX:-}" || ! -x "${CONDA_PREFIX}/bin/python" ]]; then
    echo "Stage 3.7 requires an activated Conda environment with CONDA_PREFIX/bin/python" >&2
    echo "CONDA_DEFAULT_ENV=${CONDA_DEFAULT_ENV:-unset}, CONDA_PREFIX=${CONDA_PREFIX:-unset}" >&2
    exit 2
fi
python_bin="${CONDA_PREFIX}/bin/python"
"${python_bin}" -c 'import numpy, torch, sys; print("Stage 3.7 Python:", sys.executable, "NumPy:", numpy.__version__, "Torch:", torch.__version__)'

run_pose() {
    echo "===== Stage 3.7A: temporal-binned DINO K8 pose backtest ====="
    (
        cd "${repo_root}/src"
        STREAMVGGT_CACHE_WINDOW=8 \
        STREAMVGGT_CACHE_POLICY=temporal_binned_dino_k8 \
        STREAMVGGT_POSE_DATASETS="${STREAMVGGT_STAGE3_7_POSE_DATASETS:-sintel scannet tum}" \
        STREAMVGGT_POSE_MAX_FRAMES="${STREAMVGGT_STAGE3_7_POSE_MAX_FRAMES:-}" \
        STREAMVGGT_POSE_RESUME="${STREAMVGGT_STAGE3_7_POSE_RESUME:-0}" \
        STREAMVGGT_LOG_SELECTIONS="${STREAMVGGT_STAGE3_7_LOG_SELECTIONS:-1}" \
        STREAMVGGT_RUN_TAG=stage3_7_temporal_binned_dino_k8 \
        bash eval/pose_evaluation/run_streaming_pose.sh
    )
    "${python_bin}" "${repo_root}/scripts/summarize_stage3_3_pose.py" \
        --results-root "${repo_root}/eval_results/pose" \
        --name-filter stage3_7_temporal_binned_dino_k8 \
        --output "${repo_root}/stage3_7_pose_results.csv"
}

run_static_reconstruction() {
    echo "===== Stage 3.7B: temporal-binned DINO K8 static reconstruction ====="
    (
        cd "${repo_root}/src"
        STREAMVGGT_CACHE_WINDOW=8 \
        STREAMVGGT_CACHE_POLICY=temporal_binned_dino_k8 \
        STREAMVGGT_MV_DATASETS="${STREAMVGGT_STAGE3_7_STATIC_DATASETS:-7scenes nrgbd eth3d}" \
        STREAMVGGT_MV_PROTOCOL=dense \
        STREAMVGGT_MV_PREFIX_FRAMES="${STREAMVGGT_STAGE3_7_STATIC_PREFIX_FRAMES:-4 6 8 10}" \
        STREAMVGGT_MV_7SCENES_SEQ_LIST="${seven_scenes_successful}" \
        STREAMVGGT_MV_MAX_SCENES="${STREAMVGGT_STAGE3_7_STATIC_MAX_SCENES:-}" \
        STREAMVGGT_MV_MAX_FRAMES="${STREAMVGGT_STAGE3_7_STATIC_MAX_FRAMES:-}" \
        STREAMVGGT_MV_SAVE_ARTIFACTS="${STREAMVGGT_STAGE3_7_SAVE_ARTIFACTS:-0}" \
        STREAMVGGT_MV_RUN_TAG=stage3_7b_temporal_binned_dino_k8 \
        bash eval/mv_recon/run_streaming_recon.sh
    )
    "${python_bin}" "${repo_root}/scripts/summarize_stage3_3b_recon.py" \
        --results-root "${repo_root}/eval_results/mv_recon" \
        --name-filter stage3_7b \
        --output "${repo_root}/stage3_7b_recon_results.csv"
}

run_dynamic_reconstruction() {
    echo "===== Stage 3.7C: temporal-binned DINO K8 TUM-dynamics reconstruction ====="
    (
        cd "${repo_root}/src"
        STREAMVGGT_CACHE_WINDOW=8 \
        STREAMVGGT_CACHE_POLICY=temporal_binned_dino_k8 \
        STREAMVGGT_MV_DATASETS=tum \
        STREAMVGGT_MV_PROTOCOL=paper \
        STREAMVGGT_MV_TUM_FRAMES="${STREAMVGGT_STAGE3_7_TUM_FRAMES:-50}" \
        STREAMVGGT_MV_TUM_SAMPLING="${STREAMVGGT_STAGE3_7_TUM_SAMPLING:-first}" \
        STREAMVGGT_MV_PREFIX_FRAMES="${STREAMVGGT_STAGE3_7_TUM_PREFIX_FRAMES:-10 20 30 40 50}" \
        STREAMVGGT_MV_TUM_SEQ_LIST="${STREAMVGGT_STAGE3_7_TUM_SEQUENCES:-}" \
        STREAMVGGT_MV_MAX_SCENES="${STREAMVGGT_STAGE3_7_TUM_MAX_SCENES:-}" \
        STREAMVGGT_MV_MAX_FRAMES="${STREAMVGGT_STAGE3_7_TUM_MAX_FRAMES:-}" \
        STREAMVGGT_MV_SAVE_ARTIFACTS="${STREAMVGGT_STAGE3_7_SAVE_ARTIFACTS:-0}" \
        STREAMVGGT_MV_RUN_TAG=stage3_7c_temporal_binned_dino_k8 \
        bash eval/mv_recon/run_streaming_recon.sh
    )
    "${python_bin}" "${repo_root}/scripts/summarize_stage3_3b_recon.py" \
        --results-root "${repo_root}/eval_results/mv_recon" \
        --name-filter stage3_7c \
        --output "${repo_root}/stage3_7c_recon_results.csv"
}

for part in ${parts}; do
    case "${part}" in
        pose) run_pose ;;
        static) run_static_reconstruction ;;
        dynamic) run_dynamic_reconstruction ;;
        *) echo "Unknown Stage 3.7 part: ${part}" >&2; exit 2 ;;
    esac
done

if [[ "${STREAMVGGT_STAGE3_7_SKIP_FINALIZE:-0}" == "1" ]]; then
    echo "Skipping Stage 3.7 comparison/gate as requested."
    exit 0
fi

for required in \
    stage3_3_pose_results.csv \
    refine_stage3_3b_recon_results.csv \
    stage3_3c_recon_results.csv \
    stage3_7_pose_results.csv \
    stage3_7b_recon_results.csv \
    stage3_7c_recon_results.csv; do
    if [[ ! -f "${repo_root}/${required}" ]]; then
        echo "Missing Stage 3.7 comparison input: ${repo_root}/${required}" >&2
        exit 2
    fi
done

"${python_bin}" "${repo_root}/scripts/summarize_stage3_7.py" \
    --pose-baseline "${repo_root}/stage3_3_pose_results.csv" \
    --pose-k8 "${repo_root}/stage3_7_pose_results.csv" \
    --static-baseline "${repo_root}/refine_stage3_3b_recon_results.csv" \
    --static-k8 "${repo_root}/stage3_7b_recon_results.csv" \
    --dynamic-baseline "${repo_root}/stage3_3c_recon_results.csv" \
    --dynamic-k8 "${repo_root}/stage3_7c_recon_results.csv" \
    --output "${repo_root}/stage3_7_comparison.csv"
"${python_bin}" "${repo_root}/scripts/check_stage3_7_gate.py" \
    --input "${repo_root}/stage3_7_comparison.csv" \
    --output "${repo_root}/stage3_7_gate.csv"
