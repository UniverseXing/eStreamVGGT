#!/usr/bin/env bash

# Remaining conference supplementary work. Run from the repository root.
set -euo pipefail

repo_root="${PWD}"
python_bin="${CONDA_PREFIX:-}/bin/python"
parts="${STREAMVGGT_SUPPLEMENTARY_PARTS:-refresh_stage5a selector_trace}"

if [[ -z "${CONDA_PREFIX:-}" || ! -x "${python_bin}" ]]; then
    echo "An activated StreamVGGT Conda environment is required" >&2
    exit 2
fi

for part in ${parts}; do
    case "${part}" in
        refresh_stage5a)
            echo "===== Supplementary P0: refresh existing Stage 5A metrics ====="
            "${python_bin}" "scripts/summarize_stage5a.py" \
                --repo-root "${repo_root}" \
                --include-optional \
                --bootstrap-samples 10000 \
                --bootstrap-seed 20260824
            ;;
        selector_trace)
            echo "===== Supplementary P1: K4/K6/K8 selector trace ====="
            "${python_bin}" "scripts/run_supplementary_selector_trace.py" \
                --repo-root "${repo_root}" \
                --weights "${STREAMVGGT_SUPPLEMENTARY_WEIGHTS:-${repo_root}/ckpt/checkpoints.pth}" \
                --images-dir "${STREAMVGGT_SUPPLEMENTARY_IMAGES_DIR:-${repo_root}/data/eval/7scenes/chess/seq-01}" \
                --image-glob "${STREAMVGGT_SUPPLEMENTARY_IMAGE_GLOB:-*.color.png}" \
                --sampling-stride "${STREAMVGGT_SUPPLEMENTARY_SAMPLING_STRIDE:-5}" \
                --max-frames "${STREAMVGGT_SUPPLEMENTARY_MAX_FRAMES:-110}" \
                --output-dir "${STREAMVGGT_SUPPLEMENTARY_RESULTS_ROOT:-${repo_root}/eval_results/supplementary_selector_trace}"
            ;;
        k8_coverage)
            echo "===== Supplementary P1: matched K8 temporal coverage ====="
            "${python_bin}" "scripts/run_supplementary_k8_coverage.py" \
                --repo-root "${repo_root}" \
                --weights "${STREAMVGGT_SUPPLEMENTARY_WEIGHTS:-${repo_root}/ckpt/checkpoints.pth}" \
                --images-dir "${STREAMVGGT_SUPPLEMENTARY_IMAGES_DIR:-${repo_root}/data/eval/7scenes/chess/seq-01}" \
                --image-glob "${STREAMVGGT_SUPPLEMENTARY_IMAGE_GLOB:-*.color.png}" \
                --sampling-stride "${STREAMVGGT_SUPPLEMENTARY_SAMPLING_STRIDE:-5}" \
                --max-frames "${STREAMVGGT_SUPPLEMENTARY_MAX_FRAMES:-110}" \
                --steady-start-frame "${STREAMVGGT_SUPPLEMENTARY_STEADY_START_FRAME:-50}" \
                --output-dir "${STREAMVGGT_SUPPLEMENTARY_K8_COVERAGE_ROOT:-${repo_root}/eval_results/supplementary_k8_coverage}"
            ;;
        k8_pose)
            echo "===== Supplementary P1: matched 1000-frame K8 TUM pose ====="
            results_root="${STREAMVGGT_SUPPLEMENTARY_K8_POSE_ROOT:-${repo_root}/eval_results/supplementary_k8_pose}"
            data_root="${STREAMVGGT_STAGE4C_DATA_ROOT:-${repo_root}/data/eval/stage4c_tum}"
            weights="${STREAMVGGT_SUPPLEMENTARY_WEIGHTS:-${repo_root}/ckpt/checkpoints.pth}"
            run_id="${SLURM_JOB_ID:-manual}-k8-pose-$(date +%s)"
            sequences="${STREAMVGGT_STAGE4C_SEQUENCES:-rgbd_dataset_freiburg1_room rgbd_dataset_freiburg2_desk rgbd_dataset_freiburg3_long_office_household}"
            read -r -a sequence_array <<< "${sequences}"
            "${python_bin}" "scripts/check_stage4c_data.py" \
                --root "${data_root}" \
                --sequences "${sequence_array[@]}" \
                --min-frames 1000
            for method in anchor_recent_dino_diverse_k8 nonhierarchical_dino8; do
                for sequence in "${sequence_array[@]}"; do
                    output_dir="${results_root}/${method}/${sequence}/1000"
                    echo "===== ${method}: ${sequence}, 1000 frames ====="
                    "${python_bin}" "src/eval/long_sequence/eval_stage4c_tum_long.py" \
                        --weights "${weights}" \
                        --data-root "${data_root}" \
                        --sequence "${sequence}" \
                        --output-dir "${output_dir}" \
                        --method "${method}" \
                        --max-frames 1000 \
                        --size 518 \
                        --max-association-difference 0.02 \
                        --run-scope frozen \
                        --run-id "${run_id}"
                done
            done
            "${python_bin}" "scripts/summarize_supplementary_k8_pose.py" \
                --results-root "${results_root}" \
                --output-dir "${results_root}" \
                --expected-run-id "${run_id}"
            ;;
        k8_controls)
            echo "===== Supplementary P1: matched K8 controls ====="
            for method in recent8 nonhierarchical_dino8; do
                case "${method}" in
                    recent8) window=8; policy=fifo ;;
                    nonhierarchical_dino8) window=8; policy=anchor_recent_dino_diverse ;;
                esac
                (
                    cd "${repo_root}/src"
                    export STREAMVGGT_EVAL_DATASETS="${STREAMVGGT_SUPPLEMENTARY_DATASETS:-bonn sintel kitti}"
                    export STREAMVGGT_RUN_TAG="supplementary_p1_${method}"
                    export STREAMVGGT_CACHE_WINDOW="${window}"
                    export STREAMVGGT_CACHE_POLICY="${policy}"
                    export STREAMVGGT_CACHE_RANDOM_SEED=0
                    export STREAMVGGT_LOG_SELECTIONS=0
                    bash eval/video_depth/run.sh
                )
            done
            "${python_bin}" "scripts/summarize_supplementary_k8_controls.py" \
                --repo-root "${repo_root}"
            ;;
        *)
            echo "Unknown supplementary part: ${part}" >&2
            exit 2
            ;;
    esac
done

"${python_bin}" "scripts/build_requested_supplementary_material.py" \
    --repo-root "${repo_root}" \
    --output-root "${repo_root}/supplementary material"
