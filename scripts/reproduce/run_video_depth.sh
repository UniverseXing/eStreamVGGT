#!/usr/bin/env bash

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
# shellcheck source=common.sh
source "${script_dir}/common.sh"

usage() {
    cat <<'EOF'
Usage: scripts/reproduce/run_video_depth.sh

Reproduce the final VideoDepth matrix (Bonn, KITTI and Sintel by default) for
Full cache, K4, K6 and K8. Inference and scale-aligned depth evaluation are run
for every selected dataset-method cell.

Environment variables:
  DATASETS                  Default: "bonn kitti sintel".
  VIDEO_DEPTH_RESULTS_ROOT  Task output override.
  KITTI_ROOT                Default: data/eval/kitti.
  SEQ_LIST                  Optional whitespace-separated sequence subset.
  MAX_FRAMES                Optional per-sequence frame cap.
  SIZE                      Must be 518 for the frozen VideoDepth evaluator.
  ALIGN                     Must be scale for the frozen protocol.
  BOOTSTRAP_SAMPLES         Paired-bootstrap draws. Default: 10000.
  BOOTSTRAP_SEED            Bootstrap RNG seed. Default: 0.
EOF
    common_usage
}

only_help_or_no_args usage "$@"
init_runtime 1

split_words "${METHODS:-${DEFAULT_METHODS_STRING}}" methods
validate_methods methods
split_words "${DATASETS:-bonn kitti sintel}" datasets
require_nonempty_array DATASETS datasets
has_kitti=0
for dataset in "${datasets[@]}"; do
    case "${dataset}" in
        bonn|sintel) ;;
        kitti) has_kitti=1 ;;
        *) die "unsupported VideoDepth dataset: ${dataset}" ;;
    esac
done
if [[ "${has_kitti}" == "1" && -n "${MAX_FRAMES:-}" ]]; then
    die "MAX_FRAMES is not supported by the frozen KITTI depth scorer; omit KITTI or run its full prepared protocol"
fi
if [[ -n "${SEQ_LIST:-}" && ${#datasets[@]} -ne 1 ]]; then
    die "SEQ_LIST may only be used when DATASETS contains one dataset"
fi

resolve_repo_path "${WEIGHTS:-ckpt/checkpoints.pth}" weights
resolve_repo_path "${RESULTS_ROOT:-eval_results/reproduce}" results_root
if [[ -n "${VIDEO_DEPTH_RESULTS_ROOT:-}" ]]; then
    resolve_repo_path "${VIDEO_DEPTH_RESULTS_ROOT}" output_root
else
    output_root="${results_root}/video_depth"
fi
resolve_repo_path "${KITTI_ROOT:-data/eval/kitti}" kitti_root
require_file "${weights}" "checkpoint"
runtime_preflight

size="${SIZE:-518}"
[[ "${size}" == "518" ]] || die "VideoDepth requires SIZE=518"
align="${ALIGN:-scale}"
[[ "${align}" == "scale" ]] || die "the frozen VideoDepth protocol requires ALIGN=scale"
bootstrap_samples="${BOOTSTRAP_SAMPLES:-10000}"
bootstrap_seed="${BOOTSTRAP_SEED:-0}"
[[ "${bootstrap_samples}" =~ ^[1-9][0-9]*$ ]] || die "BOOTSTRAP_SAMPLES must be a positive integer"
(( bootstrap_samples >= 1000 )) || die "BOOTSTRAP_SAMPLES must be at least 1000"
[[ "${bootstrap_seed}" =~ ^-?[0-9]+$ ]] || die "BOOTSTRAP_SEED must be an integer"
split_words "${SEQ_LIST:-}" sequence_subset

for method in "${methods[@]}"; do
    method_config "${method}" window policy
    for dataset in "${datasets[@]}"; do
        output_dir="${output_root}/${dataset}/${method}"
        common_args=(
            --weights "${weights}"
            --output_dir "${output_dir}"
            --eval_dataset "${dataset}"
            --size "${size}"
        )
        metric_args=(
            --output_dir "${output_dir}"
            --eval_dataset "${dataset}"
            --align "${align}"
        )
        if (( ${#sequence_subset[@]} > 0 )); then
            common_args+=(--seq_list "${sequence_subset[@]}")
            metric_args+=(--seq_list "${sequence_subset[@]}")
        fi
        if [[ -n "${MAX_FRAMES:-}" ]]; then
            common_args+=(--max_frames "${MAX_FRAMES}")
            metric_args+=(--max_frames "${MAX_FRAMES}")
        fi

        note "VideoDepth ${dataset}: ${method}"
        run_in_dir "${SRC_DIR}" env \
            "CUDA_LAUNCH_BLOCKING=${CUDA_LAUNCH_BLOCKING:-1}" \
            "STREAMVGGT_CACHE_WINDOW=${window}" \
            "STREAMVGGT_CACHE_POLICY=${policy}" \
            "STREAMVGGT_KITTI_ROOT=${kitti_root}" \
            "${ACCELERATE_BIN}" launch --num_processes 1 \
            "${SRC_DIR}/eval/video_depth/launch.py" "${common_args[@]}"
        run_in_dir "${SRC_DIR}" env \
            "STREAMVGGT_KITTI_ROOT=${kitti_root}" \
            "${PYTHON_BIN}" "${SRC_DIR}/eval/video_depth/eval_depth.py" \
            "${metric_args[@]}"
    done
done

collector_args=(
    "${PYTHON_BIN}" "${script_dir}/collect_video_depth.py"
    --results-root "${output_root}"
    --output "${output_root}/video_depth_results.csv"
    --datasets "${datasets[@]}"
    --methods "${methods[@]}"
    --bootstrap-samples "${bootstrap_samples}"
    --seed "${bootstrap_seed}"
)
if (( ${#sequence_subset[@]} > 0 )) || [[ -n "${MAX_FRAMES:-}" ]]; then
    collector_args+=(--allow-subset)
fi
run_command "${collector_args[@]}"
