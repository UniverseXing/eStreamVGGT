#!/usr/bin/env bash

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
# shellcheck source=common.sh
source "${script_dir}/common.sh"

usage() {
    cat <<'EOF'
Usage: scripts/reproduce/run_pose.sh

Reproduce the final streaming pose matrix on Sintel, ScanNet and TUM.

Environment variables:
  DATASETS          Default: "sintel scannet tum".
  POSE_RESULTS_ROOT Task output override.
  DATA_ROOT         Root override when exactly one dataset is selected.
  SINTEL_ROOT       Default: data/eval/sintel/training/final.
  SINTEL_ANNO_ROOT  Default: data/eval/sintel/training/camdata_left.
  SCANNET_ROOT      Default: data/eval/scannetv2.
  TUM_ROOT          Default: data/eval/tum.
  SEQ_LIST          Optional debug subset (single dataset only).
  SINTEL_SEQUENCES  Frozen 14-sequence pose set; overriding marks a debug run.
  SCANNET_SEQUENCES Frozen 6-sequence pose set; overriding marks a debug run.
  TUM_SEQUENCES     Frozen 8-sequence pose set; overriding marks a debug run.
  MAX_FRAMES        Optional debug-only per-sequence frame cap.
  STRIDE            Must be 1 for the frozen pose protocol.
  SIZE              Must be 518 for the frozen pose protocol.
  LOG_SELECTIONS    Save cache-selection logs. Default: 1.
  RESUME            Reuse unaffected rows from pose_metrics.json. Default: 0.
EOF
    common_usage
}

only_help_or_no_args usage "$@"
init_runtime 1

split_words "${METHODS:-${DEFAULT_METHODS_STRING}}" methods
validate_methods methods
split_words "${DATASETS:-sintel scannet tum}" datasets
require_nonempty_array DATASETS datasets
for dataset in "${datasets[@]}"; do
    case "${dataset}" in
        sintel|scannet|tum) ;;
        *) die "unsupported pose dataset: ${dataset}" ;;
    esac
done
if [[ -n "${DATA_ROOT:-}" && ${#datasets[@]} -ne 1 ]]; then
    die "DATA_ROOT is only unambiguous when DATASETS contains one dataset"
fi
if [[ -n "${SEQ_LIST:-}" && ${#datasets[@]} -ne 1 ]]; then
    die "SEQ_LIST may only be used when DATASETS contains one dataset"
fi
validate_switch LOG_SELECTIONS "${LOG_SELECTIONS:-1}"
validate_switch RESUME "${RESUME:-0}"

resolve_repo_path "${WEIGHTS:-ckpt/checkpoints.pth}" weights
resolve_repo_path "${RESULTS_ROOT:-eval_results/reproduce}" results_root
if [[ -n "${POSE_RESULTS_ROOT:-}" ]]; then
    resolve_repo_path "${POSE_RESULTS_ROOT}" output_root
else
    output_root="${results_root}/pose"
fi
require_file "${weights}" "checkpoint"
runtime_preflight

split_words "${SEQ_LIST:-}" sequence_subset
size="${SIZE:-518}"
stride="${STRIDE:-1}"
[[ "${size}" == "518" ]] || die "the frozen pose protocol requires SIZE=518"
[[ "${stride}" == "1" ]] || die "the frozen pose protocol requires STRIDE=1"

allow_subset=0
if [[ -n "${SEQ_LIST+x}" \
    || -n "${MAX_FRAMES+x}" \
    || -n "${SINTEL_SEQUENCES+x}" \
    || -n "${SCANNET_SEQUENCES+x}" \
    || -n "${TUM_SEQUENCES+x}" ]]; then
    allow_subset=1
    note "pose run scope: debug_subset (a sequence/frame override was set)"
else
    note "pose run scope: frozen"
fi

default_sintel_sequences="alley_2 ambush_4 ambush_5 ambush_6 cave_2 cave_4 market_2 market_5 market_6 shaman_3 sleeping_1 sleeping_2 temple_2 temple_3"
default_scannet_sequences="scene0707_00 scene0708_00 scene0709_00 scene0710_00 scene0711_00 scene0712_00"
default_tum_sequences="rgbd_dataset_freiburg3_sitting_halfsphere rgbd_dataset_freiburg3_sitting_rpy rgbd_dataset_freiburg3_sitting_static rgbd_dataset_freiburg3_sitting_xyz rgbd_dataset_freiburg3_walking_halfsphere rgbd_dataset_freiburg3_walking_rpy rgbd_dataset_freiburg3_walking_static rgbd_dataset_freiburg3_walking_xyz"

for method in "${methods[@]}"; do
    for dataset in "${datasets[@]}"; do
        output_dir="${output_root}/${dataset}_reproduce_${method}"
        args=(
            "${SRC_DIR}/eval/pose_evaluation/eval_streaming_pose.py"
            --weights "${weights}"
            --dataset "${dataset}"
            --output-dir "${output_dir}"
            --size "${size}"
            --stride "${stride}"
        )
        append_cache_cli_args args "${method}"

        data_root=""
        anno_root=""
        if [[ -n "${DATA_ROOT:-}" ]]; then
            resolve_repo_path "${DATA_ROOT}" data_root
        else
            case "${dataset}" in
                sintel)
                    resolve_repo_path "${SINTEL_ROOT:-data/eval/sintel/training/final}" data_root
                    ;;
                scannet) resolve_repo_path "${SCANNET_ROOT:-data/eval/scannetv2}" data_root ;;
                tum) resolve_repo_path "${TUM_ROOT:-data/eval/tum}" data_root ;;
            esac
        fi
        if [[ "${dataset}" == "sintel" ]]; then
            resolve_repo_path "${SINTEL_ANNO_ROOT:-data/eval/sintel/training/camdata_left}" anno_root
        fi
        args+=(--data-root "${data_root}")
        if [[ "${dataset}" == "sintel" && -n "${anno_root}" ]]; then
            args+=(--anno-root "${anno_root}")
        fi
        dataset_sequences=()
        if (( ${#sequence_subset[@]} > 0 )); then
            dataset_sequences=("${sequence_subset[@]}")
        else
            case "${dataset}" in
                sintel) split_words "${SINTEL_SEQUENCES:-${default_sintel_sequences}}" dataset_sequences ;;
                scannet) split_words "${SCANNET_SEQUENCES:-${default_scannet_sequences}}" dataset_sequences ;;
                tum) split_words "${TUM_SEQUENCES:-${default_tum_sequences}}" dataset_sequences ;;
            esac
        fi
        require_nonempty_array "${dataset^^}_SEQUENCES" dataset_sequences
        args+=(--seq-list "${dataset_sequences[@]}")
        if [[ -n "${MAX_FRAMES:-}" ]]; then
            args+=(--max-frames "${MAX_FRAMES}")
        fi
        if [[ "${LOG_SELECTIONS:-1}" == "1" ]]; then
            args+=(--log-selections)
        fi
        if [[ "${RESUME:-0}" == "1" ]]; then
            args+=(--resume)
        fi

        note "pose ${dataset}: ${method}"
        run_in_dir "${SRC_DIR}" "${ACCELERATE_BIN}" launch --num_processes 1 "${args[@]}"
    done
done

summary_args=(
    "${REPO_ROOT}/scripts/summarize_stage3_3_pose.py"
    --results-root "${output_root}"
    --name-filter reproduce
    --output "${output_root}/pose_results.csv"
    --expected-runs "$(( ${#methods[@]} * ${#datasets[@]} ))"
    --require-all-success
)
if [[ "${allow_subset}" == "1" ]]; then
    summary_args+=(--allow-subset)
fi
run_command "${PYTHON_BIN}" "${summary_args[@]}"
