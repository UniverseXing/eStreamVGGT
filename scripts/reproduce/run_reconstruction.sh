#!/usr/bin/env bash

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
# shellcheck source=common.sh
source "${script_dir}/common.sh"

usage() {
    cat <<'EOF'
Usage: scripts/reproduce/run_reconstruction.sh

Reproduce the final multi-view reconstruction matrix. The runner separates the
static dense protocol (7-Scenes, Neural RGB-D, ETH3D) from the 50-frame TUM
dynamics paper protocol so one invocation preserves the reported settings.

Environment variables:
  DATASETS                    Default: "7scenes nrgbd eth3d tum".
  RECON_RESULTS_ROOT          Task output override.
  SEVEN_SCENES_ROOT           Default: data/eval/7scenes.
  NRGBD_ROOT                  Default: data/eval/neural_rgbd.
  ETH3D_ROOT                  Default: data/eval/eth3d.
  TUM_ROOT                    Default: data/eval/tum.
  MAX_SCENES / MAX_FRAMES     Optional smoke/debug caps.
  PREFIX_FRAMES               Static prefixes. Default: "4 6 8 10".
  TUM_PREFIX_FRAMES           TUM prefixes. Default: "10 20 30 40 50".
  TUM_FRAMES                  TUM input length. Default: 50.
  TUM_SAMPLING                first (default) or uniform.
  SEVEN_SCENES_SEQUENCES      Default: the 12 valid common-paper sequences.
  TUM_SEQUENCES               Default: the frozen 8-sequence TUM Dynamics set.
  SAVE_ARTIFACTS              Save point clouds/arrays. Default: 0.
  LOG_SELECTIONS              Save selection logs. Default: 1.
  USE_PROJ                    Must be 0 for the frozen direct-point protocol.
  SIZE / SEED                 SIZE must be 518; SEED defaults to 0.

Explicitly setting a sampling, prefix, sequence, seed, or frame/scene-cap
override labels the collected CSV as run_scope=debug_subset. The default
settings are checked against the frozen sequence/frame signatures.
EOF
    common_usage
}

only_help_or_no_args usage "$@"
init_runtime 1

debug_overrides=()
for variable_name in \
    MAX_SCENES MAX_FRAMES PREFIX_FRAMES TUM_PREFIX_FRAMES TUM_FRAMES \
    TUM_SAMPLING SEVEN_SCENES_SEQUENCES TUM_SEQUENCES SEED \
    SEVEN_SCENES_KF_EVERY NRGBD_KF_EVERY; do
    if [[ -v "${variable_name}" ]]; then
        debug_overrides+=("${variable_name}")
    fi
done
summarizer_scope_args=()
if (( ${#debug_overrides[@]} > 0 )); then
    summarizer_scope_args+=(--allow-subset)
    note "debug_subset reconstruction overrides: ${debug_overrides[*]}"
fi

split_words "${METHODS:-${DEFAULT_METHODS_STRING}}" methods
validate_methods methods
split_words "${DATASETS:-7scenes nrgbd eth3d tum}" datasets
require_nonempty_array DATASETS datasets
static_datasets=()
run_tum=0
run_7scenes=0
for dataset in "${datasets[@]}"; do
    case "${dataset}" in
        7scenes)
            static_datasets+=("${dataset}")
            run_7scenes=1
            ;;
        nrgbd|eth3d) static_datasets+=("${dataset}") ;;
        tum) run_tum=1 ;;
        *) die "unsupported reconstruction dataset: ${dataset}" ;;
    esac
done
validate_switch SAVE_ARTIFACTS "${SAVE_ARTIFACTS:-0}"
validate_switch LOG_SELECTIONS "${LOG_SELECTIONS:-1}"
validate_switch USE_PROJ "${USE_PROJ:-0}"
[[ "${USE_PROJ:-0}" == "0" ]] || die "the frozen reconstruction protocol requires USE_PROJ=0"
size="${SIZE:-518}"
[[ "${size}" == "518" ]] || die "the frozen reconstruction protocol requires SIZE=518"

resolve_repo_path "${WEIGHTS:-ckpt/checkpoints.pth}" weights
resolve_repo_path "${RESULTS_ROOT:-eval_results/reproduce}" results_root
if [[ -n "${RECON_RESULTS_ROOT:-}" ]]; then
    resolve_repo_path "${RECON_RESULTS_ROOT}" output_root
else
    output_root="${results_root}/mv_recon"
fi
resolve_repo_path "${SEVEN_SCENES_ROOT:-data/eval/7scenes}" seven_scenes_root
resolve_repo_path "${NRGBD_ROOT:-data/eval/neural_rgbd}" nrgbd_root
resolve_repo_path "${ETH3D_ROOT:-data/eval/eth3d}" eth3d_root
resolve_repo_path "${TUM_ROOT:-data/eval/tum}" tum_root
require_file "${weights}" "checkpoint"
runtime_preflight

split_words "${PREFIX_FRAMES:-4 6 8 10}" prefix_frames
split_words "${TUM_PREFIX_FRAMES:-10 20 30 40 50}" tum_prefix_frames
seven_scenes_sequences="${SEVEN_SCENES_SEQUENCES:-chess/seq-03 chess/seq-05 fire/seq-03 fire/seq-04 heads/seq-01 office/seq-02 pumpkin/seq-01 pumpkin/seq-07 redkitchen/seq-03 redkitchen/seq-04 stairs/seq-01 stairs/seq-04}"
tum_sequences="${TUM_SEQUENCES:-rgbd_dataset_freiburg3_sitting_halfsphere rgbd_dataset_freiburg3_sitting_rpy rgbd_dataset_freiburg3_sitting_static rgbd_dataset_freiburg3_sitting_xyz rgbd_dataset_freiburg3_walking_halfsphere rgbd_dataset_freiburg3_walking_rpy rgbd_dataset_freiburg3_walking_static rgbd_dataset_freiburg3_walking_xyz}"

base_args=(
    --weights "${weights}"
    --model_name StreamVGGT
    --size "${size}"
    --seed "${SEED:-0}"
    --data-root "7scenes=${seven_scenes_root}"
    --data-root "nrgbd=${nrgbd_root}"
    --data-root "eth3d=${eth3d_root}"
    --data-root "tum=${tum_root}"
)
if [[ -n "${MAX_SCENES:-}" ]]; then
    base_args+=(--max-scenes "${MAX_SCENES}")
fi
if [[ -n "${MAX_FRAMES:-}" ]]; then
    base_args+=(--max-frames "${MAX_FRAMES}")
fi
if [[ "${LOG_SELECTIONS:-1}" == "1" ]]; then
    base_args+=(--log-selections)
fi
if [[ "${USE_PROJ:-0}" == "1" ]]; then
    base_args+=(--use_proj)
fi
if [[ "${SAVE_ARTIFACTS:-0}" == "0" ]]; then
    base_args+=(--no-save-artifacts)
fi

for method in "${methods[@]}"; do
    cache_args=()
    append_cache_cli_args cache_args "${method}"
    if (( ${#static_datasets[@]} > 0 )); then
        args=(
            "${SRC_DIR}/eval/mv_recon/launch.py"
            "${base_args[@]}"
            --output_dir "${output_root}/static/streamvggt_reproduce_${method}"
            --protocol dense
            --seven-scenes-kf-every "${SEVEN_SCENES_KF_EVERY:-50}"
            --nrgbd-kf-every "${NRGBD_KF_EVERY:-100}"
            --datasets "${static_datasets[@]}"
            --prefix-frames "${prefix_frames[@]}"
            "${cache_args[@]}"
        )
        if [[ "${run_7scenes}" == "1" ]]; then
            args+=(--dataset-seq-list "7scenes=${seven_scenes_sequences}")
        fi
        note "static reconstruction: ${method} (${static_datasets[*]})"
        run_in_dir "${SRC_DIR}" "${ACCELERATE_BIN}" launch --num_processes 1 "${args[@]}"
    fi

    if [[ "${run_tum}" == "1" ]]; then
        args=(
            "${SRC_DIR}/eval/mv_recon/launch.py"
            "${base_args[@]}"
            --output_dir "${output_root}/tum/streamvggt_reproduce_${method}"
            --protocol paper
            --datasets tum
            --tum-frames "${TUM_FRAMES:-50}"
            --tum-sampling "${TUM_SAMPLING:-first}"
            --prefix-frames "${tum_prefix_frames[@]}"
            "${cache_args[@]}"
        )
        args+=(--dataset-seq-list "tum=${tum_sequences}")
        note "TUM dynamics reconstruction: ${method}"
        run_in_dir "${SRC_DIR}" "${ACCELERATE_BIN}" launch --num_processes 1 "${args[@]}"
    fi
done

if (( ${#static_datasets[@]} > 0 )); then
    run_command "${PYTHON_BIN}" "${REPO_ROOT}/scripts/summarize_stage3_3b_recon.py" \
        --results-root "${output_root}/static" \
        --name-filter reproduce \
        --output "${output_root}/reconstruction_static_results.csv" \
        --expected-runs "${#methods[@]}" \
        --require-all-success \
        "${summarizer_scope_args[@]}"
fi
if [[ "${run_tum}" == "1" ]]; then
    run_command "${PYTHON_BIN}" "${REPO_ROOT}/scripts/summarize_stage3_3b_recon.py" \
        --results-root "${output_root}/tum" \
        --name-filter reproduce \
        --output "${output_root}/reconstruction_tum_results.csv" \
        --expected-runs "${#methods[@]}" \
        --require-all-success \
        "${summarizer_scope_args[@]}"
fi
