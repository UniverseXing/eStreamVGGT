#!/usr/bin/env bash

# Stage 5B conference experiment: Full/K4 x accumulate/release memory factorial.
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
python_bin="${CONDA_PREFIX:-}/bin/python"
results_root="${STREAMVGGT_STAGE5B_RESULTS_ROOT:-${repo_root}/eval_results/stage5b_memory}"
weights="${STREAMVGGT_STAGE5B_WEIGHTS:-${repo_root}/ckpt/checkpoints.pth}"
data_root="${STREAMVGGT_STAGE5B_BONN_ROOT:-${repo_root}/data/eval/bonn/rgbd_bonn_dataset}"
sequence="${STREAMVGGT_STAGE5B_SEQUENCE:-person_tracking2}"
frames="${STREAMVGGT_STAGE5B_FRAMES:-110}"

run_cell() {
    local cell="$1"
    local cache="$2"
    local mode="$3"
    local cache_args=()
    if [[ "${cache}" == "k4" ]]; then
        cache_args+=(--cache-window 4 --cache-policy anchor_recent_dino_diverse_k4)
    else
        cache_args+=(--full-cache)
    fi
    echo "===== Stage 5B ${cell}: ${sequence}, ${frames} frames ====="
    "${python_bin}" "${repo_root}/src/eval/long_sequence/eval_stage3_6b_streaming_memory.py" \
        --weights "${weights}" \
        --dataset bonn \
        --data-root "${data_root}" \
        --sequence "${sequence}" \
        --output-dir "${results_root}/${cell}" \
        --method "${cell}" \
        --mode "${mode}" \
        --size "${STREAMVGGT_STAGE5B_SIZE:-518}" \
        --max-frames "${frames}" \
        --metrics-filename stage5b_metrics.json \
        "${cache_args[@]}"
}

# Both lifecycle modes lazily load one input frame at a time. Their only
# difference is whether dense predictions accumulate on GPU or are released
# through the CPU evaluation sink.
run_cell full_accumulated full stream_accumulate
run_cell full_release full stream_release
run_cell k4_accumulated k4 stream_accumulate
run_cell k4_release k4 stream_release
