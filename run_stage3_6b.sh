#!/usr/bin/env bash

# Stage 3.6B: true-streaming input/output release and 100/500/1000-frame scaling.
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
results_root="${STREAMVGGT_STAGE3_6B_RESULTS_ROOT:-${repo_root}/eval_results/stage3_6b}"
weights="${STREAMVGGT_STAGE3_6B_WEIGHTS:-${repo_root}/ckpt/checkpoints.pth}"
bonn_root="${STREAMVGGT_STAGE3_6B_BONN_ROOT:-${repo_root}/data/eval/bonn/rgbd_bonn_dataset}"
seven_scenes_root="${STREAMVGGT_STAGE3_6B_7SCENES_ROOT:-${repo_root}/data/eval/7scenes}"
bonn_sequence="${STREAMVGGT_STAGE3_6B_BONN_SEQUENCE:-person_tracking2}"
seven_scenes_sequence="${STREAMVGGT_STAGE3_6B_7SCENES_SEQUENCE:-chess/seq-03}"
long_lengths="${STREAMVGGT_STAGE3_6B_LONG_LENGTHS:-100 500 1000}"

if [[ -z "${CONDA_PREFIX:-}" || ! -x "${CONDA_PREFIX}/bin/python" ]]; then
    echo "Stage 3.6B requires an activated Conda environment with CONDA_PREFIX/bin/python" >&2
    echo "CONDA_DEFAULT_ENV=${CONDA_DEFAULT_ENV:-unset}, CONDA_PREFIX=${CONDA_PREFIX:-unset}" >&2
    exit 2
fi
python_bin="${CONDA_PREFIX}/bin/python"
"${python_bin}" -c 'import numpy, torch, sys; print("Stage 3.6B Python:", sys.executable, "NumPy:", numpy.__version__, "Torch:", torch.__version__)'

run_eval() {
    local method="$1"
    local dataset="$2"
    local data_root="$3"
    local sequence="$4"
    local frames="$5"
    local mode="$6"
    shift 6
    echo "===== Stage 3.6B ${method}: ${dataset}/${sequence}, ${frames} frames ====="
    "${python_bin}" "${repo_root}/src/eval/long_sequence/eval_stage3_6b_streaming_memory.py" \
        --weights "${weights}" \
        --dataset "${dataset}" \
        --data-root "${data_root}" \
        --sequence "${sequence}" \
        --output-dir "${results_root}/${method}" \
        --method "${method}" \
        --mode "${mode}" \
        --size "${STREAMVGGT_STAGE3_6B_SIZE:-518}" \
        --max-frames "${frames}" \
        --cache-window 8 \
        --cache-policy temporal_binned_dino_k8 \
        "$@"
}

bonn_frames="${STREAMVGGT_STAGE3_6B_BONN_FRAMES:-110}"
run_eval bonn_legacy_110 bonn "${bonn_root}" "${bonn_sequence}" "${bonn_frames}" legacy_retain --collect-depth
run_eval bonn_stream_110 bonn "${bonn_root}" "${bonn_sequence}" "${bonn_frames}" stream_release --collect-depth

read -r -a lengths <<< "${long_lengths}"
for frames in "${lengths[@]}"; do
    run_eval "7scenes_stream_${frames}" 7scenes_raw "${seven_scenes_root}" \
        "${seven_scenes_sequence}" "${frames}" stream_release
done

"${python_bin}" "${repo_root}/scripts/summarize_stage3_6b.py" \
    --results-root "${results_root}" \
    --output "${repo_root}/stage3_6b_results.csv"

if [[ "${STREAMVGGT_STAGE3_6B_SKIP_GATE:-0}" == "1" ]]; then
    echo "Skipping the formal Stage 3.6B gate as requested."
else
    "${python_bin}" "${repo_root}/scripts/check_stage3_6b_gate.py" \
        --input "${repo_root}/stage3_6b_results.csv" \
        --output "${repo_root}/stage3_6b_gate.csv"
fi
