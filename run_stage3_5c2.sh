#!/usr/bin/env bash

# Stage 3.5C-2: compare standard and temporally binned DINO K8.
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
results_root="${STREAMVGGT_STAGE3_5C2_RESULTS_ROOT:-${repo_root}/eval_results/stage3_5c2}"
weights="${STREAMVGGT_STAGE3_5C2_WEIGHTS:-${repo_root}/ckpt/checkpoints.pth}"
data_root="${STREAMVGGT_STAGE3_5C2_BONN_ROOT:-${repo_root}/data/eval/bonn/rgbd_bonn_dataset}"
sequence="${STREAMVGGT_STAGE3_5C2_SEQUENCE:-person_tracking2}"
prefixes="${STREAMVGGT_STAGE3_5C2_PREFIX_FRAMES:-10 20 30 40 50 60 70 80 90 100 110}"

if [[ -z "${CONDA_PREFIX:-}" || ! -x "${CONDA_PREFIX}/bin/python" ]]; then
    echo "Stage 3.5C-2 requires an activated Conda environment with CONDA_PREFIX/bin/python" >&2
    echo "CONDA_DEFAULT_ENV=${CONDA_DEFAULT_ENV:-unset}, CONDA_PREFIX=${CONDA_PREFIX:-unset}" >&2
    exit 2
fi
python_bin="${CONDA_PREFIX}/bin/python"
"${python_bin}" -c 'import numpy, sys; print("Stage 3.5C-2 Python:", sys.executable, "NumPy:", numpy.__version__)'

run_eval() {
    local label="$1"
    local window="$2"
    local policy="$3"
    local -a command prefix_args

    read -r -a prefix_args <<< "${prefixes}"
    command=(
        "${python_bin}" "${repo_root}/src/eval/long_sequence/eval_stage3_4_long.py"
        --weights "${weights}"
        --dataset bonn
        --data-root "${data_root}"
        --output-dir "${results_root}/bonn_${label}"
        --seq-list "${sequence}"
        --size "${STREAMVGGT_STAGE3_5C2_SIZE:-518}"
        --prefix-frames "${prefix_args[@]}"
        --trace-memory
    )
    if [[ -n "${STREAMVGGT_STAGE3_5C2_MAX_FRAMES:-}" ]]; then
        command+=(--max-frames "${STREAMVGGT_STAGE3_5C2_MAX_FRAMES}")
    fi
    if [[ -n "${window}" ]]; then
        command+=(--cache-window "${window}" --cache-policy "${policy}")
    fi

    echo "===== Stage 3.5C-2 ${sequence}: ${label} ====="
    "${command[@]}"
}

run_eval full_cache "" ""
run_eval standard_dino_k8 8 anchor_recent_dino_diverse
run_eval temporal_binned_dino_k8 8 temporal_binned_dino_k8

"${python_bin}" "${repo_root}/scripts/summarize_stage3_4.py" \
    --results-root "${results_root}" \
    --output "${repo_root}/stage3_5c2_results.csv" \
    --sequence-output "${repo_root}/stage3_5c2_sequence_results.csv"

"${python_bin}" "${repo_root}/scripts/check_stage3_5b_gate.py" \
    --input "${repo_root}/stage3_5c2_results.csv" \
    --output "${repo_root}/stage3_5c2_gate.csv" \
    --candidates standard_dino_k8 temporal_binned_dino_k8 \
    --temporal-bank-candidates temporal_binned_dino_k8
