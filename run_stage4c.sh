#!/usr/bin/env bash

# Stage 4C: frozen unseen raw-TUM long-sequence validation.
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
results_root="${STREAMVGGT_STAGE4C_RESULTS_ROOT:-${repo_root}/eval_results/stage4c_tum_long}"
data_root="${STREAMVGGT_STAGE4C_DATA_ROOT:-${repo_root}/data/eval/stage4c_tum}"
weights="${STREAMVGGT_STAGE4C_WEIGHTS:-${repo_root}/ckpt/checkpoints.pth}"
methods="${STREAMVGGT_STAGE4C_METHODS:-full_cache stage3_2_k4 old_dino_k6 temporal_binned_dino_k8}"
sequences="${STREAMVGGT_STAGE4C_SEQUENCES:-rgbd_dataset_freiburg1_room rgbd_dataset_freiburg2_desk rgbd_dataset_freiburg3_long_office_household}"
lengths="${STREAMVGGT_STAGE4C_LENGTHS:-100 250 500 1000}"
read -r -a sequence_array <<< "${sequences}"
read -r -a length_array <<< "${lengths}"
max_length=0
for length in "${length_array[@]}"; do
    if (( length > max_length )); then
        max_length="${length}"
    fi
done

if [[ -z "${CONDA_PREFIX:-}" || ! -x "${CONDA_PREFIX}/bin/python" ]]; then
    echo "Stage 4C requires the activated StreamVGGT Conda environment" >&2
    echo "CONDA_DEFAULT_ENV=${CONDA_DEFAULT_ENV:-unset}, CONDA_PREFIX=${CONDA_PREFIX:-unset}" >&2
    exit 2
fi
python_bin="${CONDA_PREFIX}/bin/python"
"${python_bin}" -c 'import numpy, scipy, torch, sys; print("Stage 4C Python:", sys.executable, "NumPy:", numpy.__version__, "Torch:", torch.__version__)'

"${python_bin}" "${repo_root}/scripts/check_stage4c_data.py" \
    --root "${data_root}" \
    --sequences "${sequence_array[@]}" \
    --min-frames "${max_length}"

run_eval() {
    local method="$1"
    local sequence="$2"
    local frames="$3"
    local output_dir="${results_root}/${method}/${sequence}/${frames}"
    echo "===== Stage 4C ${method}: ${sequence}, ${frames} frames ====="
    "${python_bin}" "${repo_root}/src/eval/long_sequence/eval_stage4c_tum_long.py" \
        --weights "${weights}" \
        --data-root "${data_root}" \
        --sequence "${sequence}" \
        --output-dir "${output_dir}" \
        --method "${method}" \
        --max-frames "${frames}" \
        --size "${STREAMVGGT_STAGE4C_SIZE:-518}"
}

for method in ${methods}; do
    for sequence in ${sequences}; do
        for frames in ${lengths}; do
            if run_eval "${method}" "${sequence}" "${frames}"; then
                continue
            fi
            echo "Stage 4C recorded failure: ${method}/${sequence}/${frames}" >&2
            if [[ "${method}" == "full_cache" ]]; then
                echo "Stopping full-cache scaling for ${sequence} after its first failure."
                break
            fi
        done
    done
done

"${python_bin}" "${repo_root}/scripts/summarize_stage4c.py" \
    --results-root "${results_root}" \
    --output "${repo_root}/stage4c_results.csv"

if [[ "${STREAMVGGT_STAGE4C_SKIP_GATE:-0}" == "1" ]]; then
    echo "Skipping the formal Stage 4C gate as requested."
    exit 0
fi
"${python_bin}" "${repo_root}/scripts/check_stage4c_gate.py" \
    --input "${repo_root}/stage4c_results.csv" \
    --output "${repo_root}/stage4c_gate.csv" \
    --sequences "${sequence_array[@]}" \
    --lengths "${length_array[@]}"
