#!/usr/bin/env bash

# Stage 4E-A: offline K4/K8 pose-composability screen.
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
stage4c_results="${STREAMVGGT_STAGE4E_A_STAGE4C_RESULTS:-${repo_root}/stage4c_results.csv}"
stage4c_root="${STREAMVGGT_STAGE4E_A_STAGE4C_ROOT:-${repo_root}/eval_results/stage4c_tum_long}"
output_root="${STREAMVGGT_STAGE4E_A_OUTPUT_ROOT:-${repo_root}/eval_results/stage4e_a_pose_fusion}"
lengths="${STREAMVGGT_STAGE4E_A_LENGTHS:-250 500 1000}"
sequences="${STREAMVGGT_STAGE4E_A_SEQUENCES:-rgbd_dataset_freiburg1_room rgbd_dataset_freiburg2_desk rgbd_dataset_freiburg3_long_office_household}"

if [[ -z "${CONDA_PREFIX:-}" || ! -x "${CONDA_PREFIX}/bin/python" ]]; then
    echo "Stage 4E-A requires the activated StreamVGGT Conda environment" >&2
    echo "CONDA_DEFAULT_ENV=${CONDA_DEFAULT_ENV:-unset}, CONDA_PREFIX=${CONDA_PREFIX:-unset}" >&2
    exit 2
fi
python_bin="${CONDA_PREFIX}/bin/python"
"${python_bin}" -c 'import evo, numpy, scipy, sys; print("Stage 4E-A Python:", sys.executable, "NumPy:", numpy.__version__, "SciPy:", scipy.__version__)'

read -r -a length_array <<< "${lengths}"
read -r -a sequence_array <<< "${sequences}"
"${python_bin}" "${repo_root}/scripts/evaluate_stage4e_a_pose_fusion.py" \
    --stage4c-results "${stage4c_results}" \
    --stage4c-results-root "${stage4c_root}" \
    --output-root "${output_root}" \
    --output "${repo_root}/stage4e_a_sequence_results.csv" \
    --summary-output "${repo_root}/stage4e_a_results.csv" \
    --lengths "${length_array[@]}" \
    --sequences "${sequence_array[@]}"

expected_units=$((${#sequence_array[@]} * ${#length_array[@]}))
"${python_bin}" "${repo_root}/scripts/check_stage4e_a_gate.py" \
    --input "${repo_root}/stage4e_a_sequence_results.csv" \
    --output "${repo_root}/stage4e_a_gate.csv" \
    --expected-units "${expected_units}"
