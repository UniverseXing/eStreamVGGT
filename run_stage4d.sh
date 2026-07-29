#!/usr/bin/env bash

# Stage 4D: frozen paper tables, figures, case audit, and provenance.
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
output_root="${STREAMVGGT_STAGE4D_OUTPUT_ROOT:-${repo_root}/paper_assets/stage4d}"
stage4c_root="${STREAMVGGT_STAGE4D_STAGE4C_ROOT:-${repo_root}/eval_results/stage4c_tum_long}"
allow_missing="${STREAMVGGT_STAGE4D_ALLOW_MISSING_SERVER_ASSETS:-0}"

if [[ -z "${CONDA_PREFIX:-}" || ! -x "${CONDA_PREFIX}/bin/python" ]]; then
    echo "Stage 4D requires the activated StreamVGGT Conda environment" >&2
    echo "CONDA_DEFAULT_ENV=${CONDA_DEFAULT_ENV:-unset}, CONDA_PREFIX=${CONDA_PREFIX:-unset}" >&2
    exit 2
fi
python_bin="${CONDA_PREFIX}/bin/python"
export MPLBACKEND=Agg
export MPLCONFIGDIR="${STREAMVGGT_STAGE4D_MPLCONFIGDIR:-${TMPDIR:-/tmp}/streamvggt-matplotlib-${SLURM_JOB_ID:-local}}"
mkdir -p "${MPLCONFIGDIR}"

extra_args=()
if [[ "${allow_missing}" == "1" ]]; then
    extra_args+=(--allow-missing-server-assets)
fi

"${python_bin}" -c 'import matplotlib, numpy, scipy, sys; print("Stage 4D Python:", sys.executable, "Matplotlib:", matplotlib.__version__, "NumPy:", numpy.__version__)'
"${python_bin}" "${repo_root}/scripts/build_stage4d_paper_assets.py" \
    --stage4c-results-root "${stage4c_root}" \
    --output-root "${output_root}" \
    --results-output "${repo_root}/stage4d_results.csv" \
    --case-output "${repo_root}/stage4d_case_audit.csv" \
    --manifest-output "${repo_root}/stage4d_asset_manifest.csv" \
    "${extra_args[@]}"

if [[ "${allow_missing}" == "1" ]]; then
    echo "Stage 4D partial build finished; formal gate skipped because server assets were optional."
    exit 0
fi

"${python_bin}" "${repo_root}/scripts/check_stage4d_gate.py" \
    --asset-root "${output_root}" \
    --case-audit "${repo_root}/stage4d_case_audit.csv" \
    --asset-manifest "${repo_root}/stage4d_asset_manifest.csv" \
    --output "${repo_root}/stage4d_gate.csv"

tar -czf "${repo_root}/stage4d_paper_assets.tar.gz" \
    -C "${repo_root}" \
    "paper_assets/stage4d" \
    "stage4d_results.csv" \
    "stage4d_case_audit.csv" \
    "stage4d_asset_manifest.csv" \
    "stage4d_gate.csv"
echo "Wrote ${repo_root}/stage4d_paper_assets.tar.gz"
