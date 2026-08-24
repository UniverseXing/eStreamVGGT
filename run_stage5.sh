#!/usr/bin/env bash

# Stage 5 conference evidence completion: two experiments only.
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
python_bin="${CONDA_PREFIX:-}/bin/python"
parts="${STREAMVGGT_STAGE5_PARTS:-same_budget memory finalize}"

if [[ -z "${CONDA_PREFIX:-}" || ! -x "${python_bin}" ]]; then
    echo "Stage 5 requires the activated StreamVGGT Conda environment" >&2
    echo "CONDA_DEFAULT_ENV=${CONDA_DEFAULT_ENV:-unset}, CONDA_PREFIX=${CONDA_PREFIX:-unset}" >&2
    exit 2
fi
"${python_bin}" -c 'import numpy, torch, sys; print("Stage 5 Python:", sys.executable, "NumPy:", numpy.__version__, "Torch:", torch.__version__, "GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none")'

for part in ${parts}; do
    case "${part}" in
        same_budget)
            bash "${repo_root}/run_stage5a.sh"
            ;;
        memory)
            bash "${repo_root}/run_stage5b.sh"
            ;;
        finalize)
            stage5a_args=(--repo-root "${repo_root}")
            if [[ "${STREAMVGGT_STAGE5A_INCLUDE_OPTIONAL:-0}" == "1" ]]; then
                stage5a_args+=(--include-optional)
            fi
            if [[ "${STREAMVGGT_STAGE5A_RERUN_REFERENCES:-0}" == "1" ]]; then
                stage5a_args+=(--rerun-references)
            fi
            "${python_bin}" "${repo_root}/scripts/summarize_stage5a.py" "${stage5a_args[@]}"
            "${python_bin}" "${repo_root}/scripts/summarize_stage5b_memory.py" \
                --repo-root "${repo_root}"
            "${python_bin}" "${repo_root}/scripts/plot_stage5b_memory.py" \
                --repo-root "${repo_root}"
            ;;
        *) echo "Unknown Stage 5 part: ${part}" >&2; exit 2 ;;
    esac
done
