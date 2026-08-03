#!/usr/bin/env bash

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
# shellcheck source=common.sh
source "${script_dir}/common.sh"

usage() {
    cat <<'EOF'
Usage: scripts/reproduce/build_supplementary.sh

Rebuild the supplementary CSV result tables and the note defining normalised
regret/oracle wins. The author-side result CSV/archives must be present at their
default repository locations.

Environment variables:
  OUTPUT_ROOT            Default: supplementary.
  DRY_RUN                Print the build command without reading source files.
  PYTHON_BIN             Python executable override.
EOF
    common_usage
}

only_help_or_no_args usage "$@"
init_runtime 0

resolve_repo_path "${OUTPUT_ROOT:-supplementary}" output_root
frozen_sources=(
    "stage4a_video_depth_results(1).csv"
    stage4b_video_depth_sequence_results.csv
    stage4b_video_depth_paired_comparison.csv
    stage4b_video_depth_statistics.csv
    stage4b_pareto.csv
    stage3_3_pose_results.csv
    stage3_7_pose_results.csv
    stage4_supp_pose_metrics.tar.gz
    refine_stage3_3b_recon_results.csv
    stage3_7b_recon_results.csv
    stage3_3c_recon_results.csv
    stage3_7c_recon_results.csv
    stage3_7_sequence_metrics.tar.gz
    stage4b_cross_task_summary.csv
    stage4b_cross_task_regret.csv
    stage4c_results.csv
)
missing_sources=()
if [[ "${DRY_RUN}" != "1" ]]; then
    for source in "${frozen_sources[@]}"; do
        if [[ ! -f "${REPO_ROOT}/${source}" ]]; then
            missing_sources+=("${source}")
        fi
    done
    if (( ${#missing_sources[@]} > 0 )); then
        echo "The committed supplementary/ directory is usable from a normal clone." >&2
        echo "Rebuilding it requires the frozen author-side paper source bundle; missing:" >&2
        printf '  %s\n' "${missing_sources[@]}" >&2
        die "obtain the frozen source bundle or use the committed supplementary assets"
    fi
fi
args=(
    "${PYTHON_BIN}" "${REPO_ROOT}/scripts/build_supplementary_assets.py"
    --output-root "${output_root}"
)

note "building supplementary assets"
run_in_dir "${REPO_ROOT}" "${args[@]}"
