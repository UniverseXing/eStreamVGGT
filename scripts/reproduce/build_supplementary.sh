#!/usr/bin/env bash

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
# shellcheck source=common.sh
source "${script_dir}/common.sh"

usage() {
    cat <<'EOF'
Usage: scripts/reproduce/build_supplementary.sh

Rebuild the frozen supplementary CSV tables, audit manifests and PDF figures.
The source summary CSV/archives documented in supplementary/README.md must be
present at their default repository locations.

Environment variables:
  OUTPUT_ROOT            Default: supplementary.
  FIGURE_SOURCE_ARCHIVE  Default: stage4_supp_figure_sources.tar.gz.
  SKIP_FIGURES           Set to 1 to build CSV tables only. Default: 0.
  DRY_RUN                Print the build command without reading source files.
  PYTHON_BIN             Python executable override.
EOF
    common_usage
}

only_help_or_no_args usage "$@"
init_runtime 0
validate_switch SKIP_FIGURES "${SKIP_FIGURES:-0}"

resolve_repo_path "${OUTPUT_ROOT:-supplementary}" output_root
resolve_repo_path "${FIGURE_SOURCE_ARCHIVE:-stage4_supp_figure_sources.tar.gz}" figure_archive
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
    stage4b_method_roles.csv
    stage4b_claim_audit.csv
    stage4a_gate.csv
    stage4c_results.csv
    stage4c_gate.csv
    stage4d_case_audit.csv
    stage4d_gate.csv
)
missing_sources=()
if [[ "${DRY_RUN}" != "1" ]]; then
    for source in "${frozen_sources[@]}"; do
        if [[ ! -f "${REPO_ROOT}/${source}" ]]; then
            missing_sources+=("${source}")
        fi
    done
    if [[ "${SKIP_FIGURES:-0}" != "1" && ! -f "${figure_archive}" ]]; then
        missing_sources+=("${figure_archive}")
    fi
    if (( ${#missing_sources[@]} > 0 )); then
        echo "The committed supplementary/ directory is usable from a normal clone." >&2
        echo "Rebuilding it requires the frozen author-side paper source bundle; missing:" >&2
        printf '  %s\n' "${missing_sources[@]}" >&2
        die "obtain the frozen source bundle or use the committed supplementary assets"
    fi

    source_manifest="${REPO_ROOT}/supplementary/source_manifest.csv"
    [[ -f "${source_manifest}" ]] || die "missing frozen source manifest: ${source_manifest}"
    declare -A expected_sizes=()
    declare -A expected_hashes=()
    while IFS=, read -r source expected_size expected_hash; do
        expected_hash="${expected_hash%$'\r'}"
        [[ "${source}" == "source_file" ]] && continue
        expected_sizes["${source}"]="${expected_size}"
        expected_hashes["${source}"]="${expected_hash}"
    done < "${source_manifest}"

    figure_source_name="stage4_supp_figure_sources.tar.gz"
    sources_to_verify=("${frozen_sources[@]}")
    if [[ "${SKIP_FIGURES:-0}" != "1" ]]; then
        sources_to_verify+=("${figure_source_name}")
    fi
    mismatched_sources=()
    for source in "${sources_to_verify[@]}"; do
        source_path="${REPO_ROOT}/${source}"
        if [[ "${source}" == "${figure_source_name}" ]]; then
            source_path="${figure_archive}"
        fi
        if [[ -z "${expected_hashes[${source}]:-}" ]]; then
            mismatched_sources+=("${source} (not listed in source_manifest.csv)")
            continue
        fi
        actual_size="$(stat -c %s "${source_path}")"
        actual_hash="$(sha256sum "${source_path}" | awk '{print $1}')"
        if [[ "${actual_size}" != "${expected_sizes[${source}]}" || \
              "${actual_hash}" != "${expected_hashes[${source}]}" ]]; then
            mismatched_sources+=("${source}")
        fi
    done
    if (( ${#mismatched_sources[@]} > 0 )); then
        echo "Frozen supplementary source verification failed:" >&2
        printf '  %s\n' "${mismatched_sources[@]}" >&2
        die "source size/hash differs from supplementary/source_manifest.csv"
    fi
fi
mpl_config_dir="${MPLCONFIGDIR:-${TMPDIR:-/tmp}/estreamvggt-matplotlib}"
if [[ "${DRY_RUN}" != "1" ]]; then
    mkdir -p "${mpl_config_dir}"
fi
args=(
    env
    MPLBACKEND=Agg
    "MPLCONFIGDIR=${mpl_config_dir}"
    "${PYTHON_BIN}" "${REPO_ROOT}/scripts/build_supplementary_assets.py"
    --output-root "${output_root}"
    --figure-source-archive "${figure_archive}"
)
if [[ "${SKIP_FIGURES:-0}" == "1" ]]; then
    args+=(--skip-figures)
fi

note "building supplementary assets"
run_in_dir "${REPO_ROOT}" "${args[@]}"
