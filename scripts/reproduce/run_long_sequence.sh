#!/usr/bin/env bash

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
# shellcheck source=common.sh
source "${script_dir}/common.sh"

usage() {
    cat <<'EOF'
Usage: scripts/reproduce/run_long_sequence.sh

Reproduce the final raw-TUM long-sequence scaling matrix: three held-out
sequences, four prefix lengths and the four public methods. A Full-cache OOM is
recorded by the evaluator and stops only longer prefixes of that same sequence.

Environment variables:
  LONG_RESULTS_ROOT  Task output override.
  DATA_ROOT          Alias for LONG_DATA_ROOT.
  LONG_DATA_ROOT     Default: data/eval/stage4c_tum.
  SEQUENCES          Default: Freiburg1 room, Freiburg2 desk, Freiburg3 long office.
  LENGTHS            Default: "100 250 500 1000".
  MAX_FRAMES         If set, replace LENGTHS with this single prefix.
  SIZE               Must be 518 for the frozen long-sequence protocol.
  MAX_ASSOC_DIFF     RGB/pose association tolerance. Default: 0.02 seconds.
EOF
    common_usage
}

only_help_or_no_args usage "$@"
init_runtime 0

split_words "${METHODS:-${DEFAULT_METHODS_STRING}}" methods
validate_methods methods
split_words "${SEQUENCES:-rgbd_dataset_freiburg1_room rgbd_dataset_freiburg2_desk rgbd_dataset_freiburg3_long_office_household}" sequences
require_nonempty_array SEQUENCES sequences
if [[ -n "${MAX_FRAMES:-}" ]]; then
    lengths=("${MAX_FRAMES}")
else
    split_words "${LENGTHS:-100 250 500 1000}" lengths
fi
require_nonempty_array LENGTHS lengths
size="${SIZE:-518}"
[[ "${size}" == "518" ]] || die "the frozen long-sequence protocol requires SIZE=518"
run_scope="frozen"
if [[ -n "${SEQUENCES+x}" || -n "${LENGTHS+x}" || -n "${MAX_FRAMES+x}" || \
      -n "${MAX_ASSOC_DIFF+x}" ]]; then
    run_scope="debug_subset"
fi

resolve_repo_path "${WEIGHTS:-ckpt/checkpoints.pth}" weights
resolve_repo_path "${RESULTS_ROOT:-eval_results/reproduce}" results_root
if [[ -n "${LONG_RESULTS_ROOT:-}" ]]; then
    resolve_repo_path "${LONG_RESULTS_ROOT}" output_root
else
    output_root="${results_root}/long_sequence"
fi
resolve_repo_path "${LONG_DATA_ROOT:-${DATA_ROOT:-data/eval/stage4c_tum}}" data_root
require_file "${weights}" "checkpoint"
runtime_preflight

unexpected_failures=0
evaluated_cells=()
expected_run_ids=()
for method in "${methods[@]}"; do
    for sequence in "${sequences[@]}"; do
        for frames in "${lengths[@]}"; do
            output_dir="${output_root}/${method}/${sequence}/${frames}"
            run_id="${BASHPID}-${RANDOM}-$(date +%s%N)"
            args=(
                "${PYTHON_BIN}" "${SRC_DIR}/eval/long_sequence/eval_stage4c_tum_long.py"
                --weights "${weights}"
                --data-root "${data_root}"
                --sequence "${sequence}"
                --output-dir "${output_dir}"
                --method "${method}"
                --max-frames "${frames}"
                --size "${size}"
                --max-association-difference "${MAX_ASSOC_DIFF:-0.02}"
                --run-scope "${run_scope}"
                --run-id "${run_id}"
            )
            note "long sequence ${sequence}/${frames}: ${method}"
            evaluated_cells+=("${method}|${sequence}|${frames}")
            expected_run_ids+=("${method}|${sequence}|${frames}|${run_id}")
            if run_in_dir "${REPO_ROOT}" "${args[@]}"; then
                metrics_path="${output_dir}/stage4c_metrics.json"
                if [[ "${DRY_RUN}" == "1" ]] || {
                    [[ -f "${metrics_path}" ]] &&
                    grep -Fq "\"run_id\": \"${run_id}\"" "${metrics_path}"
                }; then
                    continue
                fi
                unexpected_failures=$((unexpected_failures + 1))
                echo "missing current-run metrics: ${method}/${sequence}/${frames}" >&2
                continue
            fi
            if [[ "${method}" == "full_cache" ]]; then
                metrics_path="${output_dir}/stage4c_metrics.json"
                if [[ -f "${metrics_path}" ]] && \
                    grep -Fq "\"run_id\": \"${run_id}\"" "${metrics_path}" && \
                    grep -Eq \
                    '"error": ".*(OutOfMemoryError|CUDA out of memory)' \
                    "${metrics_path}"; then
                    note "Full cache reached its GPU-memory ceiling at ${sequence}/${frames}; skipping longer prefixes for this sequence."
                    break
                fi
                unexpected_failures=$((unexpected_failures + 1))
                echo "unexpected Full-cache failure: ${method}/${sequence}/${frames}" >&2
                break
            fi
            unexpected_failures=$((unexpected_failures + 1))
            echo "bounded method failure: ${method}/${sequence}/${frames}" >&2
        done
    done
done

summary_args=(
    "${PYTHON_BIN}" "${REPO_ROOT}/scripts/summarize_stage4c.py"
    --results-root "${output_root}"
    --output "${output_root}/long_sequence_results.csv"
    --require-consistent-provenance
    --require-pose-success
    --expected-run-scope "${run_scope}"
)
for cell in "${evaluated_cells[@]}"; do
    summary_args+=(--include-cell "${cell}")
done
for expected_run_id in "${expected_run_ids[@]}"; do
    summary_args+=(--expected-run-id "${expected_run_id}")
done
run_command "${summary_args[@]}"

if (( unexpected_failures > 0 )); then
    die "${unexpected_failures} unexpected long-sequence cell(s) failed; inspect stage4c_metrics.json files"
fi
