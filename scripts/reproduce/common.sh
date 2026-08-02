#!/usr/bin/env bash

# Shared helpers for the public reproduction entry points. This file is meant
# to be sourced, not submitted to a scheduler or executed as an experiment.

REPRODUCE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
REPO_ROOT="$(cd "${REPRODUCE_DIR}/../.." && pwd -P)"
SRC_DIR="${REPO_ROOT}/src"

readonly REPRODUCE_DIR REPO_ROOT SRC_DIR
readonly DEFAULT_METHODS_STRING="full_cache anchor_recent_dino_diverse_k4 anchor_recent_dino_diverse_k6 anchor_recent_dino_diverse_k8"

die() {
    echo "error: $*" >&2
    exit 2
}

note() {
    echo "[reproduce] $*"
}

only_help_or_no_args() {
    local usage_function="$1"
    shift
    if (( $# == 0 )); then
        return
    fi
    if (( $# == 1 )) && [[ "$1" == "-h" || "$1" == "--help" ]]; then
        "${usage_function}"
        exit 0
    fi
    "${usage_function}" >&2
    die "this runner is configured with environment variables; unexpected argument: $1"
}

validate_switch() {
    local name="$1"
    local value="$2"
    case "${value}" in
        0|1) ;;
        *) die "${name} must be 0 or 1 (got ${value@Q})" ;;
    esac
}

DRY_RUN="${DRY_RUN:-0}"
validate_switch DRY_RUN "${DRY_RUN}"

split_words() {
    local value="$1"
    local output_name="$2"
    local -n output_ref="${output_name}"
    output_ref=()
    if [[ -n "${value//[[:space:]]/}" ]]; then
        read -r -a output_ref <<< "${value}"
    fi
}

require_nonempty_array() {
    local name="$1"
    local -n values_ref="$2"
    (( ${#values_ref[@]} > 0 )) || die "${name} must contain at least one value"
}

resolve_repo_path() {
    local value="$1"
    local output_name="$2"
    local -n output_ref="${output_name}"
    if [[ "${value}" == /* ]]; then
        output_ref="${value}"
    else
        output_ref="${REPO_ROOT}/${value}"
    fi
}

resolve_executable() {
    local candidate="$1"
    local output_name="$2"
    local -n output_ref="${output_name}"
    local found=""
    if [[ "${candidate}" == */* ]]; then
        if [[ -x "${candidate}" ]]; then
            found="${candidate}"
        fi
    else
        found="$(command -v "${candidate}" 2>/dev/null || true)"
    fi
    if [[ -n "${found}" ]]; then
        output_ref="${found}"
    elif [[ "${DRY_RUN}" == "1" ]]; then
        output_ref="${candidate}"
    else
        die "executable not found: ${candidate}; activate the project environment or set an explicit override"
    fi
}

init_runtime() {
    local require_accelerate="${1:-1}"
    local python_candidate=""
    local accelerate_candidate=""

    if [[ -n "${PYTHON_BIN:-}" ]]; then
        python_candidate="${PYTHON_BIN}"
    elif [[ -n "${CONDA_PREFIX:-}" ]]; then
        python_candidate="${CONDA_PREFIX}/bin/python"
    elif command -v python3 >/dev/null 2>&1; then
        python_candidate="python3"
    else
        python_candidate="python"
    fi
    resolve_executable "${python_candidate}" PYTHON_BIN

    if [[ "${require_accelerate}" == "1" ]]; then
        if [[ -n "${ACCELERATE_BIN:-}" ]]; then
            accelerate_candidate="${ACCELERATE_BIN}"
        elif [[ "${PYTHON_BIN}" == */* && -x "$(dirname "${PYTHON_BIN}")/accelerate" ]]; then
            accelerate_candidate="$(dirname "${PYTHON_BIN}")/accelerate"
        elif [[ -n "${CONDA_PREFIX:-}" ]]; then
            accelerate_candidate="${CONDA_PREFIX}/bin/accelerate"
        else
            accelerate_candidate="accelerate"
        fi
        resolve_executable "${accelerate_candidate}" ACCELERATE_BIN
    fi
}

require_file() {
    local path="$1"
    local description="$2"
    if [[ "${DRY_RUN}" != "1" && ! -f "${path}" ]]; then
        die "missing ${description}: ${path}"
    fi
}

runtime_preflight() {
    if [[ "${DRY_RUN}" == "1" ]]; then
        return
    fi
    "${PYTHON_BIN}" -c \
        'import numpy, torch; assert torch.cuda.is_available(), "CUDA is not available"; print("Python:", __import__("sys").executable, "NumPy:", numpy.__version__, "Torch:", torch.__version__, "GPU:", torch.cuda.get_device_name(0))'
}

print_command() {
    printf ' +'
    printf ' %q' "$@"
    printf '\n'
}

run_command() {
    print_command "$@"
    if [[ "${DRY_RUN}" == "1" ]]; then
        return 0
    fi
    "$@"
}

run_in_dir() {
    local directory="$1"
    shift
    printf ' + cd %q &&' "${directory}"
    printf ' %q' "$@"
    printf '\n'
    if [[ "${DRY_RUN}" == "1" ]]; then
        return 0
    fi
    (
        cd "${directory}"
        "$@"
    )
}

validate_methods() {
    local -n methods_ref="$1"
    local method
    require_nonempty_array METHODS "$1"
    for method in "${methods_ref[@]}"; do
        case "${method}" in
            full_cache|anchor_recent_dino_diverse_k4|anchor_recent_dino_diverse_k6|anchor_recent_dino_diverse_k8) ;;
            *)
                die "unknown method ${method@Q}; use full_cache or anchor_recent_dino_diverse_k4/k6/k8"
                ;;
        esac
    done
}

method_config() {
    local method="$1"
    local window_name="$2"
    local policy_name="$3"
    local -n window_ref="${window_name}"
    local -n policy_ref="${policy_name}"
    case "${method}" in
        full_cache)
            window_ref=""
            policy_ref=""
            ;;
        anchor_recent_dino_diverse_k4)
            window_ref="4"
            policy_ref="anchor_recent_dino_diverse_k4"
            ;;
        anchor_recent_dino_diverse_k6)
            window_ref="6"
            policy_ref="anchor_recent_dino_diverse_k6"
            ;;
        anchor_recent_dino_diverse_k8)
            window_ref="8"
            policy_ref="anchor_recent_dino_diverse_k8"
            ;;
        *) die "unknown method: ${method}" ;;
    esac
}

append_cache_cli_args() {
    local output_name="$1"
    local method="$2"
    local -n output_ref="${output_name}"
    local window policy
    method_config "${method}" window policy
    if [[ -n "${window}" ]]; then
        output_ref+=(--cache-window "${window}" --cache-policy "${policy}")
    fi
}

common_usage() {
    cat <<'EOF'
Common environment variables:
  METHODS          Space-separated methods. Default: all four paper methods.
  WEIGHTS          Checkpoint path, relative to the repository or absolute.
  RESULTS_ROOT     Base output directory. Default: eval_results/reproduce.
  MAX_FRAMES       Optional frame cap (task-specific semantics).
  DRY_RUN          Set to 1 to print commands without checking GPU/data/weights.
  PYTHON_BIN       Python executable override.
  ACCELERATE_BIN   Accelerate executable override.

Public method names:
  full_cache
  anchor_recent_dino_diverse_k4
  anchor_recent_dino_diverse_k6
  anchor_recent_dino_diverse_k8
EOF
}
