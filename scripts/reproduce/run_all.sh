#!/usr/bin/env bash

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
# shellcheck source=common.sh
source "${script_dir}/common.sh"

usage() {
    cat <<'EOF'
Usage: scripts/reproduce/run_all.sh

Run the public reproduction entry points in a fixed order. The default is the
full evaluation matrix; supplementary generation is opt-in because it consumes
the frozen paper summary bundle rather than newly produced raw directories.

Environment variables:
  TASKS                 Default: "video_depth pose reconstruction long_sequence".
                        Choices: smoke, video_depth, pose, reconstruction,
                        long_sequence, supplementary.
  METHODS               Passed to all evaluation runners.
  RESULTS_ROOT          Default: eval_results/reproduce.
  VIDEO_DEPTH_DATASETS  Default: "bonn sintel kitti".
  POSE_DATASETS         Default: "sintel scannet tum".
  RECON_DATASETS        Default: "7scenes nrgbd eth3d tum".
  DRY_RUN               Set to 1 to print the complete matrix without running it.

Task-specific variables documented by each child runner are also accepted.
For a quick environment check, use TASKS=smoke.
EOF
    common_usage
}

only_help_or_no_args usage "$@"
split_words "${TASKS:-video_depth pose reconstruction long_sequence}" tasks
require_nonempty_array TASKS tasks

# Even in dry-run mode the child must start: it performs no experiment but
# expands its full array-safe command matrix for inspection.
run_child() {
    print_command "$@"
    "$@"
}

for task in "${tasks[@]}"; do
    case "${task}" in
        smoke)
            run_child env \
                "DRY_RUN=${DRY_RUN}" \
                "WEIGHTS=${WEIGHTS:-ckpt/checkpoints.pth}" \
                "${script_dir}/smoke_test.sh"
            ;;
        video_depth)
            run_child env \
                "DRY_RUN=${DRY_RUN}" \
                "METHODS=${METHODS:-${DEFAULT_METHODS_STRING}}" \
                "DATASETS=${VIDEO_DEPTH_DATASETS:-bonn sintel kitti}" \
                "WEIGHTS=${WEIGHTS:-ckpt/checkpoints.pth}" \
                "RESULTS_ROOT=${RESULTS_ROOT:-eval_results/reproduce}" \
                "${script_dir}/run_video_depth.sh"
            ;;
        pose)
            run_child env \
                "DRY_RUN=${DRY_RUN}" \
                "METHODS=${METHODS:-${DEFAULT_METHODS_STRING}}" \
                "DATASETS=${POSE_DATASETS:-sintel scannet tum}" \
                "WEIGHTS=${WEIGHTS:-ckpt/checkpoints.pth}" \
                "RESULTS_ROOT=${RESULTS_ROOT:-eval_results/reproduce}" \
                "${script_dir}/run_pose.sh"
            ;;
        reconstruction)
            run_child env \
                "DRY_RUN=${DRY_RUN}" \
                "METHODS=${METHODS:-${DEFAULT_METHODS_STRING}}" \
                "DATASETS=${RECON_DATASETS:-7scenes nrgbd eth3d tum}" \
                "WEIGHTS=${WEIGHTS:-ckpt/checkpoints.pth}" \
                "RESULTS_ROOT=${RESULTS_ROOT:-eval_results/reproduce}" \
                "${script_dir}/run_reconstruction.sh"
            ;;
        long_sequence)
            run_child env \
                "DRY_RUN=${DRY_RUN}" \
                "METHODS=${METHODS:-${DEFAULT_METHODS_STRING}}" \
                "WEIGHTS=${WEIGHTS:-ckpt/checkpoints.pth}" \
                "RESULTS_ROOT=${RESULTS_ROOT:-eval_results/reproduce}" \
                "${script_dir}/run_long_sequence.sh"
            ;;
        supplementary)
            run_child env \
                "DRY_RUN=${DRY_RUN}" \
                "${script_dir}/build_supplementary.sh"
            ;;
        *) die "unknown TASKS entry: ${task}" ;;
    esac
done
