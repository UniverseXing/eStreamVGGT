#!/usr/bin/env bash
#SBATCH --job-name=streamvggt-stage5
#SBATCH --gpus=6000ada:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=streamvggt-stage5-%j.out
#SBATCH --error=streamvggt-stage5-%j.err

# Runs the two conference Stage 5 evidence-completion experiments.
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${repo_root}"

module load Miniforge3
conda init
source activate StreamVGGT

echo "Job ID: ${SLURM_JOB_ID:-unknown}"
echo "Node: $(hostname)"
echo "Started: $(date --iso-8601=seconds)"
nvidia-smi
case "${STREAMVGGT_RUN_TARGET:-stage5}" in
    stage5)
        bash "run_stage5.sh"
        ;;
    qualitative)
        "${CONDA_PREFIX}/bin/python" "scripts/reproduce/run_qualitative_figure.py" \
            --repo-root "${repo_root}" \
            --weights "${STREAMVGGT_QUAL_WEIGHTS:-${repo_root}/ckpt/checkpoints.pth}" \
            --images-dir "${STREAMVGGT_QUAL_IMAGES_DIR:-${repo_root}/data/eval/bonn/rgbd_bonn_dataset/rgbd_bonn_person_tracking2/rgb_110_sampled}" \
            --sequence "${STREAMVGGT_QUAL_SEQUENCE:-person_tracking2}" \
            --max-frames "${STREAMVGGT_QUAL_MAX_FRAMES:-110}" \
            --frame "${STREAMVGGT_QUAL_FRAME:-110}" \
            --output-dir "${STREAMVGGT_QUAL_OUTPUT_DIR:-${repo_root}/paper_assets/qualitative/bonn_person_tracking2_f110}"
        ;;
    *)
        echo "Unknown STREAMVGGT_RUN_TARGET=${STREAMVGGT_RUN_TARGET}" >&2
        exit 2
        ;;
esac

echo "Finished: $(date --iso-8601=seconds)"
