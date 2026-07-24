#!/usr/bin/env bash
#SBATCH --job-name=streamvggt-stage4a
#SBATCH --gpus=6000ada:1
#SBATCH --time=24:00:00
#SBATCH --output=streamvggt-stage4a-6000ada-%j.out
#SBATCH --error=streamvggt-stage4a-6000ada-%j.err

# Runs Stage 4A KITTI/temporal-K8 VideoDepth completion.
set -euo pipefail


module load Miniforge3
conda init
source activate StreamVGGT

echo "Job ID: ${SLURM_JOB_ID:-unknown}"
echo "Node: $(hostname)"
echo "Started: $(date --iso-8601=seconds)"
nvidia-smi

bash "run_stage4a_video_depth.sh"

echo "Finished: $(date --iso-8601=seconds)"
