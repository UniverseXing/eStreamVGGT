#!/usr/bin/env bash
#SBATCH --job-name=streamvggt-stage4c
#SBATCH --gpus=6000ada:1
#SBATCH --time=24:00:00
#SBATCH --output=streamvggt-stage4c-6000ada-%j.out
#SBATCH --error=streamvggt-stage4c-6000ada-%j.err

# Runs Stage 4C frozen unseen long-sequence validation.
set -euo pipefail


module load Miniforge3
conda init
source activate StreamVGGT

echo "Job ID: ${SLURM_JOB_ID:-unknown}"
echo "Node: $(hostname)"
echo "Started: $(date --iso-8601=seconds)"
nvidia-smi

bash "run_stage4c.sh"

echo "Finished: $(date --iso-8601=seconds)"
