#!/usr/bin/env bash
#SBATCH --job-name=streamvggt-stage3.3
#SBATCH --gpus=6000ada:1
#SBATCH --time=12:00:00
#SBATCH --output=streamvggt-stage3-6000ada-%j.out
#SBATCH --error=streamvggt-stage3-6000ada-%j.err

# Runs the three interleaved Stage 3 timing repeats on one 6000ada GPU.
set -euo pipefail


module load Miniforge3
conda init
source activate StreamVGGT

echo "Job ID: ${SLURM_JOB_ID:-unknown}"
echo "Node: $(hostname)"
echo "Started: $(date --iso-8601=seconds)"
nvidia-smi

bash "run_stage3_3c_recon.sh"

echo "Finished: $(date --iso-8601=seconds)"