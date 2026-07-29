#!/usr/bin/env bash
#SBATCH --job-name=streamvggt-stage4e-a
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=streamvggt-stage4e-a-%j.out
#SBATCH --error=streamvggt-stage4e-a-%j.err

# Runs Stage 4E-A offline K4/K8 pose-composability screening.
set -euo pipefail


module load Miniforge3
conda init
source activate StreamVGGT

echo "Job ID: ${SLURM_JOB_ID:-unknown}"
echo "Node: $(hostname)"
echo "Started: $(date --iso-8601=seconds)"
bash "run_stage4e_a.sh"

echo "Finished: $(date --iso-8601=seconds)"
