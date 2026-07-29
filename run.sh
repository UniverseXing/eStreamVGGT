#!/usr/bin/env bash
#SBATCH --job-name=streamvggt-stage4d
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=streamvggt-stage4d-%j.out
#SBATCH --error=streamvggt-stage4d-%j.err

# Runs Stage 4D frozen paper-asset generation and audit.
set -euo pipefail


module load Miniforge3
conda init
source activate StreamVGGT

echo "Job ID: ${SLURM_JOB_ID:-unknown}"
echo "Node: $(hostname)"
echo "Started: $(date --iso-8601=seconds)"
bash "run_stage4d.sh"

echo "Finished: $(date --iso-8601=seconds)"
