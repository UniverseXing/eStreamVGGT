#!/usr/bin/env bash
#SBATCH --job-name=streamvggt-stage5a
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --output=streamvggt-stage5a-%j.out
#SBATCH --error=streamvggt-stage5a-%j.err

# Runs Stage 5A same-budget controls and component ablations.
set -euo pipefail


module load Miniforge3
conda init
source activate StreamVGGT

echo "Job ID: ${SLURM_JOB_ID:-unknown}"
echo "Node: $(hostname)"
echo "Started: $(date --iso-8601=seconds)"
bash "run_stage5a.sh"

echo "Finished: $(date --iso-8601=seconds)"
