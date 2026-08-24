#!/usr/bin/env bash
#SBATCH --job-name=streamvggt-stage5
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=streamvggt-stage5-%j.out
#SBATCH --error=streamvggt-stage5-%j.err

# Runs the two conference Stage 5 evidence-completion experiments.
set -euo pipefail


module load Miniforge3
conda init
source activate StreamVGGT

echo "Job ID: ${SLURM_JOB_ID:-unknown}"
echo "Node: $(hostname)"
echo "Started: $(date --iso-8601=seconds)"
bash "run_stage5.sh"

echo "Finished: $(date --iso-8601=seconds)"
