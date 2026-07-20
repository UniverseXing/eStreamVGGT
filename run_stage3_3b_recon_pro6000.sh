#!/usr/bin/env bash
#SBATCH --job-name=streamvggt-stage3-3b-recon
#SBATCH --gpus=pro6000:1
#SBATCH --time=12:00:00
#SBATCH --output=streamvggt-stage3-3b-recon-pro6000-%j.out
#SBATCH --error=streamvggt-stage3-3b-recon-pro6000-%j.err

set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${repo_root}"

module load Miniforge3
conda activate StreamVGGT

echo "Job ID: ${SLURM_JOB_ID:-unknown}"
echo "Node: $(hostname)"
echo "Started: $(date --iso-8601=seconds)"
nvidia-smi

bash "${repo_root}/run_stage3_3b_recon.sh"

echo "Finished: $(date --iso-8601=seconds)"
