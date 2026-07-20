#!/usr/bin/env bash
#SBATCH --job-name=streamvggt-stage3-3c-recon
#SBATCH --gpus=pro6000:1
#SBATCH --time=12:00:00
#SBATCH --output=streamvggt-stage3-3c-recon-pro6000-%j.out
#SBATCH --error=streamvggt-stage3-3c-recon-pro6000-%j.err

set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${repo_root}"

module load Miniforge3
conda activate StreamVGGT

echo "Job ID: ${SLURM_JOB_ID:-unknown}"
echo "Node: $(hostname)"
echo "Started: $(date --iso-8601=seconds)"
nvidia-smi

python scripts/check_stage3_3c_data.py
bash "${repo_root}/run_stage3_3c_recon.sh"

echo "Finished: $(date --iso-8601=seconds)"
