#!/usr/bin/env bash
#SBATCH --job-name=streamvggt-stage2
#SBATCH --gpus=pro6000:1
#SBATCH --time=06:00:00
#SBATCH --output=streamvggt-stage2-pro6000-%j.out
#SBATCH --error=streamvggt-stage2-pro6000-%j.err

# Runs the three interleaved Stage 2 timing repeats on one PRO 6000 GPU.
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"


cd "${repo_root}"

module load Miniforge3
conda activate StreamVGGT

echo "Job ID: ${SLURM_JOB_ID:-unknown}"
echo "Node: $(hostname)"
echo "Started: $(date --iso-8601=seconds)"
nvidia-smi

python - <<'PY'
import torch

if not torch.cuda.is_available():
    raise RuntimeError("Slurm allocated no CUDA device")

device = torch.cuda.get_device_name(0)
capability = torch.cuda.get_device_capability(0)
print(f"PyTorch: {torch.__version__}")
print(f"PyTorch CUDA: {torch.version.cuda}")
print(f"GPU: {device}")
print(f"Compute capability: sm_{capability[0]}{capability[1]}")
print(f"Compiled CUDA architectures: {torch.cuda.get_arch_list()}")
PY

bash "${repo_root}/run1.sh"
bash "${repo_root}/run2.sh"
bash "${repo_root}/run3.sh"

echo "Finished: $(date --iso-8601=seconds)"
