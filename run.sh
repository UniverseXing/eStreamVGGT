#!/usr/bin/env bash
#SBATCH --job-name=streamvggt-stage5
#SBATCH --gpus=6000ada:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=streamvggt-stage5-%j.out
#SBATCH --error=streamvggt-stage5-%j.err

# Runs the conference Stage 5 experiments and the optional emergency Stage 5E
# direct-baseline comparison.
set -euo pipefail

module load Miniforge3
conda init
source activate StreamVGGT

echo "Job ID: ${SLURM_JOB_ID:-unknown}"
echo "Node: $(hostname)"
echo "Started: $(date --iso-8601=seconds)"
nvidia-smi
case "${STREAMVGGT_RUN_TARGET:-stage5}" in
    stage5)
        bash "run_stage5.sh"
        ;;
    qualitative)
        "${CONDA_PREFIX}/bin/python" "scripts/reproduce/run_qualitative_figure.py" \
            --repo-root "$(pwd)" \
            --weights "${STREAMVGGT_QUAL_WEIGHTS:-$(pwd)/ckpt/checkpoints.pth}" \
            --images-dir "${STREAMVGGT_QUAL_IMAGES_DIR:-$(pwd)/data/eval/7scenes/chess/seq-01}" \
            --sequence "${STREAMVGGT_QUAL_SEQUENCE:-chess_seq01}" \
            --image-glob "${STREAMVGGT_QUAL_IMAGE_GLOB:-*.color.png}" \
            --sampling-stride "${STREAMVGGT_QUAL_SAMPLING_STRIDE:-5}" \
            --max-frames "${STREAMVGGT_QUAL_MAX_FRAMES:-110}" \
            --frame "${STREAMVGGT_QUAL_FRAME:-110}" \
            --output-dir "${STREAMVGGT_QUAL_OUTPUT_DIR:-$(pwd)/paper_assets/qualitative/7scenes_chess_seq01_v110}"
        ;;
    supplementary)
        bash "run_supplementary.sh"
        ;;
    stage5e)
        bash "run_stage5e.sh"
        ;;
    *)
        echo "Unknown STREAMVGGT_RUN_TARGET=${STREAMVGGT_RUN_TARGET}" >&2
        exit 2
        ;;
esac

echo "Finished: $(date --iso-8601=seconds)"
