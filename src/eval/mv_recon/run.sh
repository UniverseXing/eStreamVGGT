#!/usr/bin/env bash

set -euo pipefail

export STREAMVGGT_MV_RUN_TAG="${STREAMVGGT_MV_RUN_TAG:-checkpoints}"
bash eval/mv_recon/run_streaming_recon.sh
