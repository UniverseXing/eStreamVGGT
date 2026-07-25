#!/usr/bin/env bash

# Download the three frozen, previously unused TUM RGB-D long sequences.
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
root="${STREAMVGGT_STAGE4C_DATA_ROOT:-${repo_root}/data/eval/stage4c_tum}"
downloads="${root}/downloads"
base="https://cvg.cit.tum.de/rgbd/dataset"

mkdir -p "${root}" "${downloads}"

sequences=(
    freiburg1/rgbd_dataset_freiburg1_room
    freiburg2/rgbd_dataset_freiburg2_desk
    freiburg3/rgbd_dataset_freiburg3_long_office_household
)

for item in "${sequences[@]}"; do
    sequence="$(basename "${item}")"
    archive="${downloads}/${sequence}.tgz"
    wget -c "${base}/${item}.tgz" -O "${archive}"
    # Stage 4C evaluates RGB and mocap poses only. Do not extract the unused
    # depth stream, which roughly doubles the working dataset footprint.
    tar -xzf "${archive}" -C "${root}" \
        "${sequence}/rgb" \
        "${sequence}/rgb.txt" \
        "${sequence}/groundtruth.txt"
    if [[ "${STREAMVGGT_STAGE4C_DELETE_ARCHIVES:-0}" == "1" ]]; then
        rm -f "${archive}"
    fi
done

echo "Stage 4C TUM RGB-D sequences downloaded under ${root}"
if [[ "${STREAMVGGT_STAGE4C_DELETE_ARCHIVES:-0}" != "1" ]]; then
    echo "Archives are retained under ${downloads} for resumability."
fi
