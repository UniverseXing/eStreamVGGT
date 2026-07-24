#!/usr/bin/env bash

# Download the KITTI sources used by the official MonST3R video-depth protocol.
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
root="${STREAMVGGT_KITTI_ROOT:-${repo_root}/data/eval/kitti}"
downloads="${root}/downloads"
base="https://s3.eu-central-1.amazonaws.com/avg-kitti"

mkdir -p "${root}" "${downloads}"

download_and_extract() {
    local url="$1"
    local archive="${downloads}/$(basename "${url}")"
    echo "Downloading ${url}"
    wget -c "${url}" -O "${archive}"
    echo "Extracting ${archive} into ${root}"
    unzip -q -o "${archive}" -d "${root}"
}

# Annotated depths provide val/<drive>/proj_depth/groundtruth/image_02.
download_and_extract "${base}/data_depth_annotated.zip"

drives=(
    2011_09_26_drive_0002
    2011_09_26_drive_0005
    2011_09_26_drive_0013
    2011_09_26_drive_0020
    2011_09_26_drive_0023
    2011_09_26_drive_0036
    2011_09_26_drive_0079
    2011_09_26_drive_0095
    2011_09_26_drive_0113
    2011_09_28_drive_0037
    2011_09_29_drive_0026
    2011_09_30_drive_0016
    2011_10_03_drive_0047
)

for drive in "${drives[@]}"; do
    date="${drive:0:10}"
    download_and_extract "${base}/raw_data/${drive}/${drive}_sync.zip"
    if [[ ! -d "${root}/${date}/${drive}_sync/image_02/data" ]]; then
        echo "Missing extracted RGB directory for ${drive}" >&2
        exit 2
    fi
done

echo "KITTI sources downloaded under ${root}"
echo "Archives are retained in ${downloads} for resumability."
echo "Next: python scripts/prepare_stage4a_kitti.py --root ${root}"
