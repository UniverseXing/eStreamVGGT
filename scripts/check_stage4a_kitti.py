#!/usr/bin/env python3
"""Validate the prepared KITTI layout before allocating a GPU job."""

import argparse
import json
from pathlib import Path

from PIL import Image


EXPECTED_FRAMES = {
    "2011_09_26_drive_0002_sync_02": 67,
    "2011_09_26_drive_0005_sync_02": 110,
    "2011_09_26_drive_0013_sync_02": 110,
    "2011_09_26_drive_0020_sync_02": 76,
    "2011_09_26_drive_0023_sync_02": 110,
    "2011_09_26_drive_0036_sync_02": 110,
    "2011_09_26_drive_0079_sync_02": 90,
    "2011_09_26_drive_0095_sync_02": 110,
    "2011_09_26_drive_0113_sync_02": 77,
    "2011_09_28_drive_0037_sync_02": 79,
    "2011_09_29_drive_0026_sync_02": 110,
    "2011_09_30_drive_0016_sync_02": 110,
    "2011_10_03_drive_0047_sync_02": 110,
}
EXPECTED_TOTAL_FRAMES = sum(EXPECTED_FRAMES.values())


def main():
    parser = argparse.ArgumentParser("Check Stage 4A KITTI data")
    parser.add_argument("--root", type=Path, default=Path("data/eval/kitti"))
    args = parser.parse_args()

    root = args.root.resolve() / "depth_selection" / "val_selection_cropped"
    image_root = root / "image_gathered"
    depth_root = root / "groundtruth_depth_gathered"
    manifest_path = root / "stage4a_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"missing preparation manifest: {manifest_path}")
    with open(manifest_path) as handle:
        manifest = json.load(handle)

    image_groups = {path.name: path for path in image_root.iterdir() if path.is_dir()}
    depth_groups = {path.name: path for path in depth_root.iterdir() if path.is_dir()}
    if set(image_groups) != set(depth_groups):
        raise RuntimeError(
            f"image/depth groups differ: images-only={sorted(set(image_groups)-set(depth_groups))}, "
            f"depth-only={sorted(set(depth_groups)-set(image_groups))}"
        )
    if set(image_groups) != set(EXPECTED_FRAMES) or manifest.get("num_drives") != 13:
        raise RuntimeError(
            "prepared drives do not match the frozen 13-drive protocol: "
            f"missing={sorted(set(EXPECTED_FRAMES) - set(image_groups))}, "
            f"extra={sorted(set(image_groups) - set(EXPECTED_FRAMES))}, "
            f"manifest={manifest.get('num_drives')}"
        )
    if manifest.get("frames_per_drive_limit") != 110:
        raise RuntimeError(
            "the frozen KITTI protocol requires manifest frames_per_drive_limit=110"
        )

    manifest_drives = manifest.get("drives")
    if not isinstance(manifest_drives, list):
        raise RuntimeError("manifest drives must be a list")
    manifest_counts = {
        item.get("group"): item.get("num_frames")
        for item in manifest_drives
        if isinstance(item, dict)
    }
    if manifest_counts != EXPECTED_FRAMES:
        raise RuntimeError(
            f"manifest per-drive counts do not match the frozen protocol: {manifest_counts}"
        )

    total_frames = 0
    for group in sorted(image_groups):
        images = sorted(path.name for path in image_groups[group].glob("*.png"))
        depths = sorted(path.name for path in depth_groups[group].glob("*.png"))
        if images != depths:
            raise RuntimeError(f"frame-name mismatch in {group}")
        if len(images) != EXPECTED_FRAMES[group]:
            raise RuntimeError(
                f"invalid frame count for {group}: expected {EXPECTED_FRAMES[group]}, "
                f"found {len(images)}"
            )
        with Image.open(depth_groups[group] / depths[0]) as depth:
            extrema = depth.getextrema()
            maximum = extrema[1] if isinstance(extrema, tuple) else extrema
            if maximum <= 255:
                raise RuntimeError(f"depth map is not 16-bit KITTI depth: {group}/{depths[0]}")
        total_frames += len(images)
        print(f"{group}: {len(images)} paired frames")

    if total_frames != EXPECTED_TOTAL_FRAMES or manifest.get("total_frames") != EXPECTED_TOTAL_FRAMES:
        raise RuntimeError(
            f"expected {EXPECTED_TOTAL_FRAMES} frames; manifest={manifest.get('total_frames')}, "
            f"prepared={total_frames}"
        )
    print(f"KITTI Stage 4A data OK: 13 drives, {total_frames} paired frames")


if __name__ == "__main__":
    main()
