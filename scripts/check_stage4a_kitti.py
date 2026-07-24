#!/usr/bin/env python3
"""Validate the prepared KITTI layout before allocating a GPU job."""

import argparse
import json
from pathlib import Path

from PIL import Image


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
    if len(image_groups) != 13 or manifest.get("num_drives") != 13:
        raise RuntimeError(
            f"expected 13 prepared drives, found {len(image_groups)} "
            f"(manifest={manifest.get('num_drives')})"
        )

    total_frames = 0
    for group in sorted(image_groups):
        images = sorted(path.name for path in image_groups[group].glob("*.png"))
        depths = sorted(path.name for path in depth_groups[group].glob("*.png"))
        if images != depths:
            raise RuntimeError(f"frame-name mismatch in {group}")
        if not 1 <= len(images) <= 110:
            raise RuntimeError(f"invalid frame count for {group}: {len(images)}")
        with Image.open(depth_groups[group] / depths[0]) as depth:
            extrema = depth.getextrema()
            maximum = extrema[1] if isinstance(extrema, tuple) else extrema
            if maximum <= 255:
                raise RuntimeError(f"depth map is not 16-bit KITTI depth: {group}/{depths[0]}")
        total_frames += len(images)
        print(f"{group}: {len(images)} paired frames")

    if total_frames != manifest.get("total_frames"):
        raise RuntimeError(
            f"manifest total {manifest.get('total_frames')} != prepared total {total_frames}"
        )
    print(f"KITTI Stage 4A data OK: 13 drives, {total_frames} paired frames")


if __name__ == "__main__":
    main()
