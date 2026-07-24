#!/usr/bin/env python3
"""Prepare the 13-drive, first-110-frame KITTI VideoDepth protocol."""

import argparse
import json
import os
import shutil
from pathlib import Path


def materialize(source, target, mode):
    if target.exists() or target.is_symlink():
        if target.resolve() == source.resolve() or target.stat().st_size == source.stat().st_size:
            return
        raise RuntimeError(f"existing target differs from source: {target}")
    if mode == "hardlink":
        try:
            os.link(source, target)
            return
        except OSError as error:
            print(f"Hardlink unavailable for {source}: {error}; falling back to copy")
            shutil.copy2(source, target)
    elif mode == "symlink":
        target.symlink_to(source.resolve())
    else:
        shutil.copy2(source, target)


def main():
    parser = argparse.ArgumentParser("Prepare Stage 4A KITTI VideoDepth data")
    parser.add_argument("--root", type=Path, default=Path("data/eval/kitti"))
    parser.add_argument("--frames-per-drive", type=int, default=110)
    parser.add_argument("--mode", choices=("hardlink", "symlink", "copy"), default="hardlink")
    args = parser.parse_args()

    root = args.root.resolve()
    depth_sources = sorted(root.glob("val/*/proj_depth/groundtruth/image_02"))
    if len(depth_sources) != 13:
        raise RuntimeError(
            f"expected 13 KITTI validation drives under {root / 'val'}, "
            f"found {len(depth_sources)}"
        )

    output_root = root / "depth_selection" / "val_selection_cropped"
    image_root = output_root / "image_gathered"
    depth_root = output_root / "groundtruth_depth_gathered"
    image_root.mkdir(parents=True, exist_ok=True)
    depth_root.mkdir(parents=True, exist_ok=True)

    manifest = {
        "protocol": "MonST3R KITTI VideoDepth first-110 validation frames",
        "frames_per_drive_limit": args.frames_per_drive,
        "materialization_mode": args.mode,
        "drives": [],
    }
    for depth_source in depth_sources:
        drive = depth_source.parents[2].name
        date = "_".join(drive.split("_")[:3])
        image_source = root / date / drive / "image_02" / "data"
        if not image_source.is_dir():
            raise FileNotFoundError(f"missing raw KITTI images: {image_source}")

        group = f"{drive}_02"
        output_images = image_root / group
        output_depths = depth_root / group
        output_images.mkdir(parents=True, exist_ok=True)
        output_depths.mkdir(parents=True, exist_ok=True)

        depth_files = sorted(depth_source.glob("*.png"))[: args.frames_per_drive]
        if not depth_files:
            raise RuntimeError(f"no depth frames found in {depth_source}")
        missing_images = []
        for depth_file in depth_files:
            image_file = image_source / depth_file.name
            if not image_file.is_file():
                missing_images.append(str(image_file))
                continue
            materialize(image_file, output_images / image_file.name, args.mode)
            materialize(depth_file, output_depths / depth_file.name, args.mode)
        if missing_images:
            raise FileNotFoundError(
                f"{drive} is missing {len(missing_images)} RGB frames; "
                f"first example: {missing_images[0]}"
            )
        manifest["drives"].append(
            {"drive": drive, "group": group, "num_frames": len(depth_files)}
        )
        print(f"{group}: {len(depth_files)} paired frames")

    manifest["num_drives"] = len(manifest["drives"])
    manifest["total_frames"] = sum(item["num_frames"] for item in manifest["drives"])
    manifest_path = output_root / "stage4a_manifest.json"
    with open(manifest_path, "w") as handle:
        json.dump(manifest, handle, indent=2)
    print(f"Prepared {manifest['total_frames']} frames across 13 drives")
    print(f"Wrote {manifest_path}")


if __name__ == "__main__":
    main()
