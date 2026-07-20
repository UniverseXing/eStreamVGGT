#!/usr/bin/env python3
"""Validate MonST3R-prepared TUM-dynamics data for Stage 3.3C."""

import argparse
from bisect import bisect_left
from pathlib import Path

import numpy as np


EXPECTED_SEQUENCES = (
    "rgbd_dataset_freiburg3_sitting_halfsphere",
    "rgbd_dataset_freiburg3_sitting_rpy",
    "rgbd_dataset_freiburg3_sitting_static",
    "rgbd_dataset_freiburg3_sitting_xyz",
    "rgbd_dataset_freiburg3_walking_halfsphere",
    "rgbd_dataset_freiburg3_walking_rpy",
    "rgbd_dataset_freiburg3_walking_static",
    "rgbd_dataset_freiburg3_walking_xyz",
)


def read_index(path):
    records = []
    with path.open() as handle:
        for line in handle:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            fields = line.split()
            if len(fields) >= 2:
                records.append((float(fields[0]), fields[1]))
    return records


def count_associated(scene, max_delta):
    rgb_paths = sorted(
        path
        for path in (scene / "rgb_90").iterdir()
        if path.suffix.lower() in (".png", ".jpg", ".jpeg")
    )
    poses = np.atleast_2d(np.loadtxt(scene / "groundtruth_90.txt", comments="#"))
    if len(rgb_paths) != len(poses) or poses.shape[1] != 8:
        raise ValueError(
            f"{scene.name}: rgb_90/groundtruth_90 mismatch: "
            f"{len(rgb_paths)} vs {poses.shape}"
        )
    rgb_times = {Path(relpath).name: timestamp for timestamp, relpath in read_index(scene / "rgb.txt")}
    depths = sorted(read_index(scene / "depth.txt"))
    depth_times = [timestamp for timestamp, _ in depths]
    associated = 0
    for rgb_path in rgb_paths:
        timestamp = rgb_times.get(rgb_path.name)
        if timestamp is None:
            continue
        insertion = bisect_left(depth_times, timestamp)
        candidates = [index for index in (insertion - 1, insertion) if 0 <= index < len(depths)]
        if not candidates:
            continue
        index = min(candidates, key=lambda item: abs(depth_times[item] - timestamp))
        depth_path = scene / depths[index][1]
        if abs(depth_times[index] - timestamp) <= max_delta and depth_path.is_file():
            associated += 1
    return len(rgb_paths), associated


def main():
    parser = argparse.ArgumentParser("Validate Stage 3.3C TUM-dynamics data")
    parser.add_argument("--root", default="data/eval/tum")
    parser.add_argument("--frames", type=int, default=50)
    parser.add_argument("--max-association-delta", type=float, default=0.02)
    parser.add_argument("--allow-subset", action="store_true")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"missing TUM root: {root}")
    available = [name for name in EXPECTED_SEQUENCES if (root / name).is_dir()]
    if not args.allow_subset and len(available) != len(EXPECTED_SEQUENCES):
        missing = sorted(set(EXPECTED_SEQUENCES) - set(available))
        raise FileNotFoundError(f"missing TUM-dynamics sequences below {root}: {missing}")
    if not available:
        raise FileNotFoundError(f"no expected TUM-dynamics sequences below {root}")

    for name in available:
        scene = root / name
        required = ("rgb_90", "groundtruth_90.txt", "rgb.txt", "depth.txt", "depth")
        missing = [item for item in required if not (scene / item).exists()]
        if missing:
            raise FileNotFoundError(f"{name}: missing {missing}")
        rgb_count, associated = count_associated(scene, args.max_association_delta)
        if associated < args.frames:
            raise ValueError(
                f"{name}: only {associated}/{rgb_count} aligned RGB-depth frames; "
                f"need {args.frames}"
            )
        print(f"{name}: {associated}/{rgb_count} aligned RGB-depth-pose frames")
    print(f"Stage 3.3C data ready: {len(available)} sequence(s), {args.frames} frames each")


if __name__ == "__main__":
    main()
