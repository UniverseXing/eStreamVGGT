#!/usr/bin/env python3
"""Validate the Bonn and 7-Scenes inputs used by Stage 3.4."""

import argparse
from pathlib import Path

import numpy as np


BONN_SEQUENCES = (
    "balloon2",
    "crowd2",
    "crowd3",
    "person_tracking2",
    "synchronous",
)
SEVEN_SCENES = ("chess", "fire", "heads", "office", "pumpkin", "redkitchen", "stairs")
IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg")


def image_files(path):
    return sorted(item for item in path.iterdir() if item.suffix.lower() in IMAGE_SUFFIXES)


def prepared_dir(scene, kind):
    for dirname in (f"{kind}_110_sampled", f"{kind}_110", kind):
        path = scene / dirname
        if path.is_dir():
            return path
    raise FileNotFoundError(f"missing Bonn {kind} directory below {scene}")


def check_bonn(root, frames, allow_subset):
    available = [name for name in BONN_SEQUENCES if (root / f"rgbd_bonn_{name}").is_dir()]
    if not allow_subset and len(available) != len(BONN_SEQUENCES):
        missing = sorted(set(BONN_SEQUENCES) - set(available))
        raise FileNotFoundError(f"missing Bonn Stage 3.4 sequences: {missing}")
    if not available:
        raise FileNotFoundError(f"no Stage 3.4 Bonn sequences below {root}")
    for name in available:
        scene = root / f"rgbd_bonn_{name}"
        rgb = image_files(prepared_dir(scene, "rgb"))
        depth = image_files(prepared_dir(scene, "depth"))
        trajectory = scene / "groundtruth_110.txt"
        if not trajectory.is_file():
            trajectory = scene / "groundtruth.txt"
        if not trajectory.is_file():
            raise FileNotFoundError(f"{name}: missing ground-truth trajectory")
        poses = np.atleast_2d(np.loadtxt(trajectory, comments="#"))
        if len(rgb) < frames or len(depth) < frames:
            raise ValueError(
                f"{name}: only {len(rgb)} RGB/{len(depth)} depth frames; need {frames}"
            )
        if poses.shape[1] != 8 or not np.isfinite(poses).all():
            raise ValueError(f"{name}: invalid TUM trajectory {trajectory} ({poses.shape})")
        broken = [str(path) for path in rgb[:frames] + depth[:frames] if not path.is_file()]
        if broken:
            raise FileNotFoundError(f"{name}: broken sampled-frame links, first: {broken[0]}")
        print(
            f"Bonn {name}: {len(rgb)} RGB, {len(depth)} depth, "
            f"{len(poses)} trajectory records"
        )


def test_sequences(root):
    sequences = []
    for scene in SEVEN_SCENES:
        split = root / scene / "TestSplit.txt"
        if not split.is_file():
            continue
        with split.open() as handle:
            for line in handle:
                digits = "".join(character for character in line if character.isdigit())
                sequences.append(f"{scene}/seq-{digits.zfill(2)}")
    return sequences


def valid_pose_count(sequence):
    count = 0
    for color in sorted(sequence.glob("frame-*.color.png")):
        pose_path = color.with_name(color.name.replace(".color.png", ".pose.txt"))
        try:
            pose = np.loadtxt(pose_path)
        except (OSError, ValueError):
            continue
        count += int(pose.shape == (4, 4) and np.isfinite(pose).all())
    return count


def check_seven_scenes(root, forward_frames, allow_subset):
    available_scenes = [scene for scene in SEVEN_SCENES if (root / scene).is_dir()]
    if not allow_subset and len(available_scenes) != len(SEVEN_SCENES):
        missing = sorted(set(SEVEN_SCENES) - set(available_scenes))
        raise FileNotFoundError(f"missing 7-Scenes scenes: {missing}")
    sequences = test_sequences(root)
    if not sequences:
        raise FileNotFoundError(f"no official 7-Scenes test sequences below {root}")
    eligible = []
    excluded = []
    for name in sequences:
        count = valid_pose_count(root / name)
        (eligible if count >= forward_frames else excluded).append((name, count))
    for name, count in eligible:
        print(f"7-Scenes eligible {name}: {count} valid poses")
    for name, count in excluded:
        print(f"7-Scenes excluded {name}: {count} valid poses (<{forward_frames})")
    if not eligible:
        raise ValueError(f"no 7-Scenes sequence has {forward_frames} valid poses")
    print(
        f"7-Scenes loop ready: {len(eligible)}/{len(sequences)} test sequences eligible; "
        f"each run uses {forward_frames} forward + {forward_frames} reverse frames"
    )


def main():
    parser = argparse.ArgumentParser("Validate Stage 3.4 long-sequence data")
    parser.add_argument(
        "--bonn-root", default="data/eval/bonn/rgbd_bonn_dataset"
    )
    parser.add_argument("--seven-scenes-root", default="data/eval/7scenes")
    parser.add_argument("--bonn-frames", type=int, default=110)
    parser.add_argument("--loop-forward-frames", type=int, default=50)
    parser.add_argument("--parts", nargs="+", choices=("bonn", "7scenes_loop"), default=("bonn", "7scenes_loop"))
    parser.add_argument("--allow-subset", action="store_true")
    args = parser.parse_args()

    if "bonn" in args.parts:
        check_bonn(Path(args.bonn_root).resolve(), args.bonn_frames, args.allow_subset)
    if "7scenes_loop" in args.parts:
        check_seven_scenes(
            Path(args.seven_scenes_root).resolve(),
            args.loop_forward_frames,
            args.allow_subset,
        )


if __name__ == "__main__":
    main()
