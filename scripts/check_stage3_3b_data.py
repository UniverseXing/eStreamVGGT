#!/usr/bin/env python3
import argparse
from pathlib import Path

import numpy as np


def require(path, description, search_root=None):
    if path.exists():
        return
    message = f"missing {description}: {path}"
    if search_root is not None and search_root.exists():
        candidates = sorted(search_root.rglob(path.name))
        if candidates:
            found = "\n  ".join(str(candidate) for candidate in candidates[:10])
            message += (
                "\nA file/directory with the expected name exists elsewhere, but the "
                f"layout is wrong:\n  {found}"
            )
        elif path.is_symlink():
            message += f"\nThe expected path is a broken symlink -> {path.readlink()}"
    raise FileNotFoundError(message)


def check_7scenes(root):
    scenes = ("chess", "fire", "heads", "office", "pumpkin", "redkitchen", "stairs")
    sequences = projected = invalid_poses = 0
    for scene in scenes:
        scene_root = root / scene
        require(scene_root / "TestSplit.txt", f"7-Scenes {scene} test split")
        split_lines = (scene_root / "TestSplit.txt").read_text().splitlines()
        scene_sequences = []
        for line in split_lines:
            digits = "".join(character for character in line if character.isdigit())
            sequence = scene_root / f"seq-{digits.zfill(2)}"
            require(sequence, f"7-Scenes {scene} test sequence")
            scene_sequences.append(sequence)
        sequences += len(scene_sequences)
        for sequence in scene_sequences:
            projected += sum(1 for _ in sequence.glob("frame-*.depth.proj.png"))
            for pose_path in sequence.glob("frame-*.pose.txt"):
                try:
                    pose = np.loadtxt(pose_path)
                except (OSError, ValueError):
                    invalid_poses += 1
                    continue
                invalid_poses += int(pose.shape != (4, 4) or not np.isfinite(pose).all())
    if not projected:
        raise FileNotFoundError(f"no projected 7-Scenes depth maps below {root}")
    print(
        f"7scenes: {sequences} test sequences, {projected} projected depth maps, "
        f"{invalid_poses} invalid pose frames (evaluator will skip them)"
    )


def check_nrgbd(root):
    scenes = [path for path in root.iterdir() if path.is_dir()]
    if not scenes:
        raise FileNotFoundError(f"no Neural-RGBD scenes below {root}")
    for scene in scenes:
        require(scene / "images", f"Neural-RGBD {scene.name} images")
        require(scene / "depth", f"Neural-RGBD {scene.name} depth")
        require(scene / "poses.txt", f"Neural-RGBD {scene.name} poses")
    print(f"nrgbd: {len(scenes)} scenes")


def check_eth3d(root):
    scenes = [path for path in root.iterdir() if path.is_dir() and not path.name.startswith("_")]
    if not scenes:
        raise FileNotFoundError(f"no ETH3D scenes below {root}")
    for scene in scenes:
        require(
            scene / "dslr_calibration_jpg" / "cameras.txt",
            f"ETH3D {scene.name} cameras",
            scene,
        )
        require(
            scene / "dslr_calibration_jpg" / "images.txt",
            f"ETH3D {scene.name} poses",
            scene,
        )
        require(scene / "images" / "dslr_images", f"ETH3D {scene.name} images", scene)
        require(
            scene / "ground_truth_depth" / "dslr_images",
            f"ETH3D {scene.name} depth",
            scene,
        )
    print(f"eth3d: {len(scenes)} scenes")


def main():
    parser = argparse.ArgumentParser("Validate Stage 3.3B reconstruction data")
    parser.add_argument("--root", default="data/eval")
    args = parser.parse_args()
    root = Path(args.root).resolve()
    check_7scenes(root / "7scenes")
    check_nrgbd(root / "neural_rgbd")
    check_eth3d(root / "eth3d")
    print("Stage 3.3B dataset layout is ready")


if __name__ == "__main__":
    main()
