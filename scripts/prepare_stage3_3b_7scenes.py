#!/usr/bin/env python3
"""Register the original 7-Scenes depth maps into the RGB camera.

This is a vectorized CLI version of SimpleRecon's 7scenes_preprocessing.py and
produces the frame-XXXXXX.depth.proj.png files expected by the evaluator.
"""

import argparse
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image


SCENES = ("chess", "fire", "heads", "office", "pumpkin", "redkitchen", "stairs")
RGB_FOCAL = 525.0
DEPTH_FOCAL = 585.0
RGB_WIDTH = 640
RGB_HEIGHT = 480
DEPTH_TO_RGB = np.array(
    [
        [0.9999651801256764, 0.0026765126460343, -0.0079041012313001, -0.0255589431781525],
        [-0.0027409311281317, 0.9999630280302759, -0.0081504520778013, 0.0001010963626806],
        [0.0078819942130445, 0.0081718328771891, 0.999935545540394, 0.0020318321729487],
        [0.0, 0.0, 0.0, 1.0],
    ],
    dtype=np.float64,
)


def project_depth(path, overwrite):
    source = Path(path)
    output = source.with_name(source.name.replace("depth.png", "depth.proj.png"))
    if output.exists() and not overwrite:
        return 0
    with Image.open(source) as depth_image:
        depth_mm = np.asarray(depth_image, dtype=np.uint16)
    depth = depth_mm.astype(np.float64) / 1000.0
    height, width = depth.shape
    yy, xx = np.indices((height, width), dtype=np.float64)
    valid = (depth > 0.0) & (depth < 100.0)

    z = depth[valid]
    eye = np.empty((4, len(z)), dtype=np.float64)
    eye[0] = ((xx[valid] + 0.5) - width / 2.0) / DEPTH_FOCAL * z
    eye[1] = ((yy[valid] + 0.5) - height / 2.0) / DEPTH_FOCAL * z
    eye[2] = z
    eye[3] = 1.0
    eye = DEPTH_TO_RGB @ eye

    projected_z = eye[2]
    projected_x = np.rint(eye[0] / projected_z * RGB_FOCAL + RGB_WIDTH / 2.0).astype(np.int64)
    projected_y = np.rint(eye[1] / projected_z * RGB_FOCAL + RGB_HEIGHT / 2.0).astype(np.int64)
    in_bounds = (
        (projected_x >= 0)
        & (projected_x < RGB_WIDTH)
        & (projected_y >= 0)
        & (projected_y < RGB_HEIGHT)
        & np.isfinite(projected_z)
    )

    registered = np.full((RGB_HEIGHT, RGB_WIDTH), 2000.0, dtype=np.float64)
    np.minimum.at(
        registered,
        (projected_y[in_bounds], projected_x[in_bounds]),
        projected_z[in_bounds],
    )
    registered[registered > 1000.0] = 0.0
    registered_mm = (registered * 1000.0).astype(np.uint16)
    Image.fromarray(registered_mm).save(output)
    return 1


def process_scene(task):
    scene_root, overwrite = task
    depth_files = sorted(Path(scene_root).glob("seq-*/frame-*.depth.png"))
    if not depth_files:
        raise RuntimeError(f"no original depth maps found below {scene_root}")
    written = sum(project_depth(path, overwrite) for path in depth_files)
    return Path(scene_root).name, len(depth_files), written


def main():
    parser = argparse.ArgumentParser("Prepare 7-Scenes projected depth for Stage 3.3B")
    parser.add_argument("--root", default="data/eval/7scenes")
    parser.add_argument("--workers", type=int, default=min(7, os.cpu_count() or 1))
    parser.add_argument("--scenes", nargs="+", choices=SCENES, default=SCENES)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    root = Path(args.root).resolve()
    missing = [scene for scene in args.scenes if not (root / scene).is_dir()]
    if missing:
        raise FileNotFoundError(f"missing 7-Scenes directories below {root}: {missing}")

    tasks = [(str(root / scene), args.overwrite) for scene in args.scenes]
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        for scene, total, written in executor.map(process_scene, tasks):
            print(f"{scene}: {written} generated, {total - written} already present")


if __name__ == "__main__":
    main()
