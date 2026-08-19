#!/usr/bin/env python3
"""Run one small, bounded-cache inference on the bundled example images."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch


METHODS = {
    "anchor_recent_dino_diverse_k4": 4,
    "anchor_recent_dino_diverse_k6": 6,
    "anchor_recent_dino_diverse_k8": 8,
    "anchor_uniform_k4": 4,
    "random_reservoir_k4": 4,
    "dino_diverse_no_anchor_k4": 4,
    "anchor_dino_diverse_no_recent_k6": 6,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--images-dir", type=Path, required=True)
    parser.add_argument("--method", choices=tuple(METHODS), default="anchor_recent_dino_diverse_k4")
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--random-seed", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = args.repo_root.resolve()
    sys.path.insert(0, str(repo_root / "src"))

    from streamvggt.models.streamvggt import StreamVGGT
    from streamvggt.utils.load_fn import load_and_preprocess_images

    suffixes = {".jpg", ".jpeg", ".png"}
    image_paths = sorted(
        path for path in args.images_dir.iterdir() if path.suffix.lower() in suffixes
    )
    if args.max_frames is not None:
        image_paths = image_paths[: args.max_frames]
    window = METHODS[args.method]
    if len(image_paths) <= window:
        raise ValueError(
            f"the smoke test needs at least {window + 1} images to exercise "
            f"{args.method} pruning, but found {len(image_paths)}"
        )
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for StreamVGGT inference")

    device = torch.device("cuda")
    images = load_and_preprocess_images([str(path) for path in image_paths]).to(device)
    frames = [{"img": image.unsqueeze(0)} for image in images]
    model = StreamVGGT()
    checkpoint = torch.load(args.weights, map_location=device)
    model.load_state_dict(checkpoint, strict=True)
    model.eval().to(device)
    del checkpoint

    dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    with torch.no_grad(), torch.cuda.amp.autocast(dtype=dtype):
        output = model.inference(
            frames,
            cache_window_size=window,
            cache_policy=args.method,
            cache_random_seed=args.random_seed,
            return_memory_events=True,
            return_memory_trace=True,
        )
    if len(output.ress) != len(image_paths):
        raise RuntimeError(f"received {len(output.ress)} predictions for {len(image_paths)} inputs")
    if not output.memory_events:
        raise RuntimeError("bounded-cache pruning was not exercised")
    retained = output.memory_trace[-1]["retained_frame_ids"]
    if len(retained) != window:
        raise RuntimeError(f"expected {window} retained frames, got {retained}")

    summary = {
        "status": "ok",
        "method": args.method,
        "random_seed": args.random_seed,
        "input_frames": len(image_paths),
        "retained_frame_ids": retained,
        "depth_shape": list(output.ress[-1]["depth"].shape),
        "camera_pose_shape": list(output.ress[-1]["camera_pose"].shape),
        "gpu": torch.cuda.get_device_name(0),
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
