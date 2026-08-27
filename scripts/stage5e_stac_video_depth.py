#!/usr/bin/env python3
"""Run the official STAC StreamVGGT adapter with project-matched inputs.

This file deliberately lives outside the external STAC checkout.  It uses the
official model wrapper and streaming implementation without patching upstream,
while recording the end-to-end VideoDepth coverage, model-only runtime, and
CUDA peaks required by the Stage 5E comparison.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import time
import traceback
from pathlib import Path


BONN_SEQUENCES = (
    "balloon2",
    "crowd2",
    "crowd3",
    "person_tracking2",
    "synchronous",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--stac-root", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--dataset", choices=("bonn", "sintel", "kitti"), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seq-list", nargs="+")
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--size", type=int, default=518)
    parser.add_argument("--mode", choices=("stac", "full"), default="stac")
    parser.add_argument("--backend", choices=("cuda", "portable"), default="cuda")
    parser.add_argument("--allow-failures", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    return parser.parse_args()


def git_commit(path: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(path), "rev-parse", "HEAD"], text=True
        ).strip()
    except Exception:
        return "unknown"


def bonn_frame_dir(root: Path, sequence: str, kind: str) -> Path:
    sequence_root = root / "bonn/rgbd_bonn_dataset" / f"rgbd_bonn_{sequence}"
    for name in (f"{kind}_110_sampled", f"{kind}_110", kind):
        candidate = sequence_root / name
        if candidate.is_dir():
            return candidate
    return sequence_root / f"{kind}_110"


def dataset_sequences(data_root: Path, dataset: str, requested: list[str] | None):
    if requested:
        return tuple(requested)
    if dataset == "bonn":
        return BONN_SEQUENCES
    if dataset == "sintel":
        return tuple(
            path.name
            for path in sorted((data_root / "sintel/training/final").iterdir())
            if path.is_dir()
        )
    image_root = data_root / "kitti/depth_selection/val_selection_cropped/image_gathered"
    return tuple(path.name for path in sorted(image_root.iterdir()) if path.is_dir())


def frame_paths(data_root: Path, dataset: str, sequence: str) -> list[Path]:
    if dataset == "bonn":
        directory = bonn_frame_dir(data_root, sequence, "rgb")
    elif dataset == "sintel":
        directory = data_root / "sintel/training/final" / sequence
    else:
        directory = (
            data_root
            / "kitti/depth_selection/val_selection_cropped/image_gathered"
            / sequence
        )
    paths = sorted((*directory.glob("*.png"), *directory.glob("*.jpg")))
    if not paths:
        raise FileNotFoundError(f"no RGB frames for {dataset}/{sequence}: {directory}")
    return paths


def normalized_depth_tensor(torch, value, expected_frames: int):
    depth = value.detach().float().cpu() if torch.is_tensor(value) else torch.as_tensor(value)
    while depth.ndim > 3 and depth.shape[0] == 1:
        depth = depth.squeeze(0)
    if depth.ndim == 4 and depth.shape[-1] == 1:
        depth = depth.squeeze(-1)
    if depth.ndim == 4 and depth.shape[1] == 1:
        depth = depth.squeeze(1)
    if depth.ndim == 2 and expected_frames == 1:
        depth = depth.unsqueeze(0)
    if depth.ndim != 3 or depth.shape[0] != expected_frames:
        raise RuntimeError(
            f"unexpected STAC depth shape {tuple(depth.shape)} for {expected_frames} frames"
        )
    return depth


def write_depth_maps(np, torch, depth, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    for index in range(depth.shape[0]):
        np.save(output_dir / f"frame_{index:04d}.npy", depth[index].numpy())


def main() -> None:
    args = parse_args()
    repo_root = args.repo_root.resolve()
    stac_root = args.stac_root.resolve()
    weights = args.weights.resolve()
    output_dir = args.output_dir.resolve()
    if not (stac_root / "model_wrapper.py").is_file():
        raise FileNotFoundError(f"not an official STAC checkout: {stac_root}")
    if not weights.is_file():
        raise FileNotFoundError(weights)

    # STAC uses a top-level package named eval.  Put its checkout first and do
    # not add this project's src directory in the external environment.
    sys.path.insert(0, str(stac_root))
    os.chdir(stac_root)
    import numpy as np
    import torch
    from eval.video_depth import launch as stac_launch

    if not torch.cuda.is_available():
        raise RuntimeError("Stage 5E STAC inference requires a CUDA GPU")
    device = torch.device("cuda")
    data_root = repo_root / "data/eval"
    sequences = dataset_sequences(data_root, args.dataset, args.seq_list)
    if not sequences:
        raise RuntimeError(f"no sequences selected for {args.dataset}")

    model = stac_launch.load_model(
        "causalvggt", "streamvggt", str(device), model_path=str(weights)
    )
    run_args = stac_launch.get_args_parser().parse_args([])
    run_args.device = str(device)
    run_args.model_name = "causalvggt"
    run_args.base_model = "streamvggt"
    run_args.mode = args.mode
    run_args.streaming = args.mode == "stac"
    run_args.use_cam_cache = False
    run_args.pinned = [0]
    run_args.chunk_size = 4 if args.mode == "stac" else 1
    run_args.window_size = 4 if args.mode == "stac" else 0
    run_args.hh_size = 2 if args.mode == "stac" else 0
    run_args.retrieval_size = 2 if args.mode == "stac" else 0
    run_args.retrieve_buf = args.mode == "stac"
    run_args.temperature = 0.9
    run_args.subsample = 1.0
    run_args.voxel_size = 0.05
    run_args.voxel_num = 4096
    run_args.voxel_conf = 2.0
    run_args.voxel_buf_cap = 8
    run_args.voxel_piv_cap = 4
    if args.backend == "cuda":
        run_args.attn_backend = "cuda"
        run_args.voxel_backend = "cuda"
        run_args.allocator = "segment"
    else:
        run_args.attn_backend = "triton"
        run_args.voxel_backend = "python"
        run_args.allocator = "slab"

    output_dir.mkdir(parents=True, exist_ok=True)
    sequence_rows = []
    for sequence in sequences:
        sequence_output = output_dir / sequence
        existing = sorted(sequence_output.glob("frame_*.npy"))
        paths = frame_paths(data_root, args.dataset, sequence)
        if args.max_frames is not None:
            paths = paths[: args.max_frames]
        if args.skip_existing and len(existing) == len(paths):
            sequence_rows.append(
                {
                    "sequence": sequence,
                    "status": "existing_predictions",
                    "num_frames": len(paths),
                    "inference_sec": None,
                    "fps_inference": None,
                    "peak_allocated_mb": None,
                    "peak_reserved_mb": None,
                }
            )
            continue

        started = time.perf_counter()
        try:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)
            loaded = stac_launch.load_images(
                [str(path) for path in paths], size=args.size, verbose=False, crop=False
            )
            loaded = stac_launch.collate_with_cat([tuple(loaded)])
            images = torch.stack([view["img"] for view in loaded], dim=1)
            images = stac_launch.ImgDust3r2Stream3r(images).to(device)
            torch.cuda.synchronize(device)
            inference_start = time.perf_counter()
            predictions = stac_launch.run(
                images, model, dtype=torch.float16, device=device, args=run_args
            )
            torch.cuda.synchronize(device)
            inference_sec = time.perf_counter() - inference_start
            peak_allocated = torch.cuda.max_memory_allocated(device) / 1024**2
            peak_reserved = torch.cuda.max_memory_reserved(device) / 1024**2
            depth = normalized_depth_tensor(torch, predictions["depth"], len(paths))
            write_depth_maps(np, torch, depth, sequence_output)
            sequence_rows.append(
                {
                    "sequence": sequence,
                    "status": "ok",
                    "num_frames": len(paths),
                    "total_sec": time.perf_counter() - started,
                    "inference_sec": inference_sec,
                    "fps_inference": len(paths) / max(inference_sec, 1e-12),
                    "peak_allocated_mb": peak_allocated,
                    "peak_reserved_mb": peak_reserved,
                    "timing": predictions.get("timing", {}),
                    "effective_config": predictions.get("effective_config", {}),
                }
            )
            del depth, predictions, images, loaded
            torch.cuda.empty_cache()
        except Exception as error:
            is_oom = isinstance(error, torch.cuda.OutOfMemoryError) or "out of memory" in str(error).lower()
            sequence_rows.append(
                {
                    "sequence": sequence,
                    "status": "oom" if is_oom else "error",
                    "num_frames": len(paths),
                    "total_sec": time.perf_counter() - started,
                    "inference_sec": None,
                    "fps_inference": None,
                    "peak_allocated_mb": torch.cuda.max_memory_allocated(device) / 1024**2,
                    "peak_reserved_mb": torch.cuda.max_memory_reserved(device) / 1024**2,
                    "error": f"{type(error).__name__}: {error}",
                    "traceback": traceback.format_exc(),
                }
            )
            torch.cuda.empty_cache()

    successful = [row for row in sequence_rows if row["status"] == "ok"]
    summary = {
        "method": "streamvggt_stac",
        "dataset": args.dataset,
        "mode": args.mode,
        "backend": args.backend,
        "stac_commit": git_commit(stac_root),
        "weights_path": str(weights),
        "input_size": args.size,
        "requested_max_frames": args.max_frames,
        "num_sequences": len(sequence_rows),
        "num_ok": len(successful),
        "num_failed": len(sequence_rows) - len(successful),
        "total_frames": sum(row["num_frames"] for row in successful),
        "total_inference_sec": sum(row["inference_sec"] for row in successful),
        "max_peak_allocated_mb": max(
            (row["peak_allocated_mb"] for row in successful), default=None
        ),
        "max_peak_reserved_mb": max(
            (row["peak_reserved_mb"] for row in successful), default=None
        ),
        "gpu_name": torch.cuda.get_device_name(device),
        "torch_version": str(torch.__version__),
        "cuda_version": torch.version.cuda or "",
        "python_version": platform.python_version(),
        "hostname": platform.node(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID", ""),
    }
    if summary["total_inference_sec"]:
        summary["fps_inference"] = summary["total_frames"] / summary["total_inference_sec"]
    payload = {"summary": summary, "sequences": sequence_rows}
    runtime_path = output_dir / "stage5e_runtime_memory.json"
    runtime_path.write_text(json.dumps(payload, indent=2))
    print(json.dumps(summary, indent=2))
    print(f"Wrote {runtime_path}")
    if summary["num_failed"] and not args.allow_failures:
        raise RuntimeError(
            f"STAC failed on {summary['num_failed']}/{summary['num_sequences']} sequences; "
            f"see {runtime_path}"
        )


if __name__ == "__main__":
    main()
