#!/usr/bin/env python3
"""Run official OVGGT on the project's matched VideoDepth inputs.

The external checkout remains unmodified.  This adapter only supplies the
same checkpoint, preprocessing, sequence coverage and metric-compatible depth
files used by the project experiments while recording runtime and CUDA peaks.
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
    parser.add_argument("--ovggt-root", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--dataset", choices=("bonn", "sintel", "kitti"), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seq-list", nargs="+")
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--size", type=int, default=518)
    parser.add_argument("--mode", choices=("ovggt", "full"), default="ovggt")
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


def bonn_frame_dir(root: Path, sequence: str) -> Path:
    sequence_root = root / "bonn/rgbd_bonn_dataset" / f"rgbd_bonn_{sequence}"
    for name in ("rgb_110_sampled", "rgb_110", "rgb"):
        candidate = sequence_root / name
        if candidate.is_dir():
            return candidate
    return sequence_root / "rgb_110"


def dataset_sequences(data_root: Path, dataset: str, requested: list[str] | None):
    if requested:
        return tuple(requested)
    if dataset == "bonn":
        return BONN_SEQUENCES
    if dataset == "sintel":
        root = data_root / "sintel/training/final"
    else:
        root = data_root / "kitti/depth_selection/val_selection_cropped/image_gathered"
    if not root.is_dir():
        raise FileNotFoundError(root)
    return tuple(path.name for path in sorted(root.iterdir()) if path.is_dir())


def frame_paths(data_root: Path, dataset: str, sequence: str) -> list[Path]:
    if dataset == "bonn":
        directory = bonn_frame_dir(data_root, sequence)
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


def normalized_depth_tensor(torch, values, expected_frames: int):
    if isinstance(values, (list, tuple)):
        depth = torch.cat(
            [value.detach().float().cpu() for value in values], dim=0
        )
    else:
        depth = values.detach().float().cpu()
    if depth.ndim >= 3 and depth.shape[-1] == 1:
        depth = depth.squeeze(-1)
    while depth.ndim > 3 and depth.shape[0] == 1:
        depth = depth.squeeze(0)
    if depth.ndim == 4 and depth.shape[1] == 1:
        depth = depth.squeeze(1)
    if depth.ndim == 2 and expected_frames == 1:
        depth = depth.unsqueeze(0)
    if depth.ndim != 3 or depth.shape[0] != expected_frames:
        raise RuntimeError(
            f"unexpected OVGGT depth shape {tuple(depth.shape)} "
            f"for {expected_frames} frames"
        )
    return depth


def write_depth_maps(np, depth, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    for index in range(depth.shape[0]):
        np.save(output_dir / f"frame_{index:04d}.npy", depth[index].numpy())


def prepare_frames(load_images, paths: list[Path], size: int, device):
    loaded = load_images(
        [str(path) for path in paths], size=size, verbose=False, crop=False
    )
    # Match both the project and official OVGGT VideoDepth launchers.  DUSt3R's
    # loader returns ImageNet-normalized tensors in [-1, 1], while StreamVGGT
    # inference expects the remapped [0, 1] representation.
    return [{"img": ((item["img"] + 1.0) / 2.0).to(device)} for item in loaded]


def main() -> None:
    args = parse_args()
    repo_root = args.repo_root.resolve()
    ovggt_root = args.ovggt_root.resolve()
    weights = args.weights.resolve()
    output_dir = args.output_dir.resolve()
    ovggt_src = ovggt_root / "src"
    if not (ovggt_src / "ovggt/models/ovggt.py").is_file():
        raise FileNotFoundError(f"not an official OVGGT checkout: {ovggt_root}")
    if not weights.is_file():
        raise FileNotFoundError(weights)

    sys.path.insert(0, str(ovggt_src))
    os.chdir(ovggt_root)
    import numpy as np
    import torch
    from dust3r.utils.image import load_images_for_eval
    from ovggt.models.ovggt import OVGGT

    if not torch.cuda.is_available():
        raise RuntimeError("Stage 5E OVGGT inference requires a CUDA GPU")
    device = torch.device("cuda")
    dtype = (
        torch.bfloat16
        if torch.cuda.get_device_capability(device)[0] >= 8
        else torch.float16
    )
    data_root = repo_root / "data/eval"
    sequences = dataset_sequences(data_root, args.dataset, args.seq_list)
    if not sequences:
        raise RuntimeError(f"no sequences selected for {args.dataset}")

    # The parity path must remain causal/streaming like StreamVGGT Full.  A
    # very large budget disables OVGGT eviction for the 10-frame smoke without
    # switching to its offline all-frame forward path.
    if args.mode == "full":
        model = OVGGT(total_budget=1_000_000_000, camera_budget=1_000_000_000)
    else:
        model = OVGGT()
    checkpoint = torch.load(weights, map_location="cpu")
    model.load_state_dict(checkpoint, strict=True)
    del checkpoint
    model.eval().to(device)

    output_dir.mkdir(parents=True, exist_ok=True)
    sequence_rows = []
    for sequence in sequences:
        sequence_output = output_dir / sequence
        paths = frame_paths(data_root, args.dataset, sequence)
        if args.max_frames is not None:
            paths = paths[: args.max_frames]
        existing = sorted(sequence_output.glob("frame_*.npy"))
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
            frames = prepare_frames(load_images_for_eval, paths, args.size, device)
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)
            torch.cuda.synchronize(device)
            inference_start = time.perf_counter()
            with torch.no_grad(), torch.amp.autocast("cuda", dtype=dtype):
                predictions = model.inference(frames)
            torch.cuda.synchronize(device)
            inference_sec = time.perf_counter() - inference_start
            peak_allocated = torch.cuda.max_memory_allocated(device) / 1024**2
            peak_reserved = torch.cuda.max_memory_reserved(device) / 1024**2
            depths = [item["depth"] for item in predictions.ress]
            depth = normalized_depth_tensor(torch, depths, len(paths))
            write_depth_maps(np, depth, sequence_output)
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
                }
            )
            del depth, depths, predictions, frames
            torch.cuda.empty_cache()
        except Exception as error:
            is_oom = isinstance(error, torch.cuda.OutOfMemoryError) or (
                "out of memory" in str(error).lower()
            )
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
        "method": "ovggt" if args.mode == "ovggt" else "ovggt_full",
        "dataset": args.dataset,
        "mode": args.mode,
        "backend": "official_pytorch",
        "competitor_commit": git_commit(ovggt_root),
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
        summary["fps_inference"] = (
            summary["total_frames"] / summary["total_inference_sec"]
        )
    payload = {"summary": summary, "sequences": sequence_rows}
    runtime_path = output_dir / "stage5e_runtime_memory.json"
    runtime_path.write_text(json.dumps(payload, indent=2))
    print(json.dumps(summary, indent=2))
    print(f"Wrote {runtime_path}")
    if summary["num_failed"] and not args.allow_failures:
        raise RuntimeError(
            f"OVGGT failed on {summary['num_failed']}/{summary['num_sequences']} "
            f"sequences; see {runtime_path}"
        )


if __name__ == "__main__":
    main()
