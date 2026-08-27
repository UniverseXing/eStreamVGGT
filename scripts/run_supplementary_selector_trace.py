#!/usr/bin/env python3
"""Run and visualise one fixed K4/K6/K8 selector diagnostic sequence."""

from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import sys
from pathlib import Path
from statistics import fmean, median

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/estreamvggt-supp-matplotlib")

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[1]
METHODS = (
    ("k4", "K4", 4, "anchor_recent_dino_diverse_k4"),
    ("k6", "K6", 6, "anchor_recent_dino_diverse_k6"),
    ("k8", "K8", 8, "anchor_recent_dino_diverse_k8"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument(
        "--images-dir", type=Path, default=Path("data/eval/7scenes/chess/seq-01")
    )
    parser.add_argument("--image-glob", default="*.color.png")
    parser.add_argument("--sampling-stride", type=int, default=5)
    parser.add_argument("--max-frames", type=int, default=110)
    parser.add_argument("--checkpoints", nargs="+", type=int, default=(16, 51, 110))
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path("eval_results/supplementary_selector_trace"),
    )
    return parser.parse_args()


def selected_images(directory: Path, pattern: str, stride: int, count: int) -> list[Path]:
    if stride < 1:
        raise ValueError("--sampling-stride must be at least 1")
    paths = sorted(directory.glob(pattern))[::stride][:count]
    if len(paths) != count:
        raise ValueError(f"requested {count} images but selected {len(paths)}")
    if count <= 8:
        raise ValueError("diagnostic sequence must exceed K8")
    return paths


def percentile(values: list[float], q: float) -> float:
    return float(np.percentile(np.asarray(values, dtype=np.float64), q))


def run_method(model, paths: list[Path], slug: str, label: str, window: int, policy: str, repo_root: Path) -> dict:
    sys.path.insert(0, str(repo_root / "src"))
    from streamvggt.utils.load_fn import load_and_preprocess_images

    def frames():
        for path in paths:
            image = load_and_preprocess_images([str(path)])[0]
            yield {"img": image.unsqueeze(0).to("cuda", non_blocking=True)}

    dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=dtype):
        output = model.inference(
            frames(),
            cache_window_size=window,
            cache_policy=policy,
            return_memory_events=True,
            return_memory_trace=True,
            return_frame_timings=True,
            output_sink=lambda _index, _prediction: None,
            retain_outputs=False,
            retain_views=False,
        )
    if len(output.memory_trace) != len(paths):
        raise RuntimeError(f"{label} trace length mismatch")
    if len(output.memory_trace[-1]["retained_frame_ids"]) != window:
        raise RuntimeError(f"{label} did not finish with exactly {window} states")
    selection_ms = [float(event["selection_cuda_ms"]) for event in output.memory_events]
    frame_ms = [float(value) for value in output.frame_inference_ms]
    final_trace = output.memory_trace[-1]
    retained = len(final_trace["retained_frame_ids"])
    mib_to_bytes = 1024 ** 2
    return {
        "method": slug,
        "method_label": label,
        "cache_window_size": window,
        "cache_policy": policy,
        "num_frames": len(paths),
        "events": output.memory_events,
        "memory_trace": output.memory_trace,
        "frame_inference_ms": frame_ms,
        "summary": {
            "num_selection_events": len(selection_ms),
            "mean_selection_cuda_ms": fmean(selection_ms),
            "median_selection_cuda_ms": median(selection_ms),
            "p95_selection_cuda_ms": percentile(selection_ms, 95),
            "total_selection_cuda_ms": sum(selection_ms),
            "mean_frame_inference_ms": fmean(frame_ms),
            "selection_percent_of_frame_gpu_time": 100.0 * sum(selection_ms) / sum(frame_ms),
            "total_selector_dot_products": sum(int(event["selector_dot_products"]) for event in output.memory_events),
            "mean_selector_dot_products_per_event": fmean(
                [int(event["selector_dot_products"]) for event in output.memory_events]
            ),
            "descriptor_bytes_per_retained_frame": int(
                round(final_trace["descriptor_mib"] * mib_to_bytes / retained)
            ),
            "aggregator_kv_bytes_per_retained_frame": int(
                round(final_trace["aggregator_kv_mib"] * mib_to_bytes / retained)
            ),
            "camera_kv_bytes_per_retained_frame": int(
                round(final_trace["camera_kv_mib"] * mib_to_bytes / retained)
            ),
            "peak_allocated_mib": torch.cuda.max_memory_allocated() / mib_to_bytes,
            "peak_reserved_mib": torch.cuda.max_memory_reserved() / mib_to_bytes,
        },
    }


def write_payload(result: dict, paths: list[Path], output_dir: Path) -> None:
    payload = {
        **result,
        "source_images": [path.name for path in paths],
        "gpu_name": torch.cuda.get_device_name(0),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "python_version": platform.python_version(),
        "timing_scope": "CUDA events around pruning/index selection only; trace logging excluded",
    }
    path = output_dir / f"{result['method']}_selector_trace.json"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote {path}")


def plot_timeline(results: list[dict], output_dir: Path) -> None:
    figure, axes = plt.subplots(3, 1, figsize=(11.5, 8.2), dpi=180, sharex=True)
    for axis, result in zip(axes, results):
        for trace in result["memory_trace"]:
            frame = trace["frame_index"]
            retained = trace["retained_frame_ids"]
            axis.scatter([frame] * len(retained), retained, s=7, alpha=0.72)
        axis.plot([0, result["num_frames"] - 1], [0, result["num_frames"] - 1], color="0.75", linewidth=0.8)
        axis.set_ylabel(f"{result['method_label']} retained ID")
        axis.grid(alpha=0.18)
    axes[-1].set_xlabel("Current frame ID (zero-based sampled sequence)")
    figure.suptitle("Actual retained-frame trajectories on the same 110-view sequence")
    figure.tight_layout()
    path = output_dir / "figure_p1_01_selector_timeline.png"
    figure.savefig(path, bbox_inches="tight")
    plt.close(figure)
    print(f"Wrote {path}")


def score_text(event: dict, frame_id: int) -> str:
    for item in event.get("candidate_similarities", []):
        if int(item["frame_id"]) != frame_id:
            continue
        score = item.get("max_similarity_to_recent", item.get("max_similarity_to_reference"))
        bank = item.get("temporal_bank")
        prefix = f"{bank}, " if bank else ""
        return f"{prefix}cos={float(score):.3f}"
    if frame_id == 0:
        return "anchor"
    return "recent/fill"


def plot_checkpoint(results: list[dict], paths: list[Path], one_based: int, output_dir: Path) -> None:
    step = one_based - 1
    max_slots = max(result["cache_window_size"] for result in results)
    figure, axes = plt.subplots(3, max_slots, figsize=(2.0 * max_slots, 6.3), dpi=170)
    for row_index, result in enumerate(results):
        event = next((item for item in result["events"] if int(item["step"]) == step), None)
        if event is None:
            raise RuntimeError(f"missing {result['method_label']} event at step {step}")
        selected = [int(value) for value in event["selected_frame_ids"]]
        evicted = ",".join(str(value) for value in event.get("evicted_frame_ids", [])) or "none"
        for column in range(max_slots):
            axis = axes[row_index, column]
            axis.axis("off")
            if column >= len(selected):
                continue
            frame_id = selected[column]
            axis.imshow(Image.open(paths[frame_id]).convert("RGB"))
            axis.set_title(f"id {frame_id}\n{score_text(event, frame_id)}", fontsize=8)
        axes[row_index, 0].text(
            -0.08, 0.5,
            f"{result['method_label']}\nevicted: {evicted}",
            transform=axes[row_index, 0].transAxes,
            ha="right", va="center", fontsize=9,
        )
    figure.suptitle(
        f"Retained source thumbnails after sampled view {one_based} (source {paths[step].name})"
    )
    figure.tight_layout()
    path = output_dir / f"figure_p1_01_selector_checkpoint_{one_based:03d}.png"
    figure.savefig(path, bbox_inches="tight")
    plt.close(figure)
    print(f"Wrote {path}")


def write_summary(results: list[dict], output_dir: Path) -> None:
    rows = []
    for result in results:
        rows.append(
            {
                "method": result["method"],
                "method_label": result["method_label"],
                "cache_policy": result["cache_policy"],
                "cache_window_size": result["cache_window_size"],
                "num_frames": result["num_frames"],
                **result["summary"],
                "gpu_name": torch.cuda.get_device_name(0),
                "torch_version": torch.__version__,
                "cuda_version": torch.version.cuda or "",
            }
        )
    path = output_dir / "p1_selector_overhead.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {path}")


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    root = args.repo_root.resolve()
    weights = args.weights if args.weights.is_absolute() else root / args.weights
    images_dir = args.images_dir if args.images_dir.is_absolute() else root / args.images_dir
    output_dir = args.output_dir if args.output_dir.is_absolute() else root / args.output_dir
    if not weights.is_file():
        raise FileNotFoundError(weights)
    paths = selected_images(images_dir, args.image_glob, args.sampling_stride, args.max_frames)
    if any(value <= 8 or value > len(paths) for value in args.checkpoints):
        raise ValueError("all checkpoints must be in [9, max_frames]")
    output_dir.mkdir(parents=True, exist_ok=True)

    sys.path.insert(0, str(root / "src"))
    from streamvggt.models.streamvggt import StreamVGGT

    model = StreamVGGT().eval().to("cuda")
    checkpoint = torch.load(weights, map_location="cuda")
    model.load_state_dict(checkpoint, strict=True)
    del checkpoint

    results = []
    for slug, label, window, policy in METHODS:
        print(f"[supplementary selector] {label}: {len(paths)} views")
        result = run_method(model, paths, slug, label, window, policy, root)
        write_payload(result, paths, output_dir)
        results.append(result)
    write_summary(results, output_dir)
    plot_timeline(results, output_dir)
    for checkpoint_view in args.checkpoints:
        plot_checkpoint(results, paths, checkpoint_view, output_dir)
    metadata = {
        "dataset": "7-Scenes chess/seq-01",
        "image_glob": args.image_glob,
        "sampling_stride": args.sampling_stride,
        "num_frames": len(paths),
        "first_source_frame": paths[0].name,
        "last_source_frame": paths[-1].name,
        "checkpoints_one_based": args.checkpoints,
        "purpose": "descriptive selector audit; not used for tuning",
    }
    (output_dir / "selector_trace_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
