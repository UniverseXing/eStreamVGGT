#!/usr/bin/env python3
"""Export 12 paper panels for one Full/K4/K6/K8 qualitative comparison."""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
from pathlib import Path
from statistics import fmean, median

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/estreamvggt-matplotlib")

import matplotlib.pyplot as plt
import numpy as np
import torch


METHODS = (
    ("full_cache", "Full cache", None, None),
    ("k4", "K4", 4, "anchor_recent_dino_diverse_k4"),
    ("k6", "K6", 6, "anchor_recent_dino_diverse_k6"),
    ("k8", "K8", 8, "anchor_recent_dino_diverse_k8"),
)
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--images-dir", type=Path, required=True)
    parser.add_argument("--sequence", default="7scenes_chess_seq01")
    parser.add_argument(
        "--image-glob",
        default="*.color.png",
        help="Glob selecting RGB inputs while excluding depth/pose files.",
    )
    parser.add_argument(
        "--sampling-stride",
        type=int,
        default=5,
        help="Take every Nth matching source image before applying --max-frames.",
    )
    parser.add_argument("--max-frames", type=int, default=110)
    parser.add_argument(
        "--frame",
        type=int,
        help="One-based frame to display; defaults to the final processed frame.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("paper_assets/qualitative/7scenes_chess_seq01_v110"),
    )
    parser.add_argument("--point-stride", type=int, default=7)
    parser.add_argument("--confidence-percentile", type=float, default=50.0)
    parser.add_argument("--max-render-points", type=int, default=220_000)
    parser.add_argument("--view-elev", type=float, default=18.0)
    parser.add_argument("--view-azim", type=float, default=-70.0)
    return parser.parse_args()


def list_images(
    directory: Path,
    max_frames: int,
    image_glob: str = "*",
    sampling_stride: int = 1,
) -> list[Path]:
    if not directory.is_dir():
        raise FileNotFoundError(f"missing images directory: {directory}")
    if sampling_stride < 1:
        raise ValueError("sampling_stride must be at least 1")
    paths = sorted(
        path
        for path in directory.glob(image_glob)
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )
    paths = paths[::sampling_stride][:max_frames]
    if len(paths) != max_frames:
        raise ValueError(
            f"requested {max_frames} frames but found only {len(paths)} in {directory}"
        )
    if max_frames <= 8:
        raise ValueError("the comparison must exceed the largest K8 cache budget")
    return paths


def tensor_rgb(image: torch.Tensor) -> np.ndarray:
    array = image.detach().float().cpu().permute(1, 2, 0).numpy()
    return np.clip(array, 0.0, 1.0)


def prediction_arrays(prediction: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    depth = prediction["depth"].detach().float().cpu().numpy().squeeze()
    points = prediction["pts3d_in_other_view"].detach().float().cpu().numpy().squeeze(0)
    confidence = prediction["conf"].detach().float().cpu().numpy().squeeze(0)
    return depth, points, confidence


def sample_pointmap(
    points: np.ndarray,
    confidence: np.ndarray,
    rgb: np.ndarray,
    stride: int,
    percentile: float,
) -> tuple[np.ndarray, np.ndarray]:
    points = points[::stride, ::stride]
    confidence = confidence[::stride, ::stride]
    colors = rgb[::stride, ::stride]
    valid = np.isfinite(points).all(axis=-1) & np.isfinite(confidence)
    if valid.any():
        threshold = np.percentile(confidence[valid], percentile)
        valid &= confidence >= threshold
    valid &= np.linalg.norm(points, axis=-1) > 1e-8
    return points[valid], colors[valid]


def run_method(
    model,
    method_slug: str,
    method_label: str,
    window: int | None,
    policy: str | None,
    paths: list[Path],
    frame_number: int,
    repo_root: Path,
    point_stride: int,
    confidence_percentile: float,
) -> dict:
    sys.path.insert(0, str(repo_root / "src"))
    from streamvggt.utils.load_fn import load_and_preprocess_images

    target_index = frame_number - 1
    current_rgb: list[np.ndarray | None] = [None]
    selected: dict[str, np.ndarray] = {}
    point_chunks: list[np.ndarray] = []
    color_chunks: list[np.ndarray] = []

    def frames():
        for path in paths:
            image = load_and_preprocess_images([str(path)])[0]
            current_rgb[0] = tensor_rgb(image)
            yield {"img": image.unsqueeze(0).to("cuda", non_blocking=True)}

    def sink(index: int, prediction: dict) -> None:
        if current_rgb[0] is None:
            raise RuntimeError("RGB/prediction stream lost synchronisation")
        if index <= target_index:
            _, points, confidence = prediction_arrays(prediction)
            sampled_points, sampled_colors = sample_pointmap(
                points,
                confidence,
                current_rgb[0],
                point_stride,
                confidence_percentile,
            )
            point_chunks.append(sampled_points)
            color_chunks.append(sampled_colors)
        if index == target_index:
            depth, _, _ = prediction_arrays(prediction)
            selected["rgb"] = current_rgb[0].copy()
            selected["depth"] = depth

    inference_args = {
        "return_memory_trace": True,
        "return_frame_timings": True,
        "output_sink": sink,
        "retain_outputs": False,
        "retain_views": False,
    }
    if window is not None:
        inference_args.update(cache_window_size=window, cache_policy=policy)

    dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    started = time.perf_counter()
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=dtype):
        output = model.inference(frames(), **inference_args)
    torch.cuda.synchronize()
    wall_seconds = time.perf_counter() - started
    if set(selected) != {"rgb", "depth"}:
        raise RuntimeError(f"{method_label} did not produce display frame {frame_number}")
    if not point_chunks:
        raise RuntimeError(f"{method_label} produced no usable point samples")
    retained = output.memory_trace[target_index]["retained_frame_ids"]
    expected_states = frame_number if window is None else window
    if len(retained) != expected_states:
        raise RuntimeError(
            f"{method_label} retained {len(retained)} states, expected {expected_states}"
        )
    return {
        "method_slug": method_slug,
        "method_label": method_label,
        "cache_policy": "full_cache" if policy is None else policy,
        "cache_window_size": window,
        "frame_number": frame_number,
        "processed_frames": len(paths),
        "source_frame_name": paths[target_index].name,
        "retained_frame_ids_zero_based": retained,
        "rgb": selected["rgb"],
        "depth": selected["depth"],
        "points": np.concatenate(point_chunks, axis=0),
        "colors": np.concatenate(color_chunks, axis=0),
        "wall_seconds": wall_seconds,
        "mean_gpu_frame_ms": fmean(output.frame_inference_ms),
        "median_gpu_frame_ms": median(output.frame_inference_ms),
        "peak_allocated_mib": torch.cuda.max_memory_allocated() / (1024 ** 2),
    }


def points_for_display(
    points: np.ndarray,
    colors: np.ndarray,
    max_points: int,
) -> tuple[np.ndarray, np.ndarray]:
    valid = np.isfinite(points).all(axis=1)
    points, colors = points[valid], colors[valid]
    if len(points) == 0:
        raise ValueError("point cloud is empty")
    lower, upper = np.percentile(points, [1, 99], axis=0)
    central = ((points >= lower) & (points <= upper)).all(axis=1)
    points, colors = points[central], colors[central]
    if len(points) > max_points:
        indices = np.linspace(0, len(points) - 1, max_points, dtype=np.int64)
        points, colors = points[indices], colors[indices]
    # Camera-centric x/z/-y display only; no geometry registration or optimisation.
    displayed = np.column_stack([points[:, 0], points[:, 2], -points[:, 1]])
    return displayed, colors


def annotate(axis, text: str) -> None:
    text_method = axis.text2D if getattr(axis, "name", "") == "3d" else axis.text
    text_method(
        0.02,
        0.97,
        text,
        transform=axis.transAxes,
        va="top",
        ha="left",
        color="white",
        fontsize=10,
        fontweight="bold",
        bbox={"facecolor": "black", "alpha": 0.74, "pad": 3, "edgecolor": "none"},
    )


def save_raster_panel(array: np.ndarray, path: Path, label: str, cmap=None, vmin=None, vmax=None) -> None:
    figure, axis = plt.subplots(figsize=(4, 3), dpi=300)
    axis.imshow(array, cmap=cmap, vmin=vmin, vmax=vmax)
    annotate(axis, label)
    axis.axis("off")
    figure.subplots_adjust(left=0, right=1, top=1, bottom=0)
    figure.savefig(path, dpi=300, pad_inches=0)
    plt.close(figure)


def save_pointcloud_panel(
    points: np.ndarray,
    colors: np.ndarray,
    path: Path,
    label: str,
    center: np.ndarray,
    radius: float,
    elev: float,
    azim: float,
) -> None:
    figure = plt.figure(figsize=(4, 3), dpi=300)
    axis = figure.add_subplot(111, projection="3d")
    axis.scatter(
        points[:, 0], points[:, 1], points[:, 2],
        c=colors, s=0.32, linewidths=0, rasterized=True,
    )
    axis.set_xlim(center[0] - radius, center[0] + radius)
    axis.set_ylim(center[1] - radius, center[1] + radius)
    axis.set_zlim(center[2] - radius, center[2] + radius)
    axis.set_box_aspect((1, 1, 1))
    axis.view_init(elev=elev, azim=azim)
    axis.set_axis_off()
    annotate(axis, label)
    figure.subplots_adjust(left=0, right=1, top=1, bottom=0)
    figure.savefig(path, dpi=300, pad_inches=0)
    plt.close(figure)


def export_panels(
    results: list[dict],
    output_dir: Path,
    sequence: str,
    max_render_points: int,
    elev: float,
    azim: float,
) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    valid_depths = [
        result["depth"][np.isfinite(result["depth"]) & (result["depth"] > 0)]
        for result in results
    ]
    depth_values = np.concatenate([values for values in valid_depths if len(values)])
    depth_low, depth_high = np.percentile(depth_values, [2, 98])
    displayed_clouds = [
        points_for_display(result["points"], result["colors"], max_render_points)
        for result in results
    ]
    combined_points = np.concatenate([points for points, _ in displayed_clouds], axis=0)
    lower, upper = np.percentile(combined_points, [1, 99], axis=0)
    center = (lower + upper) / 2
    radius = max((upper - lower).max() / 2, 1e-6)
    outputs = []
    for result, (points, colors) in zip(results, displayed_clouds):
        label = (
            f"{result['method_label']} | {sequence} | "
            f"view {result['frame_number']}/{result['processed_frames']} | "
            f"source {Path(result['source_frame_name']).stem.replace('.color', '')}"
        )
        rgb_path = output_dir / f"{result['method_slug']}_rgb.png"
        depth_path = output_dir / f"{result['method_slug']}_depth.png"
        cloud_path = output_dir / f"{result['method_slug']}_pointcloud.png"
        save_raster_panel(result["rgb"], rgb_path, label)
        save_raster_panel(
            np.clip(result["depth"], depth_low, depth_high),
            depth_path,
            label,
            cmap=plt.get_cmap("Spectral_r"),
            vmin=depth_low,
            vmax=depth_high,
        )
        save_pointcloud_panel(
            points, colors, cloud_path, label, center, radius, elev, azim
        )
        outputs.extend((rgb_path, depth_path, cloud_path))
    if len(outputs) != 12 or len({path.name for path in outputs}) != 12:
        raise RuntimeError("expected exactly 12 uniquely named PNG panels")
    return outputs


def write_run_notes(
    results: list[dict],
    output_dir: Path,
    sequence: str,
    source_paths: list[Path],
    image_glob: str,
    sampling_stride: int,
    confidence_percentile: float,
    point_stride: int,
) -> None:
    metadata = {
        "sequence": sequence,
        "source_image_glob": image_glob,
        "source_sampling_stride": sampling_stride,
        "selected_source_frames": [path.name for path in source_paths],
        "methods": [
            {
                key: result[key]
                for key in (
                    "method_slug",
                    "method_label",
                    "cache_policy",
                    "cache_window_size",
                    "frame_number",
                    "processed_frames",
                    "source_frame_name",
                    "retained_frame_ids_zero_based",
                    "wall_seconds",
                    "mean_gpu_frame_ms",
                    "median_gpu_frame_ms",
                    "peak_allocated_mib",
                )
            }
            for result in results
        ],
        "online_causal_processing": True,
        "input_loading": "one frame at a time",
        "output_lifecycle": "CPU visualization sink; GPU outputs released per frame",
        "geometry_postprocessing": "none",
        "visualization_only": {
            "shared_depth_display_percentiles": [2, 98],
            "point_confidence_percentile": confidence_percentile,
            "point_stride": point_stride,
            "shared_pointcloud_view_and_bounds": True,
            "point_axis_display": "x, z, -y",
        },
        "gpu": torch.cuda.get_device_name(0),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "python_version": platform.python_version(),
    }
    (output_dir / "qualitative_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    frame_number = results[0]["frame_number"]
    processed = results[0]["processed_frames"]
    source_frame = results[0]["source_frame_name"]
    caption = (
        f"Qualitative comparison on the 7-Scenes {sequence} object-centred sequence at "
        f"sampled view {frame_number}/{processed} (source {source_frame}). The input set was "
        f"fixed in advance by taking every {sampling_stride}th RGB frame from the beginning "
        "of the sequence, rather than selecting views by reconstruction quality. Columns "
        "correspond to the original StreamVGGT Full "
        "cache and the proposed K4, K6, and K8 configurations; rows show the identical RGB "
        "input, direct depth prediction, and accumulated point-head reconstruction. All "
        f"methods processed frames causally one at a time on {torch.cuda.get_device_name(0)} "
        "and released per-frame GPU outputs through the same CPU visualization sink. No "
        "bundle adjustment, ICP, temporal smoothing, or geometric refinement was applied. "
        "For visualization only, the four depth maps share one 2nd--98th-percentile display "
        f"range, and point samples were filtered at the {confidence_percentile:g}th confidence "
        f"percentile, uniformly subsampled with stride {point_stride}, and rendered with one "
        "shared view and coordinate range."
    )
    (output_dir / "qualitative_caption.txt").write_text(caption + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    if args.max_frames < 9:
        raise ValueError("--max-frames must exceed 8")
    frame_number = args.frame or args.max_frames
    if not 9 <= frame_number <= args.max_frames:
        raise ValueError("--frame must be between 9 and --max-frames")
    if args.point_stride < 1:
        raise ValueError("--point-stride must be at least 1")
    if args.sampling_stride < 1:
        raise ValueError("--sampling-stride must be at least 1")
    if not 0 <= args.confidence_percentile < 100:
        raise ValueError("--confidence-percentile must be in [0, 100)")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the qualitative comparison")
    repo_root = args.repo_root.resolve()
    weights = args.weights.resolve()
    images_dir = args.images_dir.resolve()
    if not weights.is_file():
        raise FileNotFoundError(f"missing weights: {weights}")
    paths = list_images(
        images_dir,
        args.max_frames,
        args.image_glob,
        args.sampling_stride,
    )

    sys.path.insert(0, str(repo_root / "src"))
    from streamvggt.models.streamvggt import StreamVGGT

    device = torch.device("cuda")
    model = StreamVGGT()
    checkpoint = torch.load(weights, map_location=device)
    model.load_state_dict(checkpoint, strict=True)
    model.eval().to(device)
    del checkpoint

    results = []
    for method_slug, method_label, window, policy in METHODS:
        print(
            f"[qualitative] {method_label}: {len(paths)} frames; "
            f"display frame={frame_number}"
        )
        results.append(
            run_method(
                model,
                method_slug,
                method_label,
                window,
                policy,
                paths,
                frame_number,
                repo_root,
                args.point_stride,
                args.confidence_percentile,
            )
        )
        torch.cuda.empty_cache()

    output_dir = args.output_dir.resolve()
    outputs = export_panels(
        results,
        output_dir,
        args.sequence,
        args.max_render_points,
        args.view_elev,
        args.view_azim,
    )
    write_run_notes(
        results,
        output_dir,
        args.sequence,
        paths,
        args.image_glob,
        args.sampling_stride,
        args.confidence_percentile,
        args.point_stride,
    )
    for path in outputs:
        print(f"Wrote {path}")
    print(f"Wrote {output_dir / 'qualitative_metadata.json'}")
    print(f"Wrote {output_dir / 'qualitative_caption.txt'}")


if __name__ == "__main__":
    main()
