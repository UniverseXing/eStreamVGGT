#!/usr/bin/env python3
"""Stage 3.6A bounded-window pose stitching on Bonn person_tracking2."""

import argparse
import json
import os
import os.path as osp
import sys
import time
import traceback

import numpy as np
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from add_ckpt_path import add_path_to_dust3r
from eval.long_sequence.eval_stage3_4_long import load_bonn_sequence
from eval.pose_evaluation.trajectory_metrics import evaluate_trajectory


REPO_ROOT = osp.abspath(osp.join(osp.dirname(__file__), "../../.."))
DEFAULT_BONN_ROOT = osp.join(REPO_ROOT, "data/eval/bonn/rgbd_bonn_dataset")


def parse_args():
    parser = argparse.ArgumentParser("Stage 3.6A bounded-window pose stitching")
    parser.add_argument("--weights", required=True)
    parser.add_argument("--data-root", default=DEFAULT_BONN_ROOT)
    parser.add_argument("--sequence", default="person_tracking2")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--method", required=True)
    parser.add_argument("--mode", choices=("stream", "window_stitch"), required=True)
    parser.add_argument("--size", type=int, default=518)
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--prefix-frames", type=int, nargs="+", required=True)
    parser.add_argument("--cache-window", type=int)
    parser.add_argument("--cache-policy", default="fifo")
    parser.add_argument("--window-size", type=int)
    parser.add_argument("--overlap", type=int)
    return parser.parse_args()


def rotation_angle_deg(rotation):
    u, _, vh = np.linalg.svd(rotation)
    nearest = u @ vh
    if np.linalg.det(nearest) < 0:
        u[:, -1] *= -1
        nearest = u @ vh
    cosine = np.clip((np.trace(nearest) - 1.0) / 2.0, -1.0, 1.0)
    return float(np.degrees(np.arccos(cosine)))


def project_rotation(rotation):
    u, _, vh = np.linalg.svd(rotation)
    nearest = u @ vh
    if np.linalg.det(nearest) < 0:
        u[:, -1] *= -1
        nearest = u @ vh
    return nearest


def orientation_alignment(source_rotations, target_rotations):
    relative = np.stack(
        [target @ source.T for source, target in zip(source_rotations, target_rotations)]
    )
    return project_rotation(relative.mean(axis=0))


def estimate_sim3(source_c2w, target_c2w):
    """Estimate target ~= Sim3(source), falling back for degenerate centers."""
    source_points = np.asarray(source_c2w[:, :3, 3], dtype=np.float64)
    target_points = np.asarray(target_c2w[:, :3, 3], dtype=np.float64)
    source_mean = source_points.mean(axis=0)
    target_mean = target_points.mean(axis=0)
    source_centered = source_points - source_mean
    target_centered = target_points - target_mean
    covariance = target_centered.T @ source_centered / len(source_points)
    rank = int(np.linalg.matrix_rank(covariance, tol=1e-10))
    method = "umeyama_centers"

    try:
        if rank < 2:
            raise np.linalg.LinAlgError(f"center covariance rank {rank}")
        u, singular_values, vh = np.linalg.svd(covariance)
        signs = np.ones(3, dtype=np.float64)
        if np.linalg.det(u @ vh) < 0:
            signs[-1] = -1.0
        rotation = u @ np.diag(signs) @ vh
        source_variance = float(np.mean(np.sum(source_centered**2, axis=1)))
        if source_variance <= 1e-12:
            raise np.linalg.LinAlgError("source center variance is zero")
        scale = float(np.sum(singular_values * signs) / source_variance)
        if not np.isfinite(scale) or scale <= 1e-8:
            raise np.linalg.LinAlgError(f"invalid scale {scale}")
    except np.linalg.LinAlgError:
        method = "orientation_fallback"
        rotation = orientation_alignment(
            source_c2w[:, :3, :3], target_c2w[:, :3, :3]
        )
        rotated_source = (rotation @ source_centered.T).T
        denominator = float(np.sum(rotated_source**2))
        scale = (
            float(np.sum(rotated_source * target_centered) / denominator)
            if denominator > 1e-12
            else 1.0
        )
        if not np.isfinite(scale) or scale <= 1e-8:
            source_steps = np.linalg.norm(np.diff(source_points, axis=0), axis=1)
            target_steps = np.linalg.norm(np.diff(target_points, axis=0), axis=1)
            valid = source_steps > 1e-8
            scale = float(np.median(target_steps[valid] / source_steps[valid])) if np.any(valid) else 1.0

    translation = target_mean - scale * (rotation @ source_mean)
    return {
        "scale": scale,
        "rotation": rotation,
        "translation": translation,
        "center_covariance_rank": rank,
        "method": method,
    }


def apply_sim3(poses, transform):
    transformed = np.array(poses, dtype=np.float64, copy=True)
    rotation = transform["rotation"]
    scale = transform["scale"]
    translation = transform["translation"]
    transformed[:, :3, 3] = (
        scale * (rotation @ transformed[:, :3, 3].T)
    ).T + translation
    transformed[:, :3, :3] = rotation[None] @ transformed[:, :3, :3]
    transformed[:, 3] = np.array([0.0, 0.0, 0.0, 1.0])
    return transformed


def overlap_residual(source_aligned, target):
    translation_rmse = float(
        np.sqrt(np.mean(np.sum((source_aligned[:, :3, 3] - target[:, :3, 3]) ** 2, axis=1)))
    )
    rotations = [
        rotation_angle_deg(target_pose[:3, :3].T @ source_pose[:3, :3])
        for source_pose, target_pose in zip(source_aligned, target)
    ]
    return translation_rmse, float(np.mean(rotations)), float(np.max(rotations))


def pose_encoding_to_c2w(output, image_hw, converter):
    pose_encoding = torch.stack(
        [prediction["camera_pose"] for prediction in output.ress], dim=1
    )
    pred_w2c, _ = converter(
        pose_encoding,
        image_hw,
        build_intrinsics=False,
    )
    pred_w2c = pred_w2c[0].detach().cpu().double().numpy()
    pred_w2c_h = np.tile(np.eye(4, dtype=np.float64), (len(pred_w2c), 1, 1))
    pred_w2c_h[:, :3] = pred_w2c
    return np.linalg.inv(pred_w2c_h)


def infer_paths(model, load_images, converter, image_paths, size, device, dtype, cache_window=None, cache_policy="fifo"):
    loaded = load_images(image_paths, size=size, crop=False)
    frames = [{"img": (item["img"].to(device) + 1.0) / 2.0} for item in loaded]
    del loaded
    torch.cuda.synchronize(device)
    start = time.perf_counter()
    with torch.no_grad(), torch.cuda.amp.autocast(dtype=dtype):
        output = model.inference(
            frames,
            cache_window_size=cache_window,
            cache_policy=cache_policy,
        )
    torch.cuda.synchronize(device)
    inference_sec = time.perf_counter() - start
    pred_c2w = pose_encoding_to_c2w(
        output,
        frames[0]["img"].shape[-2:],
        converter,
    )
    output = frames = None
    torch.cuda.empty_cache()
    return pred_c2w, inference_sec


def stream_trajectory(model, load_images, converter, image_paths, args, device, dtype):
    pred_c2w, inference_sec = infer_paths(
        model,
        load_images,
        converter,
        image_paths,
        args.size,
        device,
        dtype,
        cache_window=args.cache_window,
        cache_policy=args.cache_policy,
    )
    return pred_c2w, inference_sec, len(image_paths), []


def window_stitched_trajectory(model, load_images, converter, image_paths, args, device, dtype):
    if args.window_size is None or args.overlap is None:
        raise ValueError("window_stitch requires --window-size and --overlap")
    if args.window_size < 3:
        raise ValueError("--window-size must be at least 3")
    if not 3 <= args.overlap < args.window_size:
        raise ValueError("--overlap must be at least 3 and smaller than the window")

    step = args.window_size - args.overlap
    stitched = None
    inference_sec = 0.0
    processed_frames = 0
    alignment_events = []
    for start in range(0, len(image_paths), step):
        end = min(start + args.window_size, len(image_paths))
        if stitched is not None and end <= len(stitched):
            continue
        local, elapsed = infer_paths(
            model,
            load_images,
            converter,
            image_paths[start:end],
            args.size,
            device,
            dtype,
        )
        inference_sec += elapsed
        processed_frames += len(local)
        if stitched is None:
            stitched = local
            alignment_events.append(
                {"start": start, "end": end, "overlap": 0, "method": "identity"}
            )
            continue

        overlap = min(len(stitched) - start, len(local), args.overlap)
        if overlap < 3:
            raise RuntimeError(
                f"window {start}:{end} has only {overlap} stitched overlap poses"
            )
        target_overlap = stitched[start : start + overlap]
        transform = estimate_sim3(local[:overlap], target_overlap)
        local_aligned = apply_sim3(local, transform)
        trans_rmse, rot_mean, rot_max = overlap_residual(
            local_aligned[:overlap], target_overlap
        )
        alignment_events.append(
            {
                "start": start,
                "end": end,
                "overlap": overlap,
                "method": transform["method"],
                "scale": transform["scale"],
                "center_covariance_rank": transform["center_covariance_rank"],
                "overlap_translation_rmse": trans_rmse,
                "overlap_rotation_deg_mean": rot_mean,
                "overlap_rotation_deg_max": rot_max,
            }
        )
        stitched = np.concatenate([stitched, local_aligned[overlap:]], axis=0)
        if len(stitched) >= len(image_paths):
            break

    if stitched is None or len(stitched) != len(image_paths):
        raise RuntimeError(
            f"stitched trajectory has {0 if stitched is None else len(stitched)} "
            f"poses for {len(image_paths)} frames"
        )
    return stitched, inference_sec, processed_frames, alignment_events


def prefix_metrics(gt_c2w, pred_c2w, requested):
    prefixes = sorted({value for value in requested if 2 <= value <= len(pred_c2w)} | {len(pred_c2w)})
    rows = []
    for prefix in prefixes:
        row = {"prefix_frames": prefix}
        try:
            metrics = evaluate_trajectory(gt_c2w[:prefix], pred_c2w[:prefix])
            metrics.pop("pred_c2w_aligned")
            row.update({"status": "ok", **metrics})
        except Exception as error:
            row.update(
                {
                    "status": "failed",
                    "error": f"{type(error).__name__}: {error}",
                    "ate": None,
                    "rpe_trans": None,
                    "rpe_rot_deg": None,
                }
            )
        rows.append(row)
    return rows


def main():
    args = parse_args()
    if args.max_frames is not None and args.max_frames < 2:
        raise ValueError("--max-frames must be at least 2")
    if any(value < 2 for value in args.prefix_frames):
        raise ValueError("prefix values must be at least 2")

    image_paths, _, gt_c2w, _ = load_bonn_sequence(
        osp.abspath(args.data_root), args.sequence, args.max_frames
    )
    add_path_to_dust3r(args.weights)
    from dust3r.utils.image import load_images_for_eval as load_images
    from streamvggt.models.streamvggt import StreamVGGT
    from streamvggt.utils.pose_enc import pose_encoding_to_extri_intri

    device = torch.device("cuda")
    model = StreamVGGT()
    checkpoint = torch.load(args.weights, map_location=device)
    model.load_state_dict(checkpoint, strict=True)
    model.eval().to(device)
    del checkpoint
    dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16

    os.makedirs(args.output_dir, exist_ok=True)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    wall_start = time.perf_counter()
    try:
        if args.mode == "stream":
            pred_c2w, inference_sec, processed_frames, alignment_events = stream_trajectory(
                model, load_images, pose_encoding_to_extri_intri, image_paths, args, device, dtype
            )
        else:
            pred_c2w, inference_sec, processed_frames, alignment_events = window_stitched_trajectory(
                model, load_images, pose_encoding_to_extri_intri, image_paths, args, device, dtype
            )
        wall_sec = time.perf_counter() - wall_start
        prefixes = prefix_metrics(gt_c2w, pred_c2w, args.prefix_frames)
        full = prefixes[-1]
        if full["status"] != "ok":
            raise RuntimeError(f"full trajectory metric failed: {full.get('error')}")
        result = {
            "method": args.method,
            "mode": args.mode,
            "status": "ok",
            "sequence": args.sequence,
            "num_frames": len(image_paths),
            "processed_frames": processed_frames,
            "recompute_factor": processed_frames / len(image_paths),
            "window_size": args.window_size,
            "overlap": args.overlap,
            "cache_window_size": args.cache_window,
            "cache_policy": (
                "full_cache" if args.cache_window is None else args.cache_policy
            ),
            "model_inference_sec": inference_sec,
            "wall_sec": wall_sec,
            "fps_unique_inference": len(image_paths) / inference_sec,
            "fps_processed_inference": processed_frames / inference_sec,
            "peak_allocated_mb": torch.cuda.max_memory_allocated(device) / (1024**2),
            "peak_reserved_mb": torch.cuda.max_memory_reserved(device) / (1024**2),
            "alignment_events": alignment_events,
            "prefix_metrics": prefixes,
            **{key: value for key, value in full.items() if key not in ("prefix_frames", "status")},
        }
        np.savez_compressed(
            osp.join(args.output_dir, "trajectory.npz"),
            gt_c2w=gt_c2w,
            pred_c2w=pred_c2w,
            image_paths=np.asarray(image_paths),
        )
    except Exception as error:
        result = {
            "method": args.method,
            "mode": args.mode,
            "status": "failed",
            "sequence": args.sequence,
            "error": f"{type(error).__name__}: {error}",
        }
        traceback.print_exc()
    with open(osp.join(args.output_dir, "stage3_6a_metrics.json"), "w") as handle:
        json.dump(result, handle, indent=2)
    print(json.dumps(result, indent=2))
    if result["status"] != "ok":
        raise RuntimeError(result["error"])


if __name__ == "__main__":
    main()
