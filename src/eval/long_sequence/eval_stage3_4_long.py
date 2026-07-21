#!/usr/bin/env python3
"""Stage 3.4 long-sequence evaluation without large depth/point-cloud artifacts."""

import argparse
import json
import os
import os.path as osp
import sys
import time
import traceback

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from add_ckpt_path import add_path_to_dust3r
from eval.pose_evaluation.pose_datasets import read_tum_trajectory
from eval.pose_evaluation.trajectory_metrics import evaluate_trajectory


REPO_ROOT = osp.abspath(osp.join(osp.dirname(__file__), "../../.."))
BONN_SEQUENCES = (
    "balloon2",
    "crowd2",
    "crowd3",
    "person_tracking2",
    "synchronous",
)
DEFAULT_ROOTS = {
    "bonn": osp.join(REPO_ROOT, "data/eval/bonn/rgbd_bonn_dataset"),
    "7scenes_loop": osp.join(REPO_ROOT, "data/eval/7scenes"),
}


def parse_args():
    parser = argparse.ArgumentParser("Stage 3.4 long-sequence evaluator")
    parser.add_argument("--weights", required=True)
    parser.add_argument("--dataset", choices=("bonn", "7scenes_loop"), required=True)
    parser.add_argument("--data-root")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seq-list", nargs="+")
    parser.add_argument("--max-sequences", type=int)
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--prefix-frames", type=int, nargs="+", required=True)
    parser.add_argument("--loop-forward-frames", type=int, default=50)
    parser.add_argument("--size", type=int, default=518)
    parser.add_argument("--cache-window", type=int)
    parser.add_argument("--cache-policy", default="fifo")
    parser.add_argument("--camera-cache-window", type=int)
    parser.add_argument(
        "--camera-cache-policy",
        help=(
            "omit for the legacy coupled cache, use 'full' for an independent "
            "uncropped camera cache, or provide a bounded policy together with "
            "--camera-cache-window"
        ),
    )
    parser.add_argument("--trace-memory", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def cache_name(args):
    return "full_cache" if args.cache_window is None else args.cache_policy


def camera_cache_config(args):
    if args.camera_cache_policy is None:
        return "coupled", args.cache_window
    if args.camera_cache_policy == "full":
        return "full_cache", None
    return args.camera_cache_policy, args.camera_cache_window


def resolve_bonn_dir(scene_root, kind):
    for dirname in (f"{kind}_110_sampled", f"{kind}_110", kind):
        path = osp.join(scene_root, dirname)
        if osp.isdir(path):
            return path
    raise FileNotFoundError(f"missing Bonn {kind} directory below {scene_root}")


def image_files(path):
    return sorted(
        osp.join(path, name)
        for name in os.listdir(path)
        if osp.splitext(name)[1].lower() in (".png", ".jpg", ".jpeg")
    )


def read_bonn_poses(scene_root, rgb_paths):
    aligned_path = osp.join(scene_root, "groundtruth_110.txt")
    if osp.isfile(aligned_path):
        poses = read_tum_trajectory(aligned_path)
        if len(poses) != len(rgb_paths):
            raise ValueError(
                f"aligned Bonn pose count mismatch: {len(poses)} poses for "
                f"{len(rgb_paths)} RGB frames in {scene_root}"
            )
        return poses

    trajectory_path = osp.join(scene_root, "groundtruth.txt")
    values = np.atleast_2d(np.loadtxt(trajectory_path, comments="#", dtype=np.float64))
    if values.shape[1] != 8:
        raise ValueError(f"expected 8-column TUM trajectory in {trajectory_path}")
    trajectory_poses = read_tum_trajectory(trajectory_path)
    timestamps = values[:, 0]
    matched = []
    for path in rgb_paths:
        try:
            # The prepared 110-frame directories use zero-padded symlink names;
            # the original timestamp remains in each symlink target.
            timestamp = float(osp.splitext(osp.basename(osp.realpath(path)))[0])
        except ValueError as error:
            raise ValueError(f"Bonn RGB filename is not a timestamp: {path}") from error
        index = int(np.argmin(np.abs(timestamps - timestamp)))
        matched.append(trajectory_poses[index])
    return np.stack(matched)


def load_bonn_sequence(root, sequence, max_frames):
    scene_root = osp.join(root, f"rgbd_bonn_{sequence}")
    rgb_paths = image_files(resolve_bonn_dir(scene_root, "rgb"))
    depth_paths = image_files(resolve_bonn_dir(scene_root, "depth"))
    gt_c2w = read_bonn_poses(scene_root, rgb_paths)
    count = min(len(rgb_paths), len(depth_paths), len(gt_c2w))
    if count < 2 or (len(rgb_paths), len(depth_paths), len(gt_c2w)) != (count, count, count):
        raise ValueError(
            f"{sequence}: Bonn RGB/depth/pose mismatch: "
            f"{len(rgb_paths)}/{len(depth_paths)}/{len(gt_c2w)}"
        )
    if max_frames is not None:
        count = min(count, max_frames)
    return rgb_paths[:count], depth_paths[:count], gt_c2w[:count], None


def seven_scenes_test_sequences(root):
    sequences = []
    for scene in sorted(os.listdir(root)):
        split_path = osp.join(root, scene, "TestSplit.txt")
        if not osp.isfile(split_path):
            continue
        with open(split_path) as handle:
            for line in handle:
                digits = "".join(character for character in line if character.isdigit())
                sequences.append(f"{scene}/seq-{digits.zfill(2)}")
    return sequences


def valid_seven_scenes_frames(root, sequence):
    scene_root = osp.join(root, sequence)
    frames = []
    for name in sorted(os.listdir(scene_root)):
        if not name.startswith("frame-") or not name.endswith(".color.png"):
            continue
        frame_id = name[len("frame-") : -len(".color.png")]
        pose_path = osp.join(scene_root, f"frame-{frame_id}.pose.txt")
        try:
            pose = np.loadtxt(pose_path).astype(np.float64)
        except (OSError, ValueError):
            continue
        if pose.shape == (4, 4) and np.isfinite(pose).all():
            frames.append((osp.join(scene_root, name), pose))
    return frames


def eligible_seven_scenes_sequences(root, loop_forward_frames):
    eligible = []
    excluded = []
    for sequence in seven_scenes_test_sequences(root):
        valid_count = len(valid_seven_scenes_frames(root, sequence))
        if valid_count >= loop_forward_frames:
            eligible.append(sequence)
        else:
            excluded.append((sequence, valid_count))
    if excluded:
        print(
            "[7Scenes-loop] excluded sequences with fewer than "
            f"{loop_forward_frames} valid poses: {excluded}"
        )
    return eligible


def load_seven_scenes_loop(root, sequence, loop_forward_frames, max_frames):
    frames = valid_seven_scenes_frames(root, sequence)
    if len(frames) < loop_forward_frames:
        raise ValueError(
            f"{sequence}: only {len(frames)} valid frames; need {loop_forward_frames}"
        )
    forward = frames[:loop_forward_frames]
    loop = forward + list(reversed(forward))
    if max_frames is not None:
        loop = loop[:max_frames]
    image_paths = [item[0] for item in loop]
    gt_c2w = np.stack([item[1] for item in loop])
    return image_paths, None, gt_c2w, loop_forward_frames


def load_depth_stack(paths):
    depths = []
    for path in paths:
        raw = np.asarray(Image.open(path))
        if raw.dtype == np.uint8 or raw.max(initial=0) <= 255:
            raise ValueError(f"expected 16-bit Bonn depth: {path}")
        depth = raw.astype(np.float32) / 5000.0
        depth[raw == 0] = -1.0
        depths.append(depth)
    return np.stack(depths)


def rotation_angle_deg(rotation):
    u, _, vh = np.linalg.svd(rotation)
    rotation = u @ vh
    if np.linalg.det(rotation) < 0:
        u[:, -1] *= -1
        rotation = u @ vh
    cosine = np.clip((np.trace(rotation) - 1.0) / 2.0, -1.0, 1.0)
    return float(np.degrees(np.arccos(cosine)))


def loop_pose_consistency(aligned_c2w, prefix, loop_forward_frames):
    if loop_forward_frames is None or prefix <= loop_forward_frames:
        return {}
    translations = []
    rotations = []
    total_length = 2 * loop_forward_frames
    for reverse_index in range(loop_forward_frames, min(prefix, total_length)):
        forward_index = total_length - 1 - reverse_index
        relative = np.linalg.inv(aligned_c2w[forward_index]) @ aligned_c2w[reverse_index]
        translations.append(float(np.linalg.norm(relative[:3, 3])))
        rotations.append(rotation_angle_deg(relative[:3, :3]))
    if not translations:
        return {}
    return {
        "loop_pairs": len(translations),
        "loop_translation_mean": float(np.mean(translations)),
        "loop_translation_max": float(np.max(translations)),
        "loop_rotation_deg_mean": float(np.mean(rotations)),
        "loop_rotation_deg_max": float(np.max(rotations)),
    }


def loop_depth_consistency(pred_depth, prefix, loop_forward_frames):
    if loop_forward_frames is None or prefix <= loop_forward_frames:
        return {}
    errors = []
    total_length = 2 * loop_forward_frames
    for reverse_index in range(loop_forward_frames, min(prefix, total_length)):
        forward_index = total_length - 1 - reverse_index
        forward = pred_depth[forward_index]
        reverse = pred_depth[reverse_index]
        valid = np.isfinite(forward) & np.isfinite(reverse) & (forward > 1e-5) & (reverse > 1e-5)
        if not np.any(valid):
            continue
        scale = np.median(forward[valid] / reverse[valid])
        errors.append(float(np.mean(np.abs(scale * reverse[valid] - forward[valid]) / forward[valid])))
    return {
        "loop_depth_pairs": len(errors),
        "loop_depth_abs_rel": float(np.mean(errors)) if errors else None,
    }


def selection_statistics(
    memory_trace,
    cache_window,
    loop_forward_frames,
    frame_ids_key="retained_frame_ids",
):
    if not memory_trace:
        return {}
    rows = [
        row
        for row in memory_trace
        if cache_window is None or len(row.get(frame_ids_key, [])) >= cache_window
    ]
    ages = []
    spans = []
    anchor_hits = 0
    churn = []
    unique_ids = set()
    previous = None
    loop_hits = 0
    loop_total = 0
    total_length = 2 * loop_forward_frames if loop_forward_frames is not None else None
    for row in rows:
        step = int(row["frame_index"])
        retained = [int(item) for item in row.get(frame_ids_key, [])]
        historical = [item for item in retained if item != step]
        unique_ids.update(historical)
        ages.extend(step - item for item in historical)
        if retained:
            spans.append(max(retained) - min(retained))
            anchor_hits += int(0 in retained)
        current = set(retained)
        if previous is not None and current:
            churn.append(1.0 - len(previous & current) / max(len(previous), len(current)))
        previous = current
        if total_length is not None and loop_forward_frames <= step < total_length:
            loop_total += 1
            matching_forward = total_length - 1 - step
            loop_hits += int(matching_forward in retained)
    return {
        "selection_observations": len(rows),
        "mean_retained_age": float(np.mean(ages)) if ages else None,
        "max_retained_age": int(max(ages)) if ages else None,
        "mean_temporal_span": float(np.mean(spans)) if spans else None,
        "anchor0_retention_rate": anchor_hits / len(rows) if rows else None,
        "mean_selection_churn": float(np.mean(churn)) if churn else None,
        "unique_retained_frames": len(unique_ids),
        "loop_match_retention_rate": loop_hits / loop_total if loop_total else None,
        "final_retained_frame_ids": memory_trace[-1].get(frame_ids_key, []),
    }


def resource_at_prefix(memory_trace, frame_timings, prefix):
    result = {
        "mean_frame_latency_ms": (
            float(np.mean(frame_timings[:prefix])) if frame_timings else None
        ),
        "last_frame_latency_ms": (
            float(frame_timings[prefix - 1]) if frame_timings else None
        ),
    }
    if memory_trace and prefix <= len(memory_trace):
        row = memory_trace[prefix - 1]
        for key in (
            "aggregator_kv_mib",
            "camera_kv_mib",
            "descriptor_mib",
            "input_tensors_mib",
            "retained_outputs_mib",
            "cuda_allocated_mib",
            "cuda_reserved_mib",
        ):
            result[key] = row.get(key)
    return result


def depth_metrics(pred_depth, gt_depth, prefix):
    from eval.video_depth.tools import depth_evaluation

    metrics, error_map, aligned_depth, gt_map = depth_evaluation(
        pred_depth[:prefix],
        gt_depth[:prefix],
        max_depth=70,
        align_with_scale=True,
        use_gpu=True,
    )
    result = {
        "abs_rel": float(metrics["Abs Rel"]),
        "sq_rel": float(metrics["Sq Rel"]),
        "rmse": float(metrics["RMSE"]),
        "log_rmse": float(metrics["Log RMSE"]),
        "delta_1": float(metrics["δ < 1.25"]),
        "delta_2": float(metrics["δ < 1.25^2"]),
        "delta_3": float(metrics["δ < 1.25^3"]),
        "valid_pixels": int(metrics["valid_pixels"]),
    }
    del error_map, aligned_depth, gt_map
    torch.cuda.empty_cache()
    return result


def evaluate_prefixes(
    pred_c2w,
    gt_c2w,
    pred_depth,
    gt_depth,
    requested_prefixes,
    memory_trace,
    frame_timings,
    loop_forward_frames,
):
    prefixes = sorted({prefix for prefix in requested_prefixes if 2 <= prefix <= len(pred_c2w)} | {len(pred_c2w)})
    results = []
    for prefix in prefixes:
        row = {"prefix_frames": prefix}
        try:
            pose = evaluate_trajectory(gt_c2w[:prefix], pred_c2w[:prefix])
            aligned = pose.pop("pred_c2w_aligned")
            row.update({"pose_status": "ok", **pose})
            row.update(loop_pose_consistency(aligned, prefix, loop_forward_frames))
        except Exception as error:
            row.update(
                {
                    "pose_status": "failed",
                    "pose_error": f"{type(error).__name__}: {error}",
                    "ate": None,
                    "rpe_trans": None,
                    "rpe_rot_deg": None,
                }
            )
        if gt_depth is not None:
            row.update(depth_metrics(pred_depth, gt_depth, prefix))
        row.update(loop_depth_consistency(pred_depth, prefix, loop_forward_frames))
        row.update(resource_at_prefix(memory_trace, frame_timings, prefix))
        results.append(row)
    return results


def mean_successful(rows, key):
    values = [row[key] for row in rows if row.get("status") == "ok" and row.get(key) is not None]
    return float(np.mean(values)) if values else None


def summarize(dataset, args, rows):
    successful = [row for row in rows if row["status"] == "ok"]
    failed = [row for row in rows if row["status"] != "ok"]
    total_frames = sum(row["num_frames"] for row in successful)
    total_inference = sum(row["inference_sec"] for row in successful)
    camera_policy, camera_window = camera_cache_config(args)
    summary = {
        "dataset": dataset,
        "cache_policy": cache_name(args),
        "cache_window_size": args.cache_window,
        "camera_cache_policy": camera_policy,
        "camera_cache_window_size": camera_window,
        "num_sequences": len(rows),
        "num_successful": len(successful),
        "num_failed": len(failed),
        "total_frames": total_frames,
        "total_inference_sec": total_inference,
        "fps_inference": total_frames / total_inference if total_inference else None,
        "max_peak_allocated_mb": max((row["peak_allocated_mb"] for row in successful), default=None),
        "max_peak_reserved_mb": max((row["peak_reserved_mb"] for row in successful), default=None),
    }
    for key in ("abs_rel", "rmse", "delta_1", "ate", "rpe_trans", "rpe_rot_deg"):
        summary[f"mean_{key}"] = mean_successful(rows, key)
    return summary


def main():
    args = parse_args()
    if args.max_sequences is not None and args.max_sequences < 1:
        raise ValueError("--max-sequences must be at least 1")
    if args.max_frames is not None and args.max_frames < 2:
        raise ValueError("--max-frames must be at least 2")
    if any(prefix < 2 for prefix in args.prefix_frames):
        raise ValueError("--prefix-frames values must be at least 2")

    root = osp.abspath(args.data_root or DEFAULT_ROOTS[args.dataset])
    if not osp.isdir(root):
        raise FileNotFoundError(f"missing {args.dataset} root: {root}")
    if args.seq_list:
        sequences = args.seq_list
    elif args.dataset == "bonn":
        sequences = list(BONN_SEQUENCES)
    else:
        sequences = eligible_seven_scenes_sequences(root, args.loop_forward_frames)
    if args.max_sequences is not None:
        sequences = sequences[: args.max_sequences]
    if not sequences:
        raise RuntimeError(f"no eligible {args.dataset} sequences below {root}")

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
    trajectory_dir = osp.join(args.output_dir, "trajectories")
    trace_dir = osp.join(args.output_dir, "memory_traces")
    os.makedirs(trajectory_dir, exist_ok=True)
    if args.trace_memory:
        os.makedirs(trace_dir, exist_ok=True)

    rows = []
    for sequence in sequences:
        print(f"Evaluating Stage 3.4 {args.dataset}/{sequence} with {cache_name(args)}")
        output = frames = loaded = None
        try:
            if args.dataset == "bonn":
                image_paths, depth_paths, gt_c2w, loop_forward_frames = load_bonn_sequence(
                    root, sequence, args.max_frames
                )
            else:
                image_paths, depth_paths, gt_c2w, loop_forward_frames = load_seven_scenes_loop(
                    root, sequence, args.loop_forward_frames, args.max_frames
                )
            loaded = load_images(image_paths, size=args.size, crop=False)
            frames = [{"img": (item["img"].to(device) + 1.0) / 2.0} for item in loaded]
            del loaded

            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)
            torch.cuda.synchronize(device)
            start = time.perf_counter()
            with torch.no_grad(), torch.cuda.amp.autocast(dtype=dtype):
                output = model.inference(
                    frames,
                    cache_window_size=args.cache_window,
                    cache_policy=args.cache_policy,
                    camera_cache_window_size=args.camera_cache_window,
                    camera_cache_policy=args.camera_cache_policy,
                    return_memory_events=False,
                    return_memory_trace=args.trace_memory,
                    return_frame_timings=True,
                )
            torch.cuda.synchronize(device)
            inference_sec = time.perf_counter() - start
            peak_allocated = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
            peak_reserved = torch.cuda.max_memory_reserved(device) / (1024 ** 2)

            pose_encoding = torch.stack(
                [prediction["camera_pose"] for prediction in output.ress], dim=1
            )
            pred_w2c, _ = pose_encoding_to_extri_intri(
                pose_encoding,
                frames[0]["img"].shape[-2:],
                build_intrinsics=False,
            )
            pred_w2c = pred_w2c[0].detach().cpu().double().numpy()
            pred_w2c_h = np.tile(np.eye(4, dtype=np.float64), (len(pred_w2c), 1, 1))
            pred_w2c_h[:, :3] = pred_w2c
            pred_c2w = np.linalg.inv(pred_w2c_h)
            pred_depth = np.stack(
                [np.squeeze(prediction["depth"].detach().float().cpu().numpy()) for prediction in output.ress]
            )
            memory_trace = output.memory_trace or []
            frame_timings = output.frame_inference_ms or []
            selection = selection_statistics(memory_trace, args.cache_window, loop_forward_frames)
            camera_policy, camera_window = camera_cache_config(args)
            camera_selection = selection_statistics(
                memory_trace,
                camera_window,
                loop_forward_frames,
                frame_ids_key="camera_retained_frame_ids",
            )

            np.savez_compressed(
                osp.join(trajectory_dir, sequence.replace("/", "_") + ".npz"),
                gt_c2w=gt_c2w,
                pred_c2w=pred_c2w,
                image_paths=np.asarray(image_paths),
            )
            if args.trace_memory:
                with open(
                    osp.join(trace_dir, sequence.replace("/", "_") + ".json"), "w"
                ) as handle:
                    json.dump(memory_trace, handle, indent=2)

            output = frames = None
            torch.cuda.empty_cache()
            gt_depth = load_depth_stack(depth_paths) if depth_paths is not None else None
            if gt_depth is not None:
                pred_depth_for_metrics = (
                    F.interpolate(
                        torch.from_numpy(pred_depth).unsqueeze(1),
                        size=gt_depth.shape[1:],
                        mode="bicubic",
                        align_corners=False,
                    )
                    .squeeze(1)
                    .numpy()
                )
            else:
                pred_depth_for_metrics = pred_depth

            prefix_metrics = evaluate_prefixes(
                pred_c2w,
                gt_c2w,
                pred_depth_for_metrics,
                gt_depth,
                args.prefix_frames,
                memory_trace,
                frame_timings,
                loop_forward_frames,
            )
            full = prefix_metrics[-1]
            result = {
                "sequence": sequence,
                "status": "ok",
                "num_frames": len(image_paths),
                "inference_sec": inference_sec,
                "fps_inference": len(image_paths) / inference_sec,
                "peak_allocated_mb": peak_allocated,
                "peak_reserved_mb": peak_reserved,
                "prefix_metrics": prefix_metrics,
                "selection_statistics": selection,
                "camera_selection_statistics": camera_selection,
                **{key: value for key, value in full.items() if key != "prefix_frames"},
            }
            rows.append(result)
            print(json.dumps(result, sort_keys=True))
        except Exception as error:
            failure = {
                "sequence": sequence,
                "status": "failed",
                "error": f"{type(error).__name__}: {error}",
            }
            rows.append(failure)
            print(json.dumps(failure), file=sys.stderr)
            traceback.print_exc()
        finally:
            output = frames = loaded = None
            torch.cuda.empty_cache()

    summary = summarize(args.dataset, args, rows)
    camera_policy, camera_window = camera_cache_config(args)
    payload = {
        "dataset": args.dataset,
        "cache_policy": cache_name(args),
        "cache_window_size": args.cache_window,
        "camera_cache_policy": camera_policy,
        "camera_cache_window_size": camera_window,
        "prefix_frames": sorted(set(args.prefix_frames)),
        "loop_forward_frames": args.loop_forward_frames if args.dataset == "7scenes_loop" else None,
        "summary": summary,
        "sequences": rows,
    }
    with open(osp.join(args.output_dir, "stage3_4_metrics.json"), "w") as handle:
        json.dump(payload, handle, indent=2)
    print(json.dumps(summary, indent=2))
    if summary["num_successful"] == 0:
        raise RuntimeError(f"all {len(rows)} Stage 3.4 sequences failed")


if __name__ == "__main__":
    main()
