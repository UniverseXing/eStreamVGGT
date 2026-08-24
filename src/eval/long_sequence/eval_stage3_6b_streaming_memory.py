#!/usr/bin/env python3
"""Stage 3.6B true-streaming input/output retention and memory evaluation."""

import argparse
import hashlib
import json
import os
import os.path as osp
import platform
import resource
import sys
import time
import traceback

import numpy as np
import torch
import torch.nn.functional as F

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from add_ckpt_path import add_path_to_dust3r
from eval.long_sequence.eval_stage3_4_long import (
    depth_metrics,
    load_bonn_sequence,
    load_depth_stack,
    temporal_bank_statistics,
    valid_seven_scenes_frames,
)
from eval.pose_evaluation.trajectory_metrics import evaluate_trajectory


REPO_ROOT = osp.abspath(osp.join(osp.dirname(__file__), "../../.."))
DEFAULT_BONN_ROOT = osp.join(REPO_ROOT, "data/eval/bonn/rgbd_bonn_dataset")
DEFAULT_SEVEN_SCENES_ROOT = osp.join(REPO_ROOT, "data/eval/7scenes")


def parse_args():
    parser = argparse.ArgumentParser("Stage 3.6B streaming-memory evaluator")
    parser.add_argument("--weights", required=True)
    parser.add_argument("--dataset", choices=("bonn", "7scenes_raw"), required=True)
    parser.add_argument("--data-root")
    parser.add_argument("--sequence", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--method", required=True)
    parser.add_argument(
        "--mode",
        choices=("legacy_retain", "stream_accumulate", "stream_release"),
        required=True,
    )
    parser.add_argument("--size", type=int, default=518)
    parser.add_argument("--max-frames", type=int, required=True)
    parser.add_argument("--cache-window", type=int, default=8)
    parser.add_argument("--cache-policy", default="temporal_binned_dino_k8")
    parser.add_argument(
        "--full-cache",
        action="store_true",
        help="disable KV pruning; preserves the historical K8 defaults when omitted",
    )
    parser.add_argument("--collect-depth", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--metrics-filename", default="stage3_6b_metrics.json")
    return parser.parse_args()


def max_rss_mib():
    # Linux reports ru_maxrss in KiB. Stage 3 cluster jobs run on Linux.
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def current_rss_mib():
    """Read current resident memory so checkpoint-load peaks cannot mask growth."""
    try:
        with open("/proc/self/statm") as handle:
            resident_pages = int(handle.read().split()[1])
        return resident_pages * os.sysconf("SC_PAGE_SIZE") / (1024**2)
    except (OSError, IndexError, ValueError):
        return max_rss_mib()


def load_sequence(args):
    if args.dataset == "bonn":
        root = osp.abspath(args.data_root or DEFAULT_BONN_ROOT)
        image_paths, depth_paths, gt_c2w, _ = load_bonn_sequence(
            root, args.sequence, args.max_frames
        )
        return image_paths, depth_paths, gt_c2w

    root = osp.abspath(args.data_root or DEFAULT_SEVEN_SCENES_ROOT)
    frames = valid_seven_scenes_frames(root, args.sequence)
    if len(frames) < args.max_frames:
        raise ValueError(
            f"{args.sequence} has {len(frames)} valid frames; "
            f"Stage 3.6B requested {args.max_frames}"
        )
    frames = frames[: args.max_frames]
    return [item[0] for item in frames], None, np.stack([item[1] for item in frames])


class PredictionSink:
    """Move only evaluation-sized products to CPU and release model outputs."""

    def __init__(self, collect_depth):
        self.collect_depth = collect_depth
        self.camera_poses = []
        self.depths = []
        self.pose_hash = hashlib.sha256()
        self.depth_hash = hashlib.sha256()
        self.num_frames = 0
        self.rss_peak_mib = current_rss_mib()

    def __call__(self, frame_index, prediction):
        if frame_index != self.num_frames:
            raise RuntimeError(
                f"non-contiguous sink frame index {frame_index}; expected {self.num_frames}"
            )
        pose = prediction["camera_pose"].detach().float().cpu().contiguous()
        depth = prediction["depth"].detach().float().cpu().squeeze().contiguous()
        self.camera_poses.append(pose)
        if self.collect_depth:
            self.depths.append(depth.numpy())
        self.pose_hash.update(pose.numpy().tobytes())
        self.depth_hash.update(depth.numpy().tobytes())
        self.num_frames += 1
        self.rss_peak_mib = max(self.rss_peak_mib, current_rss_mib())

    def signatures(self):
        return {
            "camera_pose_sha256": self.pose_hash.hexdigest(),
            "depth_sha256": self.depth_hash.hexdigest(),
        }


def lazy_frames(image_paths, load_images, size, device, state):
    for path in image_paths:
        loaded = load_images([path], size=size, crop=False, verbose=False)
        frame = {"img": (loaded[0]["img"].to(device) + 1.0) / 2.0}
        state.setdefault("image_hw", tuple(frame["img"].shape[-2:]))
        yield frame


def preloaded_frames(image_paths, load_images, size, device):
    loaded = load_images(image_paths, size=size, crop=False, verbose=False)
    frames = [{"img": (item["img"].to(device) + 1.0) / 2.0} for item in loaded]
    return frames, tuple(frames[0]["img"].shape[-2:])


def poses_from_sink(sink, image_hw, converter):
    pose_encoding = torch.stack(sink.camera_poses, dim=1)
    pred_w2c, _ = converter(
        pose_encoding,
        image_hw,
        build_intrinsics=False,
    )
    pred_w2c = pred_w2c[0].double().numpy()
    pred_w2c_h = np.tile(np.eye(4, dtype=np.float64), (len(pred_w2c), 1, 1))
    pred_w2c_h[:, :3] = pred_w2c
    return np.linalg.inv(pred_w2c_h)


def trace_summary(memory_trace):
    if not memory_trace:
        return {}

    def maximum(key):
        values = [row.get(key) for row in memory_trace if row.get(key) is not None]
        return max(values) if values else None

    final = memory_trace[-1]
    return {
        "final_aggregator_kv_mib": final.get("aggregator_kv_mib"),
        "max_aggregator_kv_mib": maximum("aggregator_kv_mib"),
        "final_camera_kv_mib": final.get("camera_kv_mib"),
        "max_camera_kv_mib": maximum("camera_kv_mib"),
        "final_descriptor_mib": final.get("descriptor_mib"),
        "max_descriptor_mib": maximum("descriptor_mib"),
        "final_input_tensors_mib": final.get("input_tensors_mib"),
        "max_input_tensors_mib": maximum("input_tensors_mib"),
        "final_retained_outputs_mib": final.get("retained_outputs_mib"),
        "max_retained_outputs_mib": maximum("retained_outputs_mib"),
        "final_retained_views_mib": final.get("retained_views_mib"),
        "max_retained_views_mib": maximum("retained_views_mib"),
        "max_trace_allocated_mib": maximum("cuda_allocated_mib"),
        "final_trace_allocated_mib": final.get("cuda_allocated_mib"),
        "input_mode": final.get("input_mode"),
        "output_mode": final.get("output_mode"),
    }


def main():
    args = parse_args()
    if args.full_cache:
        args.cache_window = None
        args.cache_policy = "fifo"
    if args.max_frames < 2:
        raise ValueError("--max-frames must be at least 2")
    if args.cache_window is not None and args.cache_window < 1:
        raise ValueError("--cache-window must be at least 1 when provided")
    if args.collect_depth and args.dataset != "bonn":
        raise ValueError("--collect-depth is reserved for the 110-frame Bonn equivalence run")

    os.makedirs(args.output_dir, exist_ok=True)
    image_paths, depth_paths, gt_c2w = load_sequence(args)
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
    provenance = {
        "gpu_name": torch.cuda.get_device_name(device),
        "torch_version": str(torch.__version__),
        "cuda_version": torch.version.cuda or "",
        "python_version": platform.python_version(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID", ""),
        "hostname": platform.node(),
    }

    sink = PredictionSink(collect_depth=args.collect_depth)
    output = frames = None
    state = {}
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    rss_before_mib = current_rss_mib()
    wall_start = time.perf_counter()
    try:
        if args.mode == "legacy_retain":
            frames, image_hw = preloaded_frames(
                image_paths, load_images, args.size, device
            )
            inference_start = time.perf_counter()
            with torch.no_grad(), torch.cuda.amp.autocast(dtype=dtype):
                output = model.inference(
                    frames,
                    cache_window_size=args.cache_window,
                    cache_policy=args.cache_policy,
                    return_memory_trace=True,
                )
            torch.cuda.synchronize(device)
            inference_sec = time.perf_counter() - inference_start
            for frame_index, prediction in enumerate(output.ress):
                sink(frame_index, prediction)
        elif args.mode == "stream_accumulate":
            frame_iterator = lazy_frames(
                image_paths, load_images, args.size, device, state
            )
            inference_start = time.perf_counter()
            with torch.no_grad(), torch.cuda.amp.autocast(dtype=dtype):
                output = model.inference(
                    frame_iterator,
                    cache_window_size=args.cache_window,
                    cache_policy=args.cache_policy,
                    return_memory_trace=True,
                    retain_outputs=True,
                    retain_views=False,
                )
            torch.cuda.synchronize(device)
            inference_sec = time.perf_counter() - inference_start
            image_hw = state["image_hw"]
            for frame_index, prediction in enumerate(output.ress):
                sink(frame_index, prediction)
        else:
            frame_iterator = lazy_frames(
                image_paths, load_images, args.size, device, state
            )
            inference_start = time.perf_counter()
            with torch.no_grad(), torch.cuda.amp.autocast(dtype=dtype):
                output = model.inference(
                    frame_iterator,
                    cache_window_size=args.cache_window,
                    cache_policy=args.cache_policy,
                    return_memory_trace=True,
                    output_sink=sink,
                    retain_outputs=False,
                    retain_views=False,
                )
            torch.cuda.synchronize(device)
            inference_sec = time.perf_counter() - inference_start
            image_hw = state["image_hw"]

        wall_sec = time.perf_counter() - wall_start
        peak_allocated_mb = torch.cuda.max_memory_allocated(device) / (1024**2)
        peak_reserved_mb = torch.cuda.max_memory_reserved(device) / (1024**2)
        rss_peak_mib = max(sink.rss_peak_mib, current_rss_mib())
        process_max_rss_mib = max_rss_mib()
        memory_trace = output.memory_trace or []
        if sink.num_frames != len(image_paths):
            raise RuntimeError(
                f"sink received {sink.num_frames} predictions for {len(image_paths)} frames"
            )

        pred_c2w = poses_from_sink(sink, image_hw, pose_encoding_to_extri_intri)
        pose_result = {"pose_status": "ok"}
        try:
            pose_metrics = evaluate_trajectory(gt_c2w, pred_c2w)
            pose_metrics.pop("pred_c2w_aligned")
            pose_result.update(pose_metrics)
        except Exception as error:
            pose_result.update(
                {
                    "pose_status": "failed",
                    "pose_error": f"{type(error).__name__}: {error}",
                    "ate": None,
                    "rpe_trans": None,
                    "rpe_rot_deg": None,
                }
            )

        depth_result = {}
        if args.collect_depth:
            if depth_paths is None:
                raise RuntimeError("depth collection requested without GT depth paths")
            pred_depth = np.stack(sink.depths)
            gt_depth = load_depth_stack(depth_paths)
            pred_depth = (
                F.interpolate(
                    torch.from_numpy(pred_depth).unsqueeze(1),
                    size=gt_depth.shape[1:],
                    mode="bicubic",
                    align_corners=False,
                )
                .squeeze(1)
                .numpy()
            )
            depth_result = depth_metrics(pred_depth, gt_depth, len(pred_depth))

        np.savez_compressed(
            osp.join(args.output_dir, "trajectory.npz"),
            gt_c2w=gt_c2w,
            pred_c2w=pred_c2w,
            image_paths=np.asarray(image_paths),
        )
        with open(osp.join(args.output_dir, "memory_trace.json"), "w") as handle:
            json.dump(memory_trace, handle)

        result = {
            "method": args.method,
            "mode": args.mode,
            "status": "ok",
            "dataset": args.dataset,
            "sequence": args.sequence,
            "num_frames": len(image_paths),
            "processed_frames": sink.num_frames,
            "cache_window_size": args.cache_window,
            "cache_policy": (
                "full_cache" if args.cache_window is None else args.cache_policy
            ),
            "collect_depth": args.collect_depth,
            "inference_sec": inference_sec,
            "wall_sec": wall_sec,
            "fps_inference": len(image_paths) / inference_sec,
            "fps_end_to_end": len(image_paths) / wall_sec,
            "peak_allocated_mb": peak_allocated_mb,
            "peak_reserved_mb": peak_reserved_mb,
            "rss_before_mib": rss_before_mib,
            "rss_peak_mib": rss_peak_mib,
            "rss_growth_mib": max(0.0, rss_peak_mib - rss_before_mib),
            "process_max_rss_mib": process_max_rss_mib,
            **sink.signatures(),
            **trace_summary(memory_trace),
            "temporal_bank_statistics": temporal_bank_statistics(memory_trace),
            **pose_result,
            **depth_result,
            **provenance,
        }
    except Exception as error:
        result = {
            "method": args.method,
            "mode": args.mode,
            "status": "failed",
            "dataset": args.dataset,
            "sequence": args.sequence,
            "num_frames": len(image_paths),
            "processed_frames": sink.num_frames,
            "cache_window_size": args.cache_window,
            "cache_policy": (
                "full_cache" if args.cache_window is None else args.cache_policy
            ),
            "error": f"{type(error).__name__}: {error}",
            "peak_allocated_mb": torch.cuda.max_memory_allocated(device) / (1024**2),
            "peak_reserved_mb": torch.cuda.max_memory_reserved(device) / (1024**2),
            **provenance,
        }
        traceback.print_exc()

    with open(osp.join(args.output_dir, args.metrics_filename), "w") as handle:
        json.dump(result, handle, indent=2)
    print(json.dumps(result, indent=2))
    if result["status"] != "ok":
        raise RuntimeError(result["error"])


if __name__ == "__main__":
    main()
