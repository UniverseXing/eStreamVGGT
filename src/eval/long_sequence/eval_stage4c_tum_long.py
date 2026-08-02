#!/usr/bin/env python3
"""Stage 4C frozen long-sequence evaluation on raw TUM RGB-D sequences."""

import argparse
import json
import os
import os.path as osp
import platform
import sys
import time
import traceback

import numpy as np
import torch
from scipy.spatial.transform import Rotation

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from add_ckpt_path import add_path_to_dust3r
from eval.long_sequence.eval_stage3_6b_streaming_memory import (
    PredictionSink,
    current_rss_mib,
    lazy_frames,
    max_rss_mib,
    poses_from_sink,
    trace_summary,
)
from eval.long_sequence.eval_stage3_4_long import temporal_bank_statistics
from eval.pose_evaluation.trajectory_metrics import evaluate_trajectory


REPO_ROOT = osp.abspath(osp.join(osp.dirname(__file__), "../../.."))
DEFAULT_ROOT = osp.join(REPO_ROOT, "data/eval/stage4c_tum")
PUBLIC_METHOD_CONFIGS = {
    "full_cache": (None, "fifo"),
    "anchor_recent_dino_diverse_k4": (4, "anchor_recent_dino_diverse_k4"),
    "anchor_recent_dino_diverse_k6": (6, "anchor_recent_dino_diverse_k6"),
    "anchor_recent_dino_diverse_k8": (8, "anchor_recent_dino_diverse_k8"),
}

# Keep the frozen experiment identifiers accepted so the historical Stage 4C
# commands and result manifests remain reproducible. New commands should use
# the paper-facing names above.
LEGACY_METHOD_CONFIGS = {
    "stage3_2_k4": (4, "anchor_recent_dino_diverse_2old_1recent"),
    "old_dino_k6": (6, "anchor_recent_dino_diverse"),
    "temporal_binned_dino_k8": (8, "temporal_binned_dino_k8"),
}
METHOD_CONFIGS = {**PUBLIC_METHOD_CONFIGS, **LEGACY_METHOD_CONFIGS}


def parse_args():
    parser = argparse.ArgumentParser("Stage 4C raw-TUM long-sequence evaluator")
    parser.add_argument("--weights", required=True)
    parser.add_argument("--data-root", default=DEFAULT_ROOT)
    parser.add_argument("--sequence", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--method", choices=tuple(METHOD_CONFIGS), required=True)
    parser.add_argument("--max-frames", type=int, required=True)
    parser.add_argument("--size", type=int, default=518)
    parser.add_argument("--max-association-difference", type=float, default=0.02)
    parser.add_argument(
        "--run-scope",
        choices=("frozen", "debug_subset"),
        default="frozen",
        help="label whether this cell follows the frozen matrix or an explicit debug override",
    )
    parser.add_argument(
        "--run-id",
        default="",
        help="optional caller-generated identifier used to reject stale result files",
    )
    return parser.parse_args()


def read_tum_list(path):
    rows = []
    with open(path) as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            fields = stripped.split()
            rows.append((float(fields[0]), fields[1:]))
    if not rows:
        raise ValueError(f"no records in {path}")
    return rows


def load_raw_tum_sequence(root, sequence, max_frames, max_difference):
    sequence_root = osp.join(osp.abspath(root), sequence)
    rgb_rows = read_tum_list(osp.join(sequence_root, "rgb.txt"))
    gt_rows = read_tum_list(osp.join(sequence_root, "groundtruth.txt"))
    gt_timestamps = np.asarray([row[0] for row in gt_rows], dtype=np.float64)

    image_paths = []
    poses = []
    association_differences = []
    for timestamp, fields in rgb_rows:
        if not fields:
            continue
        insertion = int(np.searchsorted(gt_timestamps, timestamp))
        candidates = [
            index
            for index in (insertion - 1, insertion)
            if 0 <= index < len(gt_rows)
        ]
        if not candidates:
            continue
        best = min(
            candidates,
            key=lambda index: abs(gt_timestamps[index] - timestamp),
        )
        difference = abs(gt_timestamps[best] - timestamp)
        if difference > max_difference:
            continue
        values = [float(value) for value in gt_rows[best][1]]
        if len(values) != 7:
            raise ValueError(
                f"expected 7 pose values in {sequence}/groundtruth.txt"
            )
        pose = np.eye(4, dtype=np.float64)
        pose[:3, :3] = Rotation.from_quat(values[3:7]).as_matrix()
        pose[:3, 3] = values[:3]
        image_path = osp.join(sequence_root, fields[0])
        if not osp.isfile(image_path):
            raise FileNotFoundError(image_path)
        image_paths.append(image_path)
        poses.append(pose)
        association_differences.append(difference)
        if len(image_paths) == max_frames:
            break

    if len(image_paths) < max_frames:
        raise ValueError(
            f"{sequence} has only {len(image_paths)} RGB/GT associations "
            f"within {max_difference}s; requested {max_frames}"
        )
    return (
        image_paths,
        np.stack(poses),
        max(association_differences),
    )


def provenance():
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else ""
    return {
        "gpu_name": gpu_name,
        "torch_version": str(torch.__version__),
        "cuda_version": torch.version.cuda,
        "python_version": platform.python_version(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID", ""),
        "hostname": platform.node(),
    }


def main():
    args = parse_args()
    if args.max_frames < 2:
        raise ValueError("--max-frames must be at least 2")
    cache_window, cache_policy = METHOD_CONFIGS[args.method]
    os.makedirs(args.output_dir, exist_ok=True)
    metrics_path = osp.join(args.output_dir, "stage4c_metrics.json")
    result = {
        "run_scope": args.run_scope,
        "method": args.method,
        "status": "failed",
        "dataset": "tum_rgbd_raw",
        "sequence": args.sequence,
        "num_frames": args.max_frames,
        "cache_window_size": cache_window,
        "cache_policy": "full_cache" if cache_window is None else cache_policy,
        "mode": "stream_release",
        "run_id": args.run_id,
        **provenance(),
    }

    sink = None
    try:
        image_paths, gt_c2w, max_difference = load_raw_tum_sequence(
            args.data_root,
            args.sequence,
            args.max_frames,
            args.max_association_difference,
        )
        result["max_association_difference_sec"] = max_difference

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
        dtype = (
            torch.bfloat16
            if torch.cuda.get_device_capability()[0] >= 8
            else torch.float16
        )

        sink = PredictionSink(collect_depth=False)
        state = {}
        frames = lazy_frames(
            image_paths, load_images, args.size, device, state
        )
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        rss_before_mib = current_rss_mib()
        wall_start = time.perf_counter()
        inference_start = time.perf_counter()
        with torch.no_grad(), torch.cuda.amp.autocast(dtype=dtype):
            output = model.inference(
                frames,
                cache_window_size=cache_window,
                cache_policy=cache_policy,
                return_memory_trace=True,
                output_sink=sink,
                retain_outputs=False,
                retain_views=False,
            )
        torch.cuda.synchronize(device)
        inference_sec = time.perf_counter() - inference_start
        wall_sec = time.perf_counter() - wall_start
        if sink.num_frames != len(image_paths):
            raise RuntimeError(
                f"sink received {sink.num_frames}/{len(image_paths)} frames"
            )

        pred_c2w = poses_from_sink(
            sink, state["image_hw"], pose_encoding_to_extri_intri
        )
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

        memory_trace = output.memory_trace or []
        rss_peak_mib = max(sink.rss_peak_mib, current_rss_mib())
        np.savez_compressed(
            osp.join(args.output_dir, "trajectory.npz"),
            gt_c2w=gt_c2w,
            pred_c2w=pred_c2w,
            image_paths=np.asarray(image_paths),
        )
        with open(
            osp.join(args.output_dir, "memory_trace.json"), "w"
        ) as handle:
            json.dump(memory_trace, handle)

        result.update(
            {
                "status": "ok",
                "processed_frames": sink.num_frames,
                "inference_sec": inference_sec,
                "wall_sec": wall_sec,
                "fps_inference": len(image_paths) / inference_sec,
                "fps_end_to_end": len(image_paths) / wall_sec,
                "peak_allocated_mb": torch.cuda.max_memory_allocated(device)
                / (1024**2),
                "peak_reserved_mb": torch.cuda.max_memory_reserved(device)
                / (1024**2),
                "rss_before_mib": rss_before_mib,
                "rss_peak_mib": rss_peak_mib,
                "rss_growth_mib": max(0.0, rss_peak_mib - rss_before_mib),
                "process_max_rss_mib": max_rss_mib(),
                **sink.signatures(),
                **trace_summary(memory_trace),
                "temporal_bank_statistics": temporal_bank_statistics(
                    memory_trace
                ),
                **pose_result,
            }
        )
    except Exception as error:
        result["error"] = f"{type(error).__name__}: {error}"
        result["processed_frames"] = sink.num_frames if sink is not None else 0
        if torch.cuda.is_available():
            result["peak_allocated_mb"] = (
                torch.cuda.max_memory_allocated() / (1024**2)
            )
            result["peak_reserved_mb"] = (
                torch.cuda.max_memory_reserved() / (1024**2)
            )
        traceback.print_exc()

    with open(metrics_path, "w") as handle:
        json.dump(result, handle, indent=2)
    print(json.dumps(result, indent=2))
    if result["status"] != "ok":
        raise RuntimeError(result["error"])


if __name__ == "__main__":
    main()
