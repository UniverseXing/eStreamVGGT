import argparse
import json
import os
import platform
import sys
import time
import traceback

import numpy as np
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from add_ckpt_path import add_path_to_dust3r
from eval.pose_evaluation.pose_datasets import (
    DEFAULT_ANNO_ROOTS,
    DEFAULT_ROOTS,
    discover_sequences,
    load_pose_sequence,
)
from eval.pose_evaluation.trajectory_metrics import evaluate_trajectory


def parse_args():
    parser = argparse.ArgumentParser("Cache-aware StreamVGGT pose evaluation")
    parser.add_argument("--weights", required=True)
    parser.add_argument("--dataset", choices=("sintel", "scannet", "tum"), required=True)
    parser.add_argument("--data-root")
    parser.add_argument("--anno-root")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seq-list", nargs="+")
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--size", type=int, default=518)
    parser.add_argument("--cache-window", type=int)
    parser.add_argument("--cache-policy", default="fifo")
    parser.add_argument("--cache-random-seed", type=int, default=0)
    parser.add_argument("--log-selections", action="store_true")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="replace evaluated sequences in an existing pose_metrics.json and keep the rest",
    )
    return parser.parse_args()


def cache_name(args):
    if args.cache_window is None:
        return "full_cache"
    return f"{args.cache_policy}_k{args.cache_window}"


def main():
    args = parse_args()
    if args.stride < 1:
        raise ValueError("--stride must be at least 1")
    if args.max_frames is not None and args.max_frames < 2:
        raise ValueError("--max-frames must be at least 2")

    data_root = args.data_root or DEFAULT_ROOTS[args.dataset]
    anno_root = args.anno_root or DEFAULT_ANNO_ROOTS.get(args.dataset)
    sequences = args.seq_list or discover_sequences(args.dataset, data_root, anno_root)
    if not sequences:
        raise RuntimeError(f"no valid {args.dataset} pose sequences found in {data_root}")

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

    os.makedirs(args.output_dir, exist_ok=True)
    pose_dir = os.path.join(args.output_dir, "trajectories")
    selection_dir = os.path.join(args.output_dir, "memory_selections")
    os.makedirs(pose_dir, exist_ok=True)
    if args.log_selections:
        os.makedirs(selection_dir, exist_ok=True)

    dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    metrics_path = os.path.join(args.output_dir, "pose_metrics.json")
    previous_results = []
    if args.resume and os.path.isfile(metrics_path):
        with open(metrics_path) as handle:
            previous_payload = json.load(handle)
        previous_summary = previous_payload.get("summary", {})
        expected_resume_metadata = {
            "dataset": args.dataset,
            "cache_window_size": args.cache_window,
            "cache_policy": "full_cache" if args.cache_window is None else args.cache_policy,
            "cache_random_seed": args.cache_random_seed,
            "gpu_name": torch.cuda.get_device_name(device),
            "torch_version": str(torch.__version__),
            "cuda_version": torch.version.cuda or "",
            "python_version": platform.python_version(),
            "input_size": args.size,
            "stride": args.stride,
            "requested_max_frames": args.max_frames,
        }
        mismatches = {
            key: (previous_summary.get(key), expected)
            for key, expected in expected_resume_metadata.items()
            if previous_summary.get(key) != expected
        }
        if mismatches:
            details = ", ".join(
                f"{key}: existing={observed!r}, current={expected!r}"
                for key, (observed, expected) in mismatches.items()
            )
            raise RuntimeError(
                "cannot resume pose results produced with different or missing "
                f"configuration/provenance ({details}); rerun without --resume"
            )
        previous_results = previous_payload.get("sequences", [])
        if not isinstance(previous_results, list):
            raise RuntimeError(f"invalid existing sequence list in {metrics_path}")
        print(f"Loaded {len(previous_results)} existing sequence results from {metrics_path}")

    sequence_results = []
    for sequence in sequences:
        print(f"Evaluating {args.dataset}/{sequence} with {cache_name(args)}")
        frames = output = loaded = pose_encoding = pred_w2c = pred_c2w = None
        try:
            image_paths, gt_c2w = load_pose_sequence(
                args.dataset,
                data_root,
                sequence,
                anno_root=anno_root,
                stride=args.stride,
                max_frames=args.max_frames,
            )
            loaded = load_images(image_paths, size=args.size, crop=False)

            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)
            frames = [{"img": (item["img"].to(device) + 1.0) / 2.0} for item in loaded]
            torch.cuda.synchronize(device)
            start = time.perf_counter()
            with torch.no_grad(), torch.cuda.amp.autocast(dtype=dtype):
                output = model.inference(
                    frames,
                    cache_window_size=args.cache_window,
                    cache_policy=args.cache_policy,
                    cache_random_seed=args.cache_random_seed,
                    return_memory_events=args.log_selections,
                )
            torch.cuda.synchronize(device)
            inference_sec = time.perf_counter() - start

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

            # Save raw predictions before metric evaluation so a metric-only
            # failure never discards an otherwise expensive inference result.
            trajectory_path = os.path.join(pose_dir, f"{sequence}.npz")
            np.savez_compressed(
                trajectory_path,
                gt_c2w=gt_c2w,
                pred_c2w=pred_c2w,
                image_paths=np.asarray(image_paths),
            )
            pose_metrics = evaluate_trajectory(gt_c2w, pred_c2w)
            aligned_c2w = pose_metrics.pop("pred_c2w_aligned")
            result = {
                "sequence": sequence,
                "status": "ok",
                "num_frames": len(image_paths),
                "inference_sec": inference_sec,
                "fps_inference": len(image_paths) / inference_sec,
                "peak_allocated_mb": torch.cuda.max_memory_allocated(device) / (1024 ** 2),
                "peak_reserved_mb": torch.cuda.max_memory_reserved(device) / (1024 ** 2),
                **pose_metrics,
            }
            sequence_results.append(result)
            np.savez_compressed(
                trajectory_path,
                gt_c2w=gt_c2w,
                pred_c2w=pred_c2w,
                pred_c2w_aligned=aligned_c2w,
                image_paths=np.asarray(image_paths),
            )
            if args.log_selections and output.memory_events:
                with open(os.path.join(selection_dir, f"{sequence}.json"), "w") as handle:
                    json.dump(output.memory_events, handle, indent=2)
        except Exception as error:
            if isinstance(error, torch.cuda.OutOfMemoryError):
                torch.cuda.empty_cache()
            failure = {
                "sequence": sequence,
                "status": "failed",
                "error": f"{type(error).__name__}: {error}",
            }
            sequence_results.append(failure)
            print(json.dumps(failure), file=sys.stderr)
            traceback.print_exc()
        finally:
            frames = output = loaded = pose_encoding = pred_w2c = pred_c2w = None
            torch.cuda.empty_cache()

    if previous_results:
        evaluated_names = {item["sequence"] for item in sequence_results}
        sequence_results = [
            item for item in previous_results if item.get("sequence") not in evaluated_names
        ] + sequence_results
        sequence_results.sort(key=lambda item: item["sequence"])

    successful = [item for item in sequence_results if item["status"] == "ok"]
    failed = [item for item in sequence_results if item["status"] != "ok"]
    if not successful:
        raise RuntimeError(f"all {len(sequence_results)} pose sequences failed")
    summary = {
        "dataset": args.dataset,
        "cache_window_size": args.cache_window,
        "cache_policy": "full_cache" if args.cache_window is None else args.cache_policy,
        "cache_random_seed": args.cache_random_seed,
        "gpu_name": torch.cuda.get_device_name(device),
        "torch_version": str(torch.__version__),
        "cuda_version": torch.version.cuda or "",
        "python_version": platform.python_version(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID", ""),
        "hostname": platform.node(),
        "input_size": args.size,
        "stride": args.stride,
        "requested_max_frames": args.max_frames,
        "num_sequences": len(sequence_results),
        "num_successful": len(successful),
        "num_failed": len(failed),
        "total_frames": sum(item["num_frames"] for item in successful),
        "mean_ate": float(np.mean([item["ate"] for item in successful])),
        "mean_rpe_trans": float(np.mean([item["rpe_trans"] for item in successful])),
        "mean_rpe_rot_deg": float(np.mean([item["rpe_rot_deg"] for item in successful])),
        "total_inference_sec": sum(item["inference_sec"] for item in successful),
        "max_peak_allocated_mb": max(item["peak_allocated_mb"] for item in successful),
        "max_peak_reserved_mb": max(item["peak_reserved_mb"] for item in successful),
    }
    summary["fps_inference"] = summary["total_frames"] / summary["total_inference_sec"]
    with open(metrics_path, "w") as handle:
        json.dump({"summary": summary, "sequences": sequence_results}, handle, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
