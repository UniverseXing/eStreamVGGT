#!/usr/bin/env python3
import argparse
import csv
import json
from pathlib import Path


METHODS = {
    "full_cache": None,
    "old_k4": "anchor_recent_dino_diverse",
    "fixed_k4": "anchor_recent_dino_diverse_2old_1recent",
    "adaptive_k4": "anchor_stable_adaptive_recent",
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--sequence", required=True)
    parser.add_argument("--prefixes", type=int, nargs="+", required=True)
    parser.add_argument("--adaptive-threshold", type=float, required=True)
    parser.add_argument("--adaptive-min-gap", type=int, required=True)
    parser.add_argument("--threshold-tag", required=True)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def experiment_name(args, method, policy, frames):
    cache = "" if policy is None else f"_{policy}_k4"
    run_tag = (
        f"stage3_2_{method}_{args.sequence}_"
        f"tau{args.threshold_tag}_gap{args.adaptive_min_gap}"
    )
    return f"{args.dataset}_streamvggt{cache}_{run_tag}_n{frames}"


def load_row(directory, method, frames, args):
    with (directory / "runtime_memory_rank0.json").open() as handle:
        runtime = json.load(handle)["sequences"][0]
    with (directory / "result_scale.json").open() as handle:
        depth = json.load(handle)
    return {
        "method": method,
        "requested_frames": frames,
        "num_frames": runtime["num_frames"],
        "status": runtime["status"],
        "adaptive_threshold": args.adaptive_threshold,
        "adaptive_min_gap": args.adaptive_min_gap,
        "peak_allocated_mb": runtime["peak_allocated_mb"],
        "peak_reserved_mb": runtime["peak_reserved_mb"],
        "inference_sec": runtime["inference_sec"],
        "fps_inference": runtime["fps_inference"],
        "abs_rel": depth["Abs Rel"],
        "rmse": depth["RMSE"],
        "delta_1": depth["δ < 1.25"],
        "delta_2": depth["δ < 1.25^2"],
        "delta_3": depth["δ < 1.25^3"],
    }


def main():
    args = parse_args()
    rows = []
    for frames in args.prefixes:
        for method, policy in METHODS.items():
            directory = args.results_root / experiment_name(
                args, method, policy, frames
            )
            rows.append(load_row(directory, method, frames, args))

    output = args.output or (
        args.results_root / f"stage3_2_scaling_{args.dataset}_{args.sequence}.csv"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(output)


if __name__ == "__main__":
    main()
