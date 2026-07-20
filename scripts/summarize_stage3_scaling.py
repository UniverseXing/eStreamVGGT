#!/usr/bin/env python3
import argparse
import csv
import json
from pathlib import Path


METHODS = {
    "full_cache": (None, None),
    "old_k4": ("anchor_recent_dino_diverse", 4),
    "old_k6": ("anchor_recent_dino_diverse", 6),
    "new_k4": ("anchor_recent_dino_diverse_2old_1recent", 4),
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--sequence", required=True)
    parser.add_argument("--prefixes", type=int, nargs="+", required=True)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def experiment_name(dataset, sequence, method, policy, window, frames):
    cache = "" if window is None else f"_{policy}_k{window}"
    return f"{dataset}_streamvggt{cache}_scaling_{method}_{sequence}_n{frames}"


def load_row(directory, method, requested_frames):
    with (directory / "runtime_memory_rank0.json").open() as handle:
        runtime = json.load(handle)
    sequences = runtime.get("sequences", [])
    if len(sequences) != 1:
        raise ValueError(f"expected one sequence in {directory}, found {len(sequences)}")
    stats = sequences[0]

    depth_path = directory / "result_scale.json"
    if depth_path.exists():
        with depth_path.open() as handle:
            depth = json.load(handle)
    else:
        depth = {}
    return {
        "method": method,
        "requested_frames": requested_frames,
        "num_frames": stats["num_frames"],
        "status": stats["status"],
        "peak_allocated_mb": stats["peak_allocated_mb"],
        "peak_reserved_mb": stats["peak_reserved_mb"],
        "inference_sec": stats["inference_sec"],
        "fps_inference": stats["fps_inference"],
        "abs_rel": depth.get("Abs Rel"),
        "rmse": depth.get("RMSE"),
        "delta_1": depth.get("δ < 1.25"),
        "delta_2": depth.get("δ < 1.25^2"),
        "delta_3": depth.get("δ < 1.25^3"),
    }


def main():
    args = parse_args()
    rows = []
    for frames in args.prefixes:
        for method, (policy, window) in METHODS.items():
            experiment = experiment_name(
                args.dataset, args.sequence, method, policy, window, frames
            )
            rows.append(
                load_row(args.results_root / experiment, method, frames)
            )

    output = args.output or (
        args.results_root / f"stage3_scaling_{args.dataset}_{args.sequence}.csv"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(output)


if __name__ == "__main__":
    main()
