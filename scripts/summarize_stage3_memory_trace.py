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
    parser.add_argument("--frames", type=int, required=True)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def experiment_name(dataset, sequence, method, policy, window, frames):
    cache = "" if window is None else f"_{policy}_k{window}"
    return f"{dataset}_streamvggt{cache}_memory_{method}_{sequence}_n{frames}"


def main():
    args = parse_args()
    rows = []
    for method, (policy, window) in METHODS.items():
        experiment = experiment_name(
            args.dataset, args.sequence, method, policy, window, args.frames
        )
        trace_path = args.results_root / experiment / "memory_traces" / f"{args.sequence}.json"
        with trace_path.open() as handle:
            trace = json.load(handle)
        for item in trace:
            rows.append({"method": method, **item})

    output = args.output or (
        args.results_root / f"stage3_memory_trace_{args.dataset}_{args.sequence}.csv"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(output)


if __name__ == "__main__":
    main()
