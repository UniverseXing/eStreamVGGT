#!/usr/bin/env python3
import argparse
import csv
import glob
import json
import os


FIELDS = (
    "dataset",
    "cache_policy",
    "cache_window_size",
    "num_sequences",
    "num_successful",
    "num_failed",
    "total_frames",
    "mean_ate",
    "mean_rpe_trans",
    "mean_rpe_rot_deg",
    "total_inference_sec",
    "fps_inference",
    "max_peak_allocated_mb",
    "max_peak_reserved_mb",
    "result_dir",
)


def main():
    parser = argparse.ArgumentParser("Summarize Stage 3.3 pose runs")
    parser.add_argument("--results-root", default="eval_results/pose")
    parser.add_argument("--name-filter", default="stage3_3")
    parser.add_argument("--output", default="stage3_3_pose_results.csv")
    args = parser.parse_args()

    pattern = os.path.join(args.results_root, f"*{args.name_filter}*", "pose_metrics.json")
    rows = []
    for path in sorted(glob.glob(pattern)):
        with open(path) as handle:
            summary = json.load(handle)["summary"]
        row = {field: summary.get(field) for field in FIELDS}
        row["result_dir"] = os.path.dirname(path)
        rows.append(row)
    if not rows:
        raise RuntimeError(f"no pose result files matched {pattern}")

    with open(args.output, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} runs to {args.output}")


if __name__ == "__main__":
    main()
