#!/usr/bin/env python3
import argparse
import csv
import glob
import json
import os
from collections import defaultdict


FIELDS = (
    "dataset",
    "method",
    "protocol",
    "sampling_stride",
    "prefix_frames",
    "cache_policy",
    "cache_window_size",
    "num_sequences",
    "num_successful",
    "num_failed",
    "total_frames",
    "mean_acc",
    "mean_acc_med",
    "mean_comp",
    "mean_comp_med",
    "mean_nc",
    "mean_nc_med",
    "mean_overall",
    "mean_ate",
    "mean_rpe_trans",
    "mean_rpe_rot_deg",
    "total_inference_sec",
    "fps_inference",
    "mean_final_frame_ms",
    "max_peak_allocated_mb",
    "max_peak_reserved_mb",
    "result_dir",
)

QUALITY_FIELDS = (
    "mean_acc",
    "mean_acc_med",
    "mean_comp",
    "mean_comp_med",
    "mean_nc",
    "mean_nc_med",
    "mean_overall",
    "mean_ate",
    "mean_rpe_trans",
    "mean_rpe_rot_deg",
    "mean_final_frame_ms",
)


def mean(values):
    values = [value for value in values if value is not None]
    return sum(values) / len(values) if values else None


def method_name(result_dir, name_filter):
    name = os.path.basename(result_dir)
    prefix = f"streamvggt_{name_filter}_"
    return name[len(prefix) :] if name.startswith(prefix) else name


def macro_row(method, rows):
    total_frames = sum(row.get("total_frames") or 0 for row in rows)
    total_time = sum(row.get("total_inference_sec") or 0.0 for row in rows)
    row = {
        "dataset": "macro_average",
        "method": method,
        "protocol": rows[0]["protocol"],
        "sampling_stride": None,
        "prefix_frames": None,
        "cache_policy": rows[0]["cache_policy"],
        "cache_window_size": rows[0]["cache_window_size"],
        "num_sequences": sum(row.get("num_sequences") or 0 for row in rows),
        "num_successful": sum(row.get("num_successful") or 0 for row in rows),
        "num_failed": sum(row.get("num_failed") or 0 for row in rows),
        "total_frames": total_frames,
        "total_inference_sec": total_time,
        "fps_inference": total_frames / total_time if total_time else None,
        "max_peak_allocated_mb": max(
            (row["max_peak_allocated_mb"] for row in rows if row.get("max_peak_allocated_mb") is not None),
            default=None,
        ),
        "max_peak_reserved_mb": max(
            (row["max_peak_reserved_mb"] for row in rows if row.get("max_peak_reserved_mb") is not None),
            default=None,
        ),
        "result_dir": rows[0]["result_dir"],
    }
    for field in QUALITY_FIELDS:
        row[field] = mean([dataset_row.get(field) for dataset_row in rows])
    return row


def main():
    parser = argparse.ArgumentParser("Summarize Stage 3.3B reconstruction runs")
    parser.add_argument("--results-root", default="eval_results/mv_recon")
    parser.add_argument("--name-filter", default="stage3_3b")
    parser.add_argument("--output", default="stage3_3b_recon_results.csv")
    args = parser.parse_args()

    pattern = os.path.join(
        args.results_root, f"*{args.name_filter}*", "reconstruction_metrics.json"
    )
    rows = []
    by_method = defaultdict(list)
    for path in sorted(glob.glob(pattern)):
        result_dir = os.path.dirname(path)
        method = method_name(result_dir, args.name_filter)
        with open(path) as handle:
            payload = json.load(handle)
        for dataset_payload in payload["datasets"].values():
            summary = dataset_payload["summary"]
            row = {field: summary.get(field) for field in FIELDS}
            row["method"] = method
            row["result_dir"] = result_dir
            rows.append(row)
            by_method[method].append(row)
            for prefix_summary in dataset_payload.get("prefix_summaries", []):
                prefix_row = {field: prefix_summary.get(field) for field in FIELDS}
                prefix_row["method"] = method
                prefix_row["sampling_stride"] = summary.get("sampling_stride")
                prefix_row["num_successful"] = prefix_summary.get("num_sequences")
                prefix_row["num_failed"] = 0
                prefix_row["result_dir"] = result_dir
                rows.append(prefix_row)
    if not rows:
        raise RuntimeError(f"no reconstruction result files matched {pattern}")

    for method, method_rows in by_method.items():
        rows.append(macro_row(method, method_rows))

    with open(args.output, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} rows to {args.output}")


if __name__ == "__main__":
    main()
