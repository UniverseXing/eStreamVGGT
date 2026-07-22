#!/usr/bin/env python3
"""Summarize Stage 3.6A bounded-window pose experiments."""

import argparse
import csv
import glob
import json
import os


FIELDS = (
    "method",
    "prefix_frames",
    "mode",
    "status",
    "error",
    "sequence",
    "num_frames",
    "processed_frames",
    "recompute_factor",
    "window_size",
    "overlap",
    "cache_window_size",
    "cache_policy",
    "ate",
    "rpe_trans",
    "rpe_rot_deg",
    "model_inference_sec",
    "wall_sec",
    "fps_unique_inference",
    "fps_processed_inference",
    "peak_allocated_mb",
    "peak_reserved_mb",
    "num_alignment_events",
    "num_alignment_fallbacks",
    "max_overlap_translation_rmse",
    "max_overlap_rotation_deg",
    "result_dir",
)


def maximum(values):
    values = [float(value) for value in values if value is not None]
    return max(values) if values else None


def flatten(payload, result_dir, prefix=None):
    alignments = [
        row for row in payload.get("alignment_events", []) if row.get("overlap", 0) > 0
    ]
    metric = payload if prefix is None else prefix
    return {
        "method": payload.get("method"),
        "prefix_frames": None if prefix is None else prefix.get("prefix_frames"),
        "mode": payload.get("mode"),
        "status": payload.get("status") if prefix is None else prefix.get("status"),
        "error": payload.get("error") if prefix is None else prefix.get("error"),
        "sequence": payload.get("sequence"),
        "num_frames": payload.get("num_frames"),
        "processed_frames": payload.get("processed_frames"),
        "recompute_factor": payload.get("recompute_factor"),
        "window_size": payload.get("window_size"),
        "overlap": payload.get("overlap"),
        "cache_window_size": payload.get("cache_window_size"),
        "cache_policy": payload.get("cache_policy"),
        "ate": metric.get("ate"),
        "rpe_trans": metric.get("rpe_trans"),
        "rpe_rot_deg": metric.get("rpe_rot_deg"),
        "model_inference_sec": payload.get("model_inference_sec") if prefix is None else None,
        "wall_sec": payload.get("wall_sec") if prefix is None else None,
        "fps_unique_inference": payload.get("fps_unique_inference") if prefix is None else None,
        "fps_processed_inference": payload.get("fps_processed_inference") if prefix is None else None,
        "peak_allocated_mb": payload.get("peak_allocated_mb") if prefix is None else None,
        "peak_reserved_mb": payload.get("peak_reserved_mb") if prefix is None else None,
        "num_alignment_events": len(alignments) if prefix is None else None,
        "num_alignment_fallbacks": sum(
            row.get("method") == "orientation_fallback" for row in alignments
        ) if prefix is None else None,
        "max_overlap_translation_rmse": maximum(
            [row.get("overlap_translation_rmse") for row in alignments]
        ) if prefix is None else None,
        "max_overlap_rotation_deg": maximum(
            [row.get("overlap_rotation_deg_max") for row in alignments]
        ) if prefix is None else None,
        "result_dir": result_dir,
    }


def main():
    parser = argparse.ArgumentParser("Summarize Stage 3.6A results")
    parser.add_argument("--results-root", default="eval_results/stage3_6a")
    parser.add_argument("--output", default="stage3_6a_results.csv")
    args = parser.parse_args()

    paths = sorted(
        glob.glob(os.path.join(args.results_root, "*", "stage3_6a_metrics.json"))
    )
    if not paths:
        raise RuntimeError(f"no Stage 3.6A metrics below {args.results_root}")
    rows = []
    for path in paths:
        with open(path) as handle:
            payload = json.load(handle)
        result_dir = os.path.dirname(path)
        rows.append(flatten(payload, result_dir))
        if payload.get("status") == "ok":
            rows.extend(
                flatten(payload, result_dir, prefix)
                for prefix in payload.get("prefix_metrics", [])
            )

    with open(args.output, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} rows to {args.output}")


if __name__ == "__main__":
    main()
