#!/usr/bin/env python3
"""Summarize Stage 3.6B streaming-memory runs."""

import argparse
import csv
import glob
import json
import os


FIELDS = (
    "method",
    "mode",
    "status",
    "error",
    "dataset",
    "sequence",
    "num_frames",
    "cache_window_size",
    "cache_policy",
    "collect_depth",
    "inference_sec",
    "wall_sec",
    "fps_inference",
    "fps_end_to_end",
    "peak_allocated_mb",
    "peak_reserved_mb",
    "rss_before_mib",
    "rss_peak_mib",
    "rss_growth_mib",
    "process_max_rss_mib",
    "input_mode",
    "output_mode",
    "final_input_tensors_mib",
    "max_input_tensors_mib",
    "final_retained_outputs_mib",
    "max_retained_outputs_mib",
    "final_retained_views_mib",
    "max_retained_views_mib",
    "final_aggregator_kv_mib",
    "max_aggregator_kv_mib",
    "final_camera_kv_mib",
    "max_camera_kv_mib",
    "final_descriptor_mib",
    "max_descriptor_mib",
    "final_trace_allocated_mib",
    "max_trace_allocated_mib",
    "camera_pose_sha256",
    "depth_sha256",
    "pose_status",
    "pose_error",
    "ate",
    "rpe_trans",
    "rpe_rot_deg",
    "abs_rel",
    "sq_rel",
    "rmse",
    "log_rmse",
    "delta_1",
    "delta_2",
    "delta_3",
    "valid_pixels",
    "near_occupancy_rate",
    "middle_occupancy_rate",
    "long_occupancy_rate",
    "final_max_temporal_gap",
    "final_bank_frame_ids",
    "result_dir",
)


def flatten(payload, result_dir):
    row = {key: payload.get(key) for key in FIELDS}
    banks = payload.get("temporal_bank_statistics") or {}
    for key in (
        "near_occupancy_rate",
        "middle_occupancy_rate",
        "long_occupancy_rate",
        "final_max_temporal_gap",
    ):
        row[key] = banks.get(key)
    if banks.get("final_bank_frame_ids") is not None:
        row["final_bank_frame_ids"] = json.dumps(
            banks["final_bank_frame_ids"], sort_keys=True
        )
    row["result_dir"] = result_dir
    return row


def main():
    parser = argparse.ArgumentParser("Summarize Stage 3.6B results")
    parser.add_argument("--results-root", default="eval_results/stage3_6b")
    parser.add_argument("--output", default="stage3_6b_results.csv")
    args = parser.parse_args()

    paths = sorted(
        glob.glob(os.path.join(args.results_root, "*", "stage3_6b_metrics.json"))
    )
    if not paths:
        raise RuntimeError(f"no Stage 3.6B metrics below {args.results_root}")
    rows = []
    for path in paths:
        with open(path) as handle:
            rows.append(flatten(json.load(handle), os.path.dirname(path)))

    with open(args.output, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} Stage 3.6B rows to {args.output}")


if __name__ == "__main__":
    main()
