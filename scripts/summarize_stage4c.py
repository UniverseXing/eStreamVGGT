#!/usr/bin/env python3
"""Summarize Stage 4C frozen raw-TUM long-sequence runs."""

import argparse
import csv
import glob
import json
import os


FIELDS = (
    "method",
    "status",
    "error",
    "dataset",
    "sequence",
    "num_frames",
    "processed_frames",
    "cache_window_size",
    "cache_policy",
    "mode",
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
    "max_input_tensors_mib",
    "max_retained_outputs_mib",
    "max_retained_views_mib",
    "max_aggregator_kv_mib",
    "max_camera_kv_mib",
    "max_descriptor_mib",
    "camera_pose_sha256",
    "pose_status",
    "pose_error",
    "ate",
    "rpe_trans",
    "rpe_rot_deg",
    "align_scale",
    "max_association_difference_sec",
    "near_occupancy_rate",
    "middle_occupancy_rate",
    "long_occupancy_rate",
    "final_max_temporal_gap",
    "final_bank_frame_ids",
    "gpu_name",
    "torch_version",
    "cuda_version",
    "python_version",
    "slurm_job_id",
    "hostname",
    "result_dir",
)


def flatten(payload, result_dir):
    row = {key: payload.get(key) for key in FIELDS}
    bank = payload.get("temporal_bank_statistics") or {}
    for key in (
        "near_occupancy_rate",
        "middle_occupancy_rate",
        "long_occupancy_rate",
        "final_max_temporal_gap",
    ):
        row[key] = bank.get(key)
    if bank.get("final_bank_frame_ids") is not None:
        row["final_bank_frame_ids"] = json.dumps(
            bank["final_bank_frame_ids"], sort_keys=True
        )
    row["result_dir"] = result_dir
    return row


def main():
    parser = argparse.ArgumentParser("Summarize Stage 4C results")
    parser.add_argument(
        "--results-root", default="eval_results/stage4c_tum_long"
    )
    parser.add_argument("--output", default="stage4c_results.csv")
    args = parser.parse_args()

    paths = sorted(
        glob.glob(
            os.path.join(
                args.results_root, "*", "*", "*", "stage4c_metrics.json"
            )
        )
    )
    if not paths:
        raise RuntimeError(f"no Stage 4C metrics below {args.results_root}")
    rows = []
    seen = set()
    for path in paths:
        with open(path) as handle:
            payload = json.load(handle)
        key = (
            payload.get("method"),
            payload.get("sequence"),
            int(payload.get("num_frames", 0)),
        )
        if key in seen:
            raise ValueError(f"duplicate Stage 4C result: {key}")
        seen.add(key)
        rows.append(flatten(payload, os.path.dirname(path)))
    rows.sort(
        key=lambda row: (
            row["method"],
            row["sequence"],
            int(row["num_frames"]),
        )
    )

    with open(args.output, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} Stage 4C rows to {args.output}")


if __name__ == "__main__":
    main()
