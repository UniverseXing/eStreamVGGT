#!/usr/bin/env python3
"""Summarize Stage 4C frozen raw-TUM long-sequence runs."""

import argparse
import csv
import glob
import json
import os


FIELDS = (
    "run_scope",
    "run_id",
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
    parser.add_argument(
        "--include-cell",
        action="append",
        default=[],
        metavar="METHOD|SEQUENCE|FRAMES",
        help="include exactly one planned cell; may be repeated",
    )
    parser.add_argument(
        "--expected-run-id",
        action="append",
        default=[],
        metavar="METHOD|SEQUENCE|FRAMES|RUN_ID",
        help="require the caller-generated run identifier for a planned cell",
    )
    parser.add_argument("--require-consistent-provenance", action="store_true")
    parser.add_argument("--require-pose-success", action="store_true")
    parser.add_argument(
        "--expected-run-scope", choices=("frozen", "debug_subset")
    )
    args = parser.parse_args()

    included = set()
    for value in args.include_cell:
        fields = value.split("|")
        if len(fields) != 3:
            raise ValueError(f"invalid --include-cell value: {value!r}")
        key = (fields[0], fields[1], int(fields[2]))
        if key in included:
            raise ValueError(f"duplicate --include-cell value: {value!r}")
        included.add(key)

    expected_run_ids = {}
    for value in args.expected_run_id:
        fields = value.split("|", 3)
        if len(fields) != 4 or not fields[3]:
            raise ValueError(f"invalid --expected-run-id value: {value!r}")
        key = (fields[0], fields[1], int(fields[2]))
        if key in expected_run_ids:
            raise ValueError(f"duplicate --expected-run-id cell: {key}")
        expected_run_ids[key] = fields[3]
    if expected_run_ids and set(expected_run_ids) != included:
        raise ValueError(
            "--expected-run-id cells must exactly match --include-cell cells"
        )

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
        if included and key not in included:
            continue
        expected_run_id = expected_run_ids.get(key)
        if expected_run_id is not None and payload.get("run_id") != expected_run_id:
            raise RuntimeError(
                f"stale Stage 4C result for {key}: expected run_id "
                f"{expected_run_id!r}, found {payload.get('run_id')!r}"
            )
        if key in seen:
            raise ValueError(f"duplicate Stage 4C result: {key}")
        seen.add(key)
        rows.append(flatten(payload, os.path.dirname(path)))
    if included and seen != included:
        raise RuntimeError(
            "missing planned Stage 4C result(s): "
            f"{sorted(included - seen)}"
        )
    if args.expected_run_scope is not None:
        mismatched_scope = [
            (row.get("method"), row.get("sequence"), row.get("num_frames"), row.get("run_scope"))
            for row in rows
            if row.get("run_scope") != args.expected_run_scope
        ]
        if mismatched_scope:
            raise RuntimeError(
                f"Stage 4C run_scope mismatch; expected {args.expected_run_scope}: "
                f"{mismatched_scope}"
            )
    if args.require_consistent_provenance:
        for field in ("gpu_name", "torch_version", "cuda_version", "python_version"):
            values = {row.get(field) for row in rows}
            if len(values) != 1 or next(iter(values), None) in (None, ""):
                raise RuntimeError(
                    f"inconsistent Stage 4C {field}: {sorted(values, key=str)}"
                )
    if args.require_pose_success:
        failed_pose = [
            (row.get("method"), row.get("sequence"), row.get("num_frames"))
            for row in rows
            if row.get("status") == "ok" and row.get("pose_status") != "ok"
        ]
        if failed_pose:
            raise RuntimeError(
                f"successful Stage 4C inference with failed pose metric: {failed_pose}"
            )
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
