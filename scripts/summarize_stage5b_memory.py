#!/usr/bin/env python3
"""Validate and decompose the Stage 5B 2x2 memory experiment."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


CELLS = {
    "full_accumulated": ("stream_accumulate", None, "full_cache"),
    "full_release": ("stream_release", None, "full_cache"),
    "k4_accumulated": ("stream_accumulate", 4, "anchor_recent_dino_diverse_k4"),
    "k4_release": ("stream_release", 4, "anchor_recent_dino_diverse_k4"),
}
RESULT_FIELDS = (
    "cell", "mode", "status", "dataset", "sequence", "num_frames",
    "processed_frames", "cache_window_size", "cache_policy", "input_mode",
    "output_mode", "inference_sec", "wall_sec", "fps_inference",
    "fps_end_to_end", "peak_allocated_mb", "peak_reserved_mb",
    "rss_before_mib", "rss_peak_mib", "rss_growth_mib",
    "max_input_tensors_mib", "max_retained_outputs_mib",
    "max_retained_views_mib", "max_aggregator_kv_mib", "max_camera_kv_mib",
    "max_descriptor_mib", "max_trace_allocated_mib", "camera_pose_sha256",
    "depth_sha256", "ate", "rpe_trans", "rpe_rot_deg", "gpu_name",
    "torch_version", "cuda_version", "python_version", "slurm_job_id",
    "hostname", "source",
)
TRACE_FIELDS = (
    "cell", "frame_index", "retained_frame_ids", "aggregator_kv_mib",
    "camera_kv_mib", "descriptor_mib", "input_tensors_mib",
    "retained_outputs_mib", "retained_views_mib", "cuda_allocated_mib",
    "cuda_reserved_mib",
)
CONTRIBUTION_FIELDS = (
    "effect", "high_memory_cell", "low_memory_cell",
    "peak_allocated_saved_mib", "peak_allocated_saved_percent",
    "peak_reserved_saved_mib", "final_trace_allocated_saved_mib",
)


def read_json(path):
    with path.open() as handle:
        return json.load(handle)


def write_csv(path, fields, rows):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)
    print(f"Wrote {len(rows)} rows to {path}")


def normalized_window(value):
    return None if value in (None, "") else int(value)


def contribution(effect, high_name, low_name, indexed, traces):
    high, low = indexed[high_name], indexed[low_name]
    high_final = float(traces[high_name][-1]["cuda_allocated_mib"])
    low_final = float(traces[low_name][-1]["cuda_allocated_mib"])
    peak_saved = float(high["peak_allocated_mb"]) - float(low["peak_allocated_mb"])
    return {
        "effect": effect, "high_memory_cell": high_name, "low_memory_cell": low_name,
        "peak_allocated_saved_mib": peak_saved,
        "peak_allocated_saved_percent": 100.0 * peak_saved / float(high["peak_allocated_mb"]),
        "peak_reserved_saved_mib": float(high["peak_reserved_mb"]) - float(low["peak_reserved_mb"]),
        "final_trace_allocated_saved_mib": high_final - low_final,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--results-root", type=Path)
    args = parser.parse_args()
    root = args.repo_root.resolve()
    results_root = args.results_root or root / "eval_results/stage5b_memory"
    rows, traces = [], {}
    reference = None
    for cell, (mode, window, policy) in CELLS.items():
        result_path = results_root / cell / "stage5b_metrics.json"
        trace_path = results_root / cell / "memory_trace.json"
        if not result_path.is_file() or not trace_path.is_file():
            raise FileNotFoundError(f"missing Stage 5B output for {cell}")
        payload = read_json(result_path)
        observed = (payload.get("mode"), normalized_window(payload.get("cache_window_size")), payload.get("cache_policy"))
        if observed != (mode, window, policy):
            raise RuntimeError(f"Stage 5B config mismatch for {cell}: {observed}")
        if payload.get("status") != "ok" or payload.get("processed_frames") != payload.get("num_frames"):
            raise RuntimeError(f"Stage 5B cell did not complete: {cell}")
        if payload.get("input_mode") != "streaming":
            raise RuntimeError(f"Stage 5B input lifecycle is not matched for {cell}")
        expected_output = "retained" if mode == "stream_accumulate" else "sink"
        if payload.get("output_mode") != expected_output:
            raise RuntimeError(f"Stage 5B output lifecycle mismatch for {cell}")
        identity = (payload["dataset"], payload["sequence"], int(payload["num_frames"]), payload.get("gpu_name"), payload.get("torch_version"), payload.get("cuda_version"))
        if reference is None:
            reference = identity
        elif identity != reference:
            raise RuntimeError(f"Stage 5B protocol/provenance mismatch for {cell}")
        row = {field: payload.get(field, "") for field in RESULT_FIELDS}
        row["cell"] = cell; row["source"] = str(result_path.parent)
        rows.append(row)
        trace = read_json(trace_path)
        if len(trace) != int(payload["num_frames"]):
            raise RuntimeError(f"incomplete memory trace for {cell}")
        traces[cell] = trace
    if reference[:3] != ("bonn", "person_tracking2", 110):
        raise RuntimeError(f"Stage 5B formal sequence must be Bonn person_tracking2/110: {reference[:3]}")
    if "6000 ada" not in str(reference[3]).lower():
        raise RuntimeError(f"Stage 5B formal GPU must be RTX 6000 Ada: {reference[3]}")
    indexed = {row["cell"]: row for row in rows}
    for cache in ("full", "k4"):
        accumulated = indexed[f"{cache}_accumulated"]
        released = indexed[f"{cache}_release"]
        for signature in ("camera_pose_sha256", "depth_sha256"):
            if accumulated[signature] != released[signature]:
                raise RuntimeError(f"prediction mismatch for {cache}: {signature}")
    trace_rows = []
    for cell, trace in traces.items():
        for item in trace:
            row = {field: item.get(field, "") for field in TRACE_FIELDS}
            row["cell"] = cell
            if isinstance(row["retained_frame_ids"], list):
                row["retained_frame_ids"] = json.dumps(row["retained_frame_ids"])
            trace_rows.append(row)
    contributions = [
        contribution("kv_pruning_with_accumulated_outputs", "full_accumulated", "k4_accumulated", indexed, traces),
        contribution("kv_pruning_with_streaming_release", "full_release", "k4_release", indexed, traces),
        contribution("output_release_with_full_cache", "full_accumulated", "full_release", indexed, traces),
        contribution("output_release_with_k4", "k4_accumulated", "k4_release", indexed, traces),
    ]
    write_csv(root / "stage5b_memory_decomposition.csv", RESULT_FIELDS, rows)
    write_csv(root / "stage5b_memory_trace.csv", TRACE_FIELDS, trace_rows)
    write_csv(root / "stage5b_memory_contributions.csv", CONTRIBUTION_FIELDS, contributions)


if __name__ == "__main__":
    main()
