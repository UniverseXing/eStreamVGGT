#!/usr/bin/env python3
"""Combine Stage 3.4 Bonn and 7-Scenes loop results into compact CSV files."""

import argparse
import csv
import glob
import json
import os
from collections import defaultdict


FIELDS = (
    "dataset",
    "method",
    "prefix_frames",
    "cache_policy",
    "cache_window_size",
    "camera_cache_policy",
    "camera_cache_window_size",
    "num_sequences",
    "num_successful",
    "num_failed",
    "total_frames",
    "mean_abs_rel",
    "mean_rmse",
    "mean_delta_1",
    "mean_ate",
    "mean_rpe_trans",
    "mean_rpe_rot_deg",
    "mean_loop_translation",
    "mean_loop_rotation_deg",
    "mean_loop_depth_abs_rel",
    "total_inference_sec",
    "fps_inference",
    "mean_frame_latency_ms",
    "mean_last_frame_latency_ms",
    "mean_aggregator_kv_mib",
    "mean_camera_kv_mib",
    "mean_descriptor_mib",
    "mean_retained_outputs_mib",
    "max_cuda_allocated_mib",
    "max_cuda_reserved_mib",
    "max_peak_allocated_mb",
    "max_peak_reserved_mb",
    "mean_retained_age",
    "max_retained_age",
    "mean_temporal_span",
    "anchor0_retention_rate",
    "mean_selection_churn",
    "unique_retained_frames",
    "loop_match_retention_rate",
    "mean_near_bank_occupancy_rate",
    "mean_middle_bank_occupancy_rate",
    "mean_long_bank_occupancy_rate",
    "mean_near_bank_updates",
    "mean_middle_bank_updates",
    "mean_long_bank_updates",
    "max_final_temporal_gap",
    "result_dir",
)

SEQUENCE_FIELDS = (
    "dataset",
    "method",
    "cache_policy",
    "cache_window_size",
    "camera_cache_policy",
    "camera_cache_window_size",
    "sequence",
    "status",
    "error",
    "num_frames",
    "abs_rel",
    "rmse",
    "delta_1",
    "ate",
    "rpe_trans",
    "rpe_rot_deg",
    "loop_translation_mean",
    "loop_rotation_deg_mean",
    "loop_depth_abs_rel",
    "inference_sec",
    "fps_inference",
    "mean_frame_latency_ms",
    "last_frame_latency_ms",
    "peak_allocated_mb",
    "peak_reserved_mb",
    "mean_retained_age",
    "max_retained_age",
    "mean_temporal_span",
    "anchor0_retention_rate",
    "mean_selection_churn",
    "unique_retained_frames",
    "loop_match_retention_rate",
    "final_retained_frame_ids",
    "mean_camera_retained_age",
    "max_camera_retained_age",
    "mean_camera_temporal_span",
    "camera_anchor0_retention_rate",
    "camera_unique_retained_frames",
    "final_camera_retained_frame_ids",
    "near_bank_occupancy_rate",
    "middle_bank_occupancy_rate",
    "long_bank_occupancy_rate",
    "near_bank_updates",
    "middle_bank_updates",
    "long_bank_updates",
    "near_bank_unique_frames",
    "middle_bank_unique_frames",
    "long_bank_unique_frames",
    "final_max_temporal_gap",
    "final_temporal_bank_frame_ids",
    "result_dir",
)


def mean(values):
    values = [float(value) for value in values if value is not None]
    return sum(values) / len(values) if values else None


def maximum(values):
    values = [float(value) for value in values if value is not None]
    return max(values) if values else None


def weighted_mean(rows, key, weight="valid_pixels"):
    pairs = [
        (float(row[key]), float(row.get(weight) or 0.0))
        for row in rows
        if row.get(key) is not None
    ]
    total_weight = sum(item[1] for item in pairs)
    if total_weight:
        return sum(value * item_weight for value, item_weight in pairs) / total_weight
    return mean([row.get(key) for row in rows])


def method_name(result_dir, dataset):
    name = os.path.basename(result_dir)
    prefix = f"{dataset}_"
    return name[len(prefix) :] if name.startswith(prefix) else name


def metric_row(dataset, method, payload, rows, prefix, result_dir):
    selection_rows = [row.get("selection_statistics", {}) for row in rows]
    temporal_rows = [row.get("temporal_bank_statistics", {}) for row in rows]
    total_frames = sum(row.get("num_frames", 0) for row in rows) if prefix is None else prefix * len(rows)
    total_inference = sum(row.get("inference_sec", 0.0) for row in rows) if prefix is None else None
    result = {
        "dataset": dataset,
        "method": method,
        "prefix_frames": prefix,
        "cache_policy": payload["cache_policy"],
        "cache_window_size": payload.get("cache_window_size"),
        "camera_cache_policy": payload.get("camera_cache_policy", "coupled"),
        "camera_cache_window_size": payload.get(
            "camera_cache_window_size", payload.get("cache_window_size")
        ),
        "num_sequences": len(payload["sequences"]),
        "num_successful": len(rows),
        "num_failed": len(payload["sequences"]) - len(rows),
        "total_frames": total_frames,
        "mean_abs_rel": weighted_mean(rows, "abs_rel"),
        "mean_rmse": weighted_mean(rows, "rmse"),
        "mean_delta_1": weighted_mean(rows, "delta_1"),
        "mean_ate": mean([row.get("ate") for row in rows]),
        "mean_rpe_trans": mean([row.get("rpe_trans") for row in rows]),
        "mean_rpe_rot_deg": mean([row.get("rpe_rot_deg") for row in rows]),
        "mean_loop_translation": mean([row.get("loop_translation_mean") for row in rows]),
        "mean_loop_rotation_deg": mean([row.get("loop_rotation_deg_mean") for row in rows]),
        "mean_loop_depth_abs_rel": mean([row.get("loop_depth_abs_rel") for row in rows]),
        "total_inference_sec": total_inference,
        "fps_inference": total_frames / total_inference if total_inference else None,
        "mean_frame_latency_ms": mean([row.get("mean_frame_latency_ms") for row in rows]),
        "mean_last_frame_latency_ms": mean(
            [row.get("last_frame_latency_ms") for row in rows]
        ),
        "mean_aggregator_kv_mib": mean([row.get("aggregator_kv_mib") for row in rows]),
        "mean_camera_kv_mib": mean([row.get("camera_kv_mib") for row in rows]),
        "mean_descriptor_mib": mean([row.get("descriptor_mib") for row in rows]),
        "mean_retained_outputs_mib": mean([row.get("retained_outputs_mib") for row in rows]),
        "max_cuda_allocated_mib": maximum([row.get("cuda_allocated_mib") for row in rows]),
        "max_cuda_reserved_mib": maximum([row.get("cuda_reserved_mib") for row in rows]),
        "max_peak_allocated_mb": maximum([row.get("peak_allocated_mb") for row in rows]) if prefix is None else None,
        "max_peak_reserved_mb": maximum([row.get("peak_reserved_mb") for row in rows]) if prefix is None else None,
        "mean_retained_age": mean([row.get("mean_retained_age") for row in selection_rows]) if prefix is None else None,
        "max_retained_age": maximum([row.get("max_retained_age") for row in selection_rows]) if prefix is None else None,
        "mean_temporal_span": mean([row.get("mean_temporal_span") for row in selection_rows]) if prefix is None else None,
        "anchor0_retention_rate": mean([row.get("anchor0_retention_rate") for row in selection_rows]) if prefix is None else None,
        "mean_selection_churn": mean([row.get("mean_selection_churn") for row in selection_rows]) if prefix is None else None,
        "unique_retained_frames": mean([row.get("unique_retained_frames") for row in selection_rows]) if prefix is None else None,
        "loop_match_retention_rate": mean([row.get("loop_match_retention_rate") for row in selection_rows]) if prefix is None else None,
        "mean_near_bank_occupancy_rate": mean([row.get("near_occupancy_rate") for row in temporal_rows]) if prefix is None else None,
        "mean_middle_bank_occupancy_rate": mean([row.get("middle_occupancy_rate") for row in temporal_rows]) if prefix is None else None,
        "mean_long_bank_occupancy_rate": mean([row.get("long_occupancy_rate") for row in temporal_rows]) if prefix is None else None,
        "mean_near_bank_updates": mean([row.get("near_updates") for row in temporal_rows]) if prefix is None else None,
        "mean_middle_bank_updates": mean([row.get("middle_updates") for row in temporal_rows]) if prefix is None else None,
        "mean_long_bank_updates": mean([row.get("long_updates") for row in temporal_rows]) if prefix is None else None,
        "max_final_temporal_gap": maximum([row.get("final_max_temporal_gap") for row in temporal_rows]) if prefix is None else None,
        "result_dir": result_dir,
    }
    return result


def flatten_sequence(dataset, method, payload, row, result_dir):
    selection = row.get("selection_statistics", {})
    camera_selection = row.get("camera_selection_statistics", {})
    temporal_banks = row.get("temporal_bank_statistics", {})
    return {
        "dataset": dataset,
        "method": method,
        "cache_policy": payload["cache_policy"],
        "cache_window_size": payload.get("cache_window_size"),
        "camera_cache_policy": payload.get("camera_cache_policy", "coupled"),
        "camera_cache_window_size": payload.get(
            "camera_cache_window_size", payload.get("cache_window_size")
        ),
        "sequence": row.get("sequence"),
        "status": row.get("status"),
        "error": row.get("error"),
        "num_frames": row.get("num_frames"),
        "abs_rel": row.get("abs_rel"),
        "rmse": row.get("rmse"),
        "delta_1": row.get("delta_1"),
        "ate": row.get("ate"),
        "rpe_trans": row.get("rpe_trans"),
        "rpe_rot_deg": row.get("rpe_rot_deg"),
        "loop_translation_mean": row.get("loop_translation_mean"),
        "loop_rotation_deg_mean": row.get("loop_rotation_deg_mean"),
        "loop_depth_abs_rel": row.get("loop_depth_abs_rel"),
        "inference_sec": row.get("inference_sec"),
        "fps_inference": row.get("fps_inference"),
        "mean_frame_latency_ms": row.get("mean_frame_latency_ms"),
        "last_frame_latency_ms": row.get("last_frame_latency_ms"),
        "peak_allocated_mb": row.get("peak_allocated_mb"),
        "peak_reserved_mb": row.get("peak_reserved_mb"),
        "mean_retained_age": selection.get("mean_retained_age"),
        "max_retained_age": selection.get("max_retained_age"),
        "mean_temporal_span": selection.get("mean_temporal_span"),
        "anchor0_retention_rate": selection.get("anchor0_retention_rate"),
        "mean_selection_churn": selection.get("mean_selection_churn"),
        "unique_retained_frames": selection.get("unique_retained_frames"),
        "loop_match_retention_rate": selection.get("loop_match_retention_rate"),
        "final_retained_frame_ids": json.dumps(selection.get("final_retained_frame_ids")),
        "mean_camera_retained_age": camera_selection.get("mean_retained_age"),
        "max_camera_retained_age": camera_selection.get("max_retained_age"),
        "mean_camera_temporal_span": camera_selection.get("mean_temporal_span"),
        "camera_anchor0_retention_rate": camera_selection.get(
            "anchor0_retention_rate"
        ),
        "camera_unique_retained_frames": camera_selection.get(
            "unique_retained_frames"
        ),
        "final_camera_retained_frame_ids": json.dumps(
            camera_selection.get("final_retained_frame_ids")
        ),
        "near_bank_occupancy_rate": temporal_banks.get("near_occupancy_rate"),
        "middle_bank_occupancy_rate": temporal_banks.get("middle_occupancy_rate"),
        "long_bank_occupancy_rate": temporal_banks.get("long_occupancy_rate"),
        "near_bank_updates": temporal_banks.get("near_updates"),
        "middle_bank_updates": temporal_banks.get("middle_updates"),
        "long_bank_updates": temporal_banks.get("long_updates"),
        "near_bank_unique_frames": temporal_banks.get("near_unique_frames"),
        "middle_bank_unique_frames": temporal_banks.get("middle_unique_frames"),
        "long_bank_unique_frames": temporal_banks.get("long_unique_frames"),
        "final_max_temporal_gap": temporal_banks.get("final_max_temporal_gap"),
        "final_temporal_bank_frame_ids": json.dumps(
            temporal_banks.get("final_bank_frame_ids")
        ),
        "result_dir": result_dir,
    }


def main():
    parser = argparse.ArgumentParser("Summarize Stage 3.4 long-sequence runs")
    parser.add_argument("--results-root", default="eval_results/stage3_4")
    parser.add_argument("--output", default="stage3_4_results.csv")
    parser.add_argument("--sequence-output", default="stage3_4_sequence_results.csv")
    args = parser.parse_args()

    paths = sorted(glob.glob(os.path.join(args.results_root, "*", "stage3_4_metrics.json")))
    if not paths:
        raise RuntimeError(f"no Stage 3.4 metrics below {args.results_root}")
    aggregate_rows = []
    sequence_rows = []
    for path in paths:
        result_dir = os.path.dirname(path)
        with open(path) as handle:
            payload = json.load(handle)
        dataset = payload["dataset"]
        method = method_name(result_dir, dataset)
        successful = [row for row in payload["sequences"] if row.get("status") == "ok"]
        aggregate_rows.append(metric_row(dataset, method, payload, successful, None, result_dir))
        for row in payload["sequences"]:
            sequence_rows.append(flatten_sequence(dataset, method, payload, row, result_dir))

        by_prefix = defaultdict(list)
        for row in successful:
            for prefix_row in row.get("prefix_metrics", []):
                by_prefix[prefix_row["prefix_frames"]].append(prefix_row)
        for prefix, prefix_rows in sorted(by_prefix.items()):
            aggregate_rows.append(
                metric_row(dataset, method, payload, prefix_rows, prefix, result_dir)
            )

    for output, rows, fieldnames in (
        (args.output, aggregate_rows, FIELDS),
        (args.sequence_output, sequence_rows, SEQUENCE_FIELDS),
    ):
        with open(output, "w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        print(f"Wrote {len(rows)} rows to {output}")


if __name__ == "__main__":
    main()
