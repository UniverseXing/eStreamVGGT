#!/usr/bin/env python3
"""Apply the predeclared Stage 3.6B equivalence and bounded-memory gate."""

import argparse
import csv


METRIC_KEYS = (
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
)

FIELDS = (
    "candidate",
    "run_ok",
    "prediction_hashes_equal",
    "metrics_equivalent",
    "max_metric_abs_diff",
    "streaming_semantics_ok",
    "bounded_cache_ok",
    "peak_memory_ok",
    "gpu_plateau_ok",
    "cpu_rss_plateau_ok",
    "throughput_ok",
    "stream_to_legacy_fps_ratio",
    "stream_110_peak_allocated_mb",
    "long_100_peak_allocated_mb",
    "long_500_peak_allocated_mb",
    "long_1000_peak_allocated_mb",
    "gpu_peak_1000_minus_500_mb",
    "rss_peak_1000_minus_500_mib",
    "max_stream_input_tensors_mib",
    "max_stream_retained_outputs_mib",
    "max_stream_retained_views_mib",
    "max_stream_aggregator_kv_mib",
    "eligible_for_geometry_ablation",
    "decision",
)


def number(row, key):
    if row is None:
        return None
    value = row.get(key)
    return None if value in (None, "") else float(value)


def yes(value):
    return "yes" if value else "no"


def main():
    parser = argparse.ArgumentParser("Check Stage 3.6B streaming-memory gate")
    parser.add_argument("--input", default="stage3_6b_results.csv")
    parser.add_argument("--output", default="stage3_6b_gate.csv")
    parser.add_argument("--legacy-method", default="bonn_legacy_110")
    parser.add_argument("--stream-method", default="bonn_stream_110")
    parser.add_argument("--long-method-prefix", default="7scenes_stream_")
    parser.add_argument("--long-lengths", type=int, nargs="+", default=(100, 500, 1000))
    parser.add_argument("--numeric-tolerance", type=float, default=1e-5)
    parser.add_argument("--max-peak-allocated-mb", type=float, default=10240.0)
    parser.add_argument("--max-gpu-growth-mb", type=float, default=256.0)
    parser.add_argument("--max-rss-growth-mib", type=float, default=256.0)
    parser.add_argument("--min-throughput-ratio", type=float, default=0.80)
    parser.add_argument("--max-aggregator-kv-mib", type=float, default=800.0)
    args = parser.parse_args()

    with open(args.input, newline="") as handle:
        rows = list(csv.DictReader(handle))
    by_method = {row["method"]: row for row in rows}
    legacy = by_method.get(args.legacy_method)
    stream = by_method.get(args.stream_method)
    long_rows = {
        length: by_method.get(f"{args.long_method_prefix}{length}")
        for length in args.long_lengths
    }
    required = [legacy, stream, *long_rows.values()]
    run_ok = all(row is not None and row.get("status") == "ok" for row in required)

    hashes_equal = bool(
        legacy
        and stream
        and legacy.get("camera_pose_sha256")
        and legacy.get("camera_pose_sha256") == stream.get("camera_pose_sha256")
        and legacy.get("depth_sha256")
        and legacy.get("depth_sha256") == stream.get("depth_sha256")
    )
    differences = []
    metrics_present = legacy is not None and stream is not None
    if legacy and stream:
        for key in METRIC_KEYS:
            reference = number(legacy, key)
            candidate = number(stream, key)
            if reference is not None and candidate is not None:
                differences.append(abs(candidate - reference))
            else:
                metrics_present = False
                differences.append(float("inf"))
    max_metric_difference = max(differences, default=float("inf"))
    metrics_equivalent = metrics_present and max_metric_difference <= args.numeric_tolerance

    stream_rows = [row for row in [stream, *long_rows.values()] if row is not None]
    legacy_input = number(legacy, "max_input_tensors_mib")
    single_frame_input_limit = (
        legacy_input / number(legacy, "num_frames") * 1.05
        if legacy_input is not None and number(legacy, "num_frames")
        else None
    )
    max_stream_input = max(
        (number(row, "max_input_tensors_mib") or float("inf") for row in stream_rows),
        default=float("inf"),
    )
    max_stream_outputs = max(
        (number(row, "max_retained_outputs_mib") or 0.0 for row in stream_rows),
        default=float("inf"),
    )
    max_stream_views = max(
        (number(row, "max_retained_views_mib") or 0.0 for row in stream_rows),
        default=float("inf"),
    )
    semantics_ok = bool(
        stream_rows
        and all(
            row.get("input_mode") == "streaming" and row.get("output_mode") == "sink"
            for row in stream_rows
        )
        and single_frame_input_limit is not None
        and max_stream_input <= single_frame_input_limit
        and max_stream_outputs <= 1e-9
        and max_stream_views <= 1e-9
    )

    max_aggregator_kv = max(
        (number(row, "max_aggregator_kv_mib") or float("inf") for row in stream_rows),
        default=float("inf"),
    )
    bounded_cache_ok = bool(
        stream_rows
        and all(number(row, "cache_window_size") == 8 for row in stream_rows)
        and all(row.get("cache_policy") == "temporal_binned_dino_k8" for row in stream_rows)
        and max_aggregator_kv <= args.max_aggregator_kv_mib
    )

    stream_peaks = [number(row, "peak_allocated_mb") for row in stream_rows]
    peak_memory_ok = bool(
        stream_peaks
        and all(value is not None and value < args.max_peak_allocated_mb for value in stream_peaks)
    )
    row_500 = long_rows.get(500)
    row_1000 = long_rows.get(1000)
    peak_500 = number(row_500, "peak_allocated_mb")
    peak_1000 = number(row_1000, "peak_allocated_mb")
    gpu_growth = (
        peak_1000 - peak_500
        if peak_500 is not None and peak_1000 is not None
        else float("inf")
    )
    gpu_plateau_ok = gpu_growth <= args.max_gpu_growth_mb

    rss_500 = number(row_500, "rss_peak_mib")
    rss_1000 = number(row_1000, "rss_peak_mib")
    rss_growth = (
        rss_1000 - rss_500
        if rss_500 is not None and rss_1000 is not None
        else float("inf")
    )
    cpu_rss_ok = rss_growth <= args.max_rss_growth_mib

    legacy_fps = number(legacy, "fps_end_to_end")
    stream_fps = number(stream, "fps_end_to_end")
    fps_ratio = (
        stream_fps / legacy_fps
        if legacy_fps is not None and stream_fps is not None and legacy_fps > 0
        else 0.0
    )
    throughput_ok = fps_ratio >= args.min_throughput_ratio

    checks = (
        ("run", run_ok),
        ("prediction_hash", hashes_equal),
        ("metric_equivalence", metrics_equivalent),
        ("streaming_semantics", semantics_ok),
        ("bounded_cache", bounded_cache_ok),
        ("peak_memory", peak_memory_ok),
        ("gpu_plateau", gpu_plateau_ok),
        ("cpu_rss_plateau", cpu_rss_ok),
        ("throughput", throughput_ok),
    )
    eligible = all(value for _, value in checks)
    failures = [name for name, value in checks if not value]
    result = {
        "candidate": "temporal_binned_dino_k8_stream_release",
        "run_ok": yes(run_ok),
        "prediction_hashes_equal": yes(hashes_equal),
        "metrics_equivalent": yes(metrics_equivalent),
        "max_metric_abs_diff": max_metric_difference,
        "streaming_semantics_ok": yes(semantics_ok),
        "bounded_cache_ok": yes(bounded_cache_ok),
        "peak_memory_ok": yes(peak_memory_ok),
        "gpu_plateau_ok": yes(gpu_plateau_ok),
        "cpu_rss_plateau_ok": yes(cpu_rss_ok),
        "throughput_ok": yes(throughput_ok),
        "stream_to_legacy_fps_ratio": fps_ratio,
        "stream_110_peak_allocated_mb": number(stream, "peak_allocated_mb"),
        "long_100_peak_allocated_mb": number(long_rows.get(100), "peak_allocated_mb"),
        "long_500_peak_allocated_mb": peak_500,
        "long_1000_peak_allocated_mb": peak_1000,
        "gpu_peak_1000_minus_500_mb": gpu_growth,
        "rss_peak_1000_minus_500_mib": rss_growth,
        "max_stream_input_tensors_mib": max_stream_input,
        "max_stream_retained_outputs_mib": max_stream_outputs,
        "max_stream_retained_views_mib": max_stream_views,
        "max_stream_aggregator_kv_mib": max_aggregator_kv,
        "eligible_for_geometry_ablation": yes(eligible),
        "decision": "PASS" if eligible else "FAIL: " + ", ".join(failures),
    }
    with open(args.output, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerow(result)
    print(result["decision"])
    print(f"Wrote Stage 3.6B gate to {args.output}")


if __name__ == "__main__":
    main()
