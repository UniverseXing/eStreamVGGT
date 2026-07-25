#!/usr/bin/env python3
"""Apply the frozen Stage 4C long-sequence deployment gate."""

import argparse
import csv
import json
import math


BOUNDED_CONFIGS = {
    "stage3_2_k4": ("4", "anchor_recent_dino_diverse_2old_1recent"),
    "old_dino_k6": ("6", "anchor_recent_dino_diverse"),
    "temporal_binned_dino_k8": ("8", "temporal_binned_dino_k8"),
}
FIELDS = (
    "candidate",
    "role",
    "coverage_ok",
    "run_ok",
    "same_gpu_ok",
    "streaming_semantics_ok",
    "frozen_cache_config_ok",
    "peak_memory_ok",
    "gpu_plateau_ok",
    "cpu_rss_plateau_ok",
    "pose_eval_ok",
    "pose_catastrophe_free",
    "max_peak_allocated_mb",
    "max_gpu_1000_minus_500_mb",
    "max_rss_1000_minus_500_mib",
    "max_pose_ate_ratio_to_best_bounded",
    "max_pose_rot_ratio_to_best_bounded",
    "full_success_ceiling_json",
    "decision",
)


def number(row, key):
    value = None if row is None else row.get(key)
    if value in (None, ""):
        return None
    return float(value)


def yes(value):
    return "yes" if value else "no"


def safe_ratio(value, oracle):
    if value is None or oracle is None:
        return math.inf
    return value / max(oracle, 1e-12)


def main():
    parser = argparse.ArgumentParser("Check Stage 4C gate")
    parser.add_argument("--input", default="stage4c_results.csv")
    parser.add_argument("--output", default="stage4c_gate.csv")
    parser.add_argument(
        "--sequences",
        nargs="+",
        default=(
            "rgbd_dataset_freiburg1_room",
            "rgbd_dataset_freiburg2_desk",
            "rgbd_dataset_freiburg3_long_office_household",
        ),
    )
    parser.add_argument(
        "--lengths", type=int, nargs="+", default=(100, 250, 500, 1000)
    )
    parser.add_argument("--max-peak-allocated-mb", type=float, default=12288.0)
    parser.add_argument("--max-gpu-growth-mb", type=float, default=256.0)
    parser.add_argument("--max-rss-growth-mib", type=float, default=256.0)
    parser.add_argument("--max-pose-ratio", type=float, default=2.0)
    args = parser.parse_args()

    with open(args.input, newline="") as handle:
        rows = list(csv.DictReader(handle))
    by_key = {
        (row["method"], row["sequence"], int(row["num_frames"])): row
        for row in rows
    }
    if len(by_key) != len(rows):
        raise ValueError("duplicate method/sequence/length rows")

    successful = [row for row in rows if row["status"] == "ok"]
    gpu_names = {row["gpu_name"] for row in successful}
    same_gpu_ok = (
        len(gpu_names) == 1
        and all("6000 Ada" in name for name in gpu_names)
    )

    bounded_oracles = {}
    for sequence in args.sequences:
        for length in args.lengths:
            cell = [
                by_key.get((method, sequence, length))
                for method in BOUNDED_CONFIGS
            ]
            for metric in ("ate", "rpe_rot_deg"):
                values = [
                    number(row, metric)
                    for row in cell
                    if row is not None
                    and row.get("status") == "ok"
                    and row.get("pose_status") == "ok"
                    and number(row, metric) is not None
                ]
                bounded_oracles[(sequence, length, metric)] = (
                    min(values) if values else None
                )

    outputs = []
    for method, (window, policy) in BOUNDED_CONFIGS.items():
        expected = [
            by_key.get((method, sequence, length))
            for sequence in args.sequences
            for length in args.lengths
        ]
        coverage_ok = all(row is not None for row in expected)
        run_ok = coverage_ok and all(
            row["status"] == "ok"
            and int(row.get("processed_frames") or 0)
            == int(row["num_frames"])
            for row in expected
        )
        semantics_ok = run_ok and all(
            row.get("mode") == "stream_release"
            and row.get("input_mode") == "streaming"
            and row.get("output_mode") == "sink"
            and (number(row, "max_retained_outputs_mib") or 0.0) <= 1e-9
            and (number(row, "max_retained_views_mib") or 0.0) <= 1e-9
            for row in expected
        )
        config_ok = coverage_ok and all(
            row.get("cache_window_size") == window
            and row.get("cache_policy") == policy
            for row in expected
        )
        peaks = [number(row, "peak_allocated_mb") for row in expected]
        peak_ok = run_ok and all(
            value is not None and value < args.max_peak_allocated_mb
            for value in peaks
        )
        gpu_growths = []
        rss_growths = []
        for sequence in args.sequences:
            row_500 = by_key.get((method, sequence, 500))
            row_1000 = by_key.get((method, sequence, 1000))
            gpu_500 = number(row_500, "peak_allocated_mb")
            gpu_1000 = number(row_1000, "peak_allocated_mb")
            rss_500 = number(row_500, "rss_peak_mib")
            rss_1000 = number(row_1000, "rss_peak_mib")
            gpu_growths.append(
                gpu_1000 - gpu_500
                if gpu_500 is not None and gpu_1000 is not None
                else math.inf
            )
            rss_growths.append(
                rss_1000 - rss_500
                if rss_500 is not None and rss_1000 is not None
                else math.inf
            )
        max_gpu_growth = max(gpu_growths)
        max_rss_growth = max(rss_growths)
        gpu_plateau_ok = max_gpu_growth <= args.max_gpu_growth_mb
        rss_plateau_ok = max_rss_growth <= args.max_rss_growth_mib
        pose_ok = run_ok and all(
            row.get("pose_status") == "ok"
            and number(row, "ate") is not None
            and number(row, "rpe_rot_deg") is not None
            for row in expected
        )
        ate_ratios = []
        rot_ratios = []
        for row in expected:
            sequence = row["sequence"] if row else ""
            length = int(row["num_frames"]) if row else 0
            ate_ratios.append(
                safe_ratio(
                    number(row, "ate"),
                    bounded_oracles.get((sequence, length, "ate")),
                )
            )
            rot_ratios.append(
                safe_ratio(
                    number(row, "rpe_rot_deg"),
                    bounded_oracles.get(
                        (sequence, length, "rpe_rot_deg")
                    ),
                )
            )
        max_ate_ratio = max(ate_ratios)
        max_rot_ratio = max(rot_ratios)
        catastrophe_free = (
            pose_ok
            and max_ate_ratio <= args.max_pose_ratio
            and max_rot_ratio <= args.max_pose_ratio
        )
        checks = {
            "coverage": coverage_ok,
            "run": run_ok,
            "same_gpu": same_gpu_ok,
            "streaming_semantics": semantics_ok,
            "frozen_cache_config": config_ok,
            "peak_memory": peak_ok,
            "gpu_plateau": gpu_plateau_ok,
            "cpu_rss_plateau": rss_plateau_ok,
            "pose_eval": pose_ok,
            "pose_catastrophe": catastrophe_free,
        }
        failures = [key for key, value in checks.items() if not value]
        role = {
            "stage3_2_k4": "primary_bounded_deployment",
            "old_dino_k6": "robust_bounded_alternative",
            "temporal_binned_dino_k8": "long_sequence_pose_specialist",
        }[method]
        outputs.append(
            {
                "candidate": method,
                "role": role,
                "coverage_ok": yes(coverage_ok),
                "run_ok": yes(run_ok),
                "same_gpu_ok": yes(same_gpu_ok),
                "streaming_semantics_ok": yes(semantics_ok),
                "frozen_cache_config_ok": yes(config_ok),
                "peak_memory_ok": yes(peak_ok),
                "gpu_plateau_ok": yes(gpu_plateau_ok),
                "cpu_rss_plateau_ok": yes(rss_plateau_ok),
                "pose_eval_ok": yes(pose_ok),
                "pose_catastrophe_free": yes(catastrophe_free),
                "max_peak_allocated_mb": max(peaks)
                if all(value is not None for value in peaks)
                else "",
                "max_gpu_1000_minus_500_mb": max_gpu_growth,
                "max_rss_1000_minus_500_mib": max_rss_growth,
                "max_pose_ate_ratio_to_best_bounded": max_ate_ratio,
                "max_pose_rot_ratio_to_best_bounded": max_rot_ratio,
                "full_success_ceiling_json": "",
                "decision": "PASS"
                if not failures
                else "FAIL: " + ", ".join(failures),
            }
        )

    ceilings = {}
    full_reference_ok = True
    for sequence in args.sequences:
        sequence_rows = sorted(
            (
                row
                for row in rows
                if row["method"] == "full_cache"
                and row["sequence"] == sequence
            ),
            key=lambda row: int(row["num_frames"]),
        )
        successes = [
            int(row["num_frames"])
            for row in sequence_rows
            if row["status"] == "ok"
        ]
        failures = [
            int(row["num_frames"])
            for row in sequence_rows
            if row["status"] != "ok"
        ]
        ceilings[sequence] = {
            "max_successful_frames": max(successes) if successes else None,
            "first_failed_frames": min(failures) if failures else None,
        }
        if not sequence_rows or not successes:
            full_reference_ok = False
    outputs.append(
        {
            "candidate": "full_cache",
            "role": "quality_resource_reference",
            "coverage_ok": "n/a",
            "run_ok": yes(full_reference_ok),
            "same_gpu_ok": yes(same_gpu_ok),
            "streaming_semantics_ok": "n/a",
            "frozen_cache_config_ok": "n/a",
            "peak_memory_ok": "n/a",
            "gpu_plateau_ok": "n/a",
            "cpu_rss_plateau_ok": "n/a",
            "pose_eval_ok": "n/a",
            "pose_catastrophe_free": "n/a",
            "max_peak_allocated_mb": "",
            "max_gpu_1000_minus_500_mb": "",
            "max_rss_1000_minus_500_mib": "",
            "max_pose_ate_ratio_to_best_bounded": "",
            "max_pose_rot_ratio_to_best_bounded": "",
            "full_success_ceiling_json": json.dumps(
                ceilings, sort_keys=True
            ),
            "decision": "REFERENCE_ONLY"
            if full_reference_ok
            else "FAIL_REFERENCE_INCOMPLETE",
        }
    )

    with open(args.output, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(outputs)
    for row in outputs:
        print(row["candidate"], row["decision"])
    print(f"Wrote Stage 4C gate to {args.output}")


if __name__ == "__main__":
    main()
