#!/usr/bin/env python3
"""Gate the Stage 4E-A offline pose-composability screen."""

import argparse
import csv


VARIANTS = (
    "direct_k4_geometry_k8_pose",
    "component_k4_translation_k8_rotation",
)
FIELDS = (
    "candidate",
    "coverage_ok",
    "run_ok",
    "baseline_reproduction_ok",
    "max_baseline_abs_diff",
    "ate_ok",
    "max_ate_ratio_to_k4",
    "translation_rpe_ok",
    "max_translation_rpe_ratio_to_k8",
    "rotation_rpe_ok",
    "max_rotation_rpe_ratio_to_k8",
    "sequential_resource_proxy_ok",
    "max_sequential_peak_allocated_mb_proxy",
    "projected_online_memory_ok",
    "max_projected_online_peak_allocated_mb",
    "throughput_proxy_ok",
    "min_dual_fps_proxy",
    "full_100_mean_fps",
    "all_quality_ok",
    "eligible_for_stage4e_b",
    "decision",
)


def number(row, key):
    value = row.get(key)
    return None if value in (None, "") else float(value)


def yes(value):
    return "yes" if value else "no"


def main():
    parser = argparse.ArgumentParser("Check Stage 4E-A gate")
    parser.add_argument(
        "--input", default="stage4e_a_sequence_results.csv"
    )
    parser.add_argument("--output", default="stage4e_a_gate.csv")
    parser.add_argument("--expected-units", type=int, default=9)
    parser.add_argument("--numeric-tolerance", type=float, default=1e-5)
    parser.add_argument("--max-ate-ratio", type=float, default=1.10)
    parser.add_argument(
        "--max-translation-rpe-ratio", type=float, default=1.10
    )
    parser.add_argument(
        "--max-rotation-rpe-ratio", type=float, default=1.10
    )
    parser.add_argument(
        "--max-peak-allocated-mb", type=float, default=12288.0
    )
    args = parser.parse_args()

    with open(args.input, newline="") as handle:
        rows = list(csv.DictReader(handle))
    outputs = []
    for variant in VARIANTS:
        selected = [row for row in rows if row["variant"] == variant]
        unique_units = {
            (row["sequence"], int(row["num_frames"]))
            for row in selected
        }
        coverage_ok = (
            len(selected) == args.expected_units
            and len(unique_units) == args.expected_units
        )
        run_ok = coverage_ok and all(
            row.get("status") == "ok" for row in selected
        )

        baseline_differences = [
            number(row, "baseline_max_abs_diff") for row in selected
        ]
        max_baseline_difference = (
            max(baseline_differences)
            if baseline_differences
            and all(value is not None for value in baseline_differences)
            else float("inf")
        )
        baseline_ok = (
            run_ok and max_baseline_difference <= args.numeric_tolerance
        )

        ate_ratios = [number(row, "ate_ratio_to_k4") for row in selected]
        translation_ratios = [
            number(row, "rpe_trans_ratio_to_k8") for row in selected
        ]
        rotation_ratios = [
            number(row, "rpe_rot_ratio_to_k8") for row in selected
        ]
        max_ate = (
            max(ate_ratios)
            if ate_ratios and all(value is not None for value in ate_ratios)
            else float("inf")
        )
        max_translation = (
            max(translation_ratios)
            if translation_ratios
            and all(value is not None for value in translation_ratios)
            else float("inf")
        )
        max_rotation = (
            max(rotation_ratios)
            if rotation_ratios
            and all(value is not None for value in rotation_ratios)
            else float("inf")
        )
        ate_ok = run_ok and max_ate <= args.max_ate_ratio
        translation_ok = (
            run_ok
            and max_translation <= args.max_translation_rpe_ratio
        )
        rotation_ok = (
            run_ok and max_rotation <= args.max_rotation_rpe_ratio
        )

        sequential_peaks = [
            number(row, "sequential_peak_allocated_mb_proxy")
            for row in selected
        ]
        projected_peaks = [
            number(row, "projected_online_peak_allocated_mb")
            for row in selected
        ]
        dual_fps = [number(row, "dual_fps_proxy") for row in selected]
        full_fps = [number(row, "full_100_mean_fps") for row in selected]
        max_sequential_peak = (
            max(sequential_peaks)
            if sequential_peaks
            and all(value is not None for value in sequential_peaks)
            else float("inf")
        )
        max_projected_peak = (
            max(projected_peaks)
            if projected_peaks
            and all(value is not None for value in projected_peaks)
            else float("inf")
        )
        min_dual_fps = (
            min(dual_fps)
            if dual_fps and all(value is not None for value in dual_fps)
            else 0.0
        )
        full_100_mean_fps = (
            full_fps[0]
            if full_fps
            and all(value is not None for value in full_fps)
            else float("inf")
        )
        sequential_resource_ok = (
            run_ok and max_sequential_peak < args.max_peak_allocated_mb
        )
        projected_memory_ok = (
            run_ok and max_projected_peak < args.max_peak_allocated_mb
        )
        throughput_ok = run_ok and min_dual_fps > full_100_mean_fps
        quality_ok = (
            baseline_ok and ate_ok and translation_ok and rotation_ok
        )
        eligible = (
            quality_ok
            and sequential_resource_ok
            and projected_memory_ok
            and throughput_ok
        )
        failures = [
            name
            for name, passed in (
                ("coverage", coverage_ok),
                ("run", run_ok),
                ("baseline_reproduction", baseline_ok),
                ("ate", ate_ok),
                ("translation_rpe", translation_ok),
                ("rotation_rpe", rotation_ok),
                ("sequential_resource_proxy", sequential_resource_ok),
                ("projected_online_memory", projected_memory_ok),
                ("throughput_proxy", throughput_ok),
            )
            if not passed
        ]
        outputs.append(
            {
                "candidate": variant,
                "coverage_ok": yes(coverage_ok),
                "run_ok": yes(run_ok),
                "baseline_reproduction_ok": yes(baseline_ok),
                "max_baseline_abs_diff": max_baseline_difference,
                "ate_ok": yes(ate_ok),
                "max_ate_ratio_to_k4": max_ate,
                "translation_rpe_ok": yes(translation_ok),
                "max_translation_rpe_ratio_to_k8": max_translation,
                "rotation_rpe_ok": yes(rotation_ok),
                "max_rotation_rpe_ratio_to_k8": max_rotation,
                "sequential_resource_proxy_ok": yes(
                    sequential_resource_ok
                ),
                "max_sequential_peak_allocated_mb_proxy": max_sequential_peak,
                "projected_online_memory_ok": yes(projected_memory_ok),
                "max_projected_online_peak_allocated_mb": max_projected_peak,
                "throughput_proxy_ok": yes(throughput_ok),
                "min_dual_fps_proxy": min_dual_fps,
                "full_100_mean_fps": full_100_mean_fps,
                "all_quality_ok": yes(quality_ok),
                "eligible_for_stage4e_b": yes(eligible),
                "decision": (
                    "PASS_OFFLINE_COMPOSABILITY"
                    if eligible
                    else "FAIL: " + ", ".join(failures)
                ),
            }
        )

    with open(args.output, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(outputs)
    for row in outputs:
        print(row["candidate"], row["decision"])
    print(f"Wrote Stage 4E-A gate to {args.output}")


if __name__ == "__main__":
    main()
