#!/usr/bin/env python3
"""Apply the long-sequence quality gate before an incremental Stage 3.3 rerun."""

import argparse
import csv


DEFAULT_CANDIDATES = (
    "split_k4_camera4",
    "split_k4_camera8",
    "split_k4_camera16",
    "split_k4_camera_full",
    "recent_dino_k6",
)

FIELDS = (
    "method",
    "all_sequences_ok",
    "depth_ok",
    "ate_ok",
    "rotation_ok",
    "prefix_stable",
    "pose_diagnostic_pass",
    "peak_memory_ok",
    "aggregator_bounded",
    "camera_bounded",
    "eligible_for_stage3_3",
    "abs_rel",
    "abs_rel_limit",
    "ate",
    "ate_limit",
    "rpe_rot_deg",
    "rpe_rot_limit_deg",
    "max_prefix_rpe_rot_deg",
    "max_prefix_rotation_step_deg",
    "peak_allocated_mb",
    "aggregator_kv_mib",
    "decision",
)


def number(row, key):
    value = row.get(key)
    if value in (None, ""):
        return None
    return float(value)


def passed(value):
    return "yes" if value else "no"


def main():
    parser = argparse.ArgumentParser(
        "Check whether long-sequence candidates may enter incremental Stage 3.3"
    )
    parser.add_argument("--input", default="stage3_5b_results.csv")
    parser.add_argument("--output", default="stage3_5b_gate.csv")
    parser.add_argument("--reference", default="full_cache")
    parser.add_argument("--candidates", nargs="+", default=DEFAULT_CANDIDATES)
    parser.add_argument("--depth-factor", type=float, default=1.10)
    parser.add_argument("--ate-factor", type=float, default=2.0)
    parser.add_argument("--rotation-factor", type=float, default=1.50)
    parser.add_argument("--max-prefix-rotation-step-deg", type=float, default=5.0)
    parser.add_argument("--max-peak-allocated-mb", type=float, default=10240.0)
    parser.add_argument("--aggregator-mib-per-frame", type=float, default=100.0)
    parser.add_argument(
        "--require-pass",
        action="store_true",
        help="return a non-zero status when no deployable candidate passes",
    )
    args = parser.parse_args()

    with open(args.input, newline="") as handle:
        rows = list(csv.DictReader(handle))
    final_rows = {
        row["method"]: row for row in rows if row.get("prefix_frames") in (None, "")
    }
    if args.reference not in final_rows:
        raise RuntimeError(f"missing final reference row: {args.reference}")
    reference = final_rows[args.reference]
    reference_abs_rel = number(reference, "mean_abs_rel")
    reference_ate = number(reference, "mean_ate")
    reference_rotation = number(reference, "mean_rpe_rot_deg")
    if None in (reference_abs_rel, reference_ate, reference_rotation):
        raise RuntimeError("full-cache reference is missing depth or pose metrics")

    abs_rel_limit = reference_abs_rel * args.depth_factor
    ate_limit = reference_ate * args.ate_factor
    rotation_limit = reference_rotation * args.rotation_factor
    results = []
    for method in args.candidates:
        row = final_rows.get(method)
        if row is None:
            missing = {key: "" for key in FIELDS}
            missing.update(
                {
                    "method": method,
                    "all_sequences_ok": "no",
                    "depth_ok": "no",
                    "ate_ok": "no",
                    "rotation_ok": "no",
                    "prefix_stable": "no",
                    "pose_diagnostic_pass": "no",
                    "peak_memory_ok": "no",
                    "aggregator_bounded": "no",
                    "camera_bounded": "no",
                    "eligible_for_stage3_3": "no",
                    "decision": "missing result",
                }
            )
            results.append(missing)
            continue

        abs_rel = number(row, "mean_abs_rel")
        ate = number(row, "mean_ate")
        rotation = number(row, "mean_rpe_rot_deg")
        peak = number(row, "max_peak_allocated_mb")
        aggregator_kv = number(row, "mean_aggregator_kv_mib")
        cache_window = number(row, "cache_window_size")
        method_prefixes = sorted(
            (
                int(prefix_row["prefix_frames"]),
                number(prefix_row, "mean_rpe_rot_deg"),
            )
            for prefix_row in rows
            if prefix_row.get("method") == method
            and prefix_row.get("prefix_frames") not in (None, "")
        )
        prefix_rotations = [
            value for _, value in method_prefixes if value is not None
        ]
        max_prefix_rotation = max(prefix_rotations, default=None)
        positive_steps = [
            current - previous
            for previous, current in zip(prefix_rotations, prefix_rotations[1:])
        ]
        max_positive_step = max(positive_steps, default=0.0)

        all_sequences_ok = (
            number(row, "num_failed") == 0
            and number(row, "num_successful") is not None
            and number(row, "num_successful") > 0
        )
        depth_ok = abs_rel is not None and abs_rel <= abs_rel_limit
        ate_ok = ate is not None and ate <= ate_limit
        rotation_ok = rotation is not None and rotation <= rotation_limit
        prefix_stable = (
            bool(prefix_rotations)
            and max_prefix_rotation <= rotation_limit
            and max_positive_step <= args.max_prefix_rotation_step_deg
        )
        pose_diagnostic_pass = all(
            (all_sequences_ok, ate_ok, rotation_ok, prefix_stable)
        )
        peak_ok = peak is not None and peak < args.max_peak_allocated_mb
        aggregator_bounded = (
            aggregator_kv is not None
            and cache_window is not None
            and aggregator_kv <= args.aggregator_mib_per_frame * cache_window
        )
        camera_policy = row.get("camera_cache_policy", "coupled")
        camera_bounded = camera_policy not in ("full", "full_cache")
        eligible = all(
            (
                all_sequences_ok,
                depth_ok,
                ate_ok,
                rotation_ok,
                prefix_stable,
                peak_ok,
                aggregator_bounded,
                camera_bounded,
            )
        )
        failed_checks = [
            name
            for name, value in (
                ("sequences", all_sequences_ok),
                ("depth", depth_ok),
                ("ATE", ate_ok),
                ("rotation", rotation_ok),
                ("prefix", prefix_stable),
                ("peak_memory", peak_ok),
                ("aggregator_bound", aggregator_bounded),
                ("camera_bound", camera_bounded),
            )
            if not value
        ]
        results.append(
            {
                "method": method,
                "all_sequences_ok": passed(all_sequences_ok),
                "depth_ok": passed(depth_ok),
                "ate_ok": passed(ate_ok),
                "rotation_ok": passed(rotation_ok),
                "prefix_stable": passed(prefix_stable),
                "pose_diagnostic_pass": passed(pose_diagnostic_pass),
                "peak_memory_ok": passed(peak_ok),
                "aggregator_bounded": passed(aggregator_bounded),
                "camera_bounded": passed(camera_bounded),
                "eligible_for_stage3_3": passed(eligible),
                "abs_rel": abs_rel,
                "abs_rel_limit": abs_rel_limit,
                "ate": ate,
                "ate_limit": ate_limit,
                "rpe_rot_deg": rotation,
                "rpe_rot_limit_deg": rotation_limit,
                "max_prefix_rpe_rot_deg": max_prefix_rotation,
                "max_prefix_rotation_step_deg": max_positive_step,
                "peak_allocated_mb": peak,
                "aggregator_kv_mib": aggregator_kv,
                "decision": "PASS" if eligible else "FAIL: " + ", ".join(failed_checks),
            }
        )

    with open(args.output, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(results)

    eligible_methods = [
        row["method"] for row in results if row["eligible_for_stage3_3"] == "yes"
    ]
    pose_diagnostic_methods = [
        row["method"] for row in results if row["pose_diagnostic_pass"] == "yes"
    ]
    print(f"Wrote {len(results)} gate decisions to {args.output}")
    print("Passed the pose-only diagnostic:", pose_diagnostic_methods or "none")
    print("Eligible for incremental Stage 3.3:", eligible_methods or "none")
    if args.require_pass and not eligible_methods:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
