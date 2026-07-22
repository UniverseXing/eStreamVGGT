#!/usr/bin/env python3
"""Gate bounded-window pose stitching before Stage 3.3A."""

import argparse
import csv


FIELDS = (
    "method",
    "run_ok",
    "ate_ok",
    "rotation_ok",
    "prefix_stable",
    "peak_memory_ok",
    "bounded_window_ok",
    "eligible_for_stage3_3a",
    "ate",
    "ate_limit",
    "rpe_trans",
    "rpe_rot_deg",
    "rotation_limit_deg",
    "max_prefix_rotation_deg",
    "max_prefix_rotation_step_deg",
    "peak_allocated_mb",
    "window_size",
    "overlap",
    "recompute_factor",
    "fps_unique_inference",
    "max_overlap_translation_rmse",
    "max_overlap_rotation_deg",
    "decision",
)


def number(row, key):
    value = row.get(key)
    return None if value in (None, "") else float(value)


def yes(value):
    return "yes" if value else "no"


def main():
    parser = argparse.ArgumentParser("Check Stage 3.6A pose-stitching gate")
    parser.add_argument("--input", default="stage3_6a_results.csv")
    parser.add_argument("--output", default="stage3_6a_gate.csv")
    parser.add_argument("--reference", default="full_cache")
    parser.add_argument(
        "--candidates",
        nargs="+",
        default=("window16_overlap4", "window32_overlap8"),
    )
    parser.add_argument("--ate-factor", type=float, default=2.0)
    parser.add_argument("--rotation-factor", type=float, default=1.5)
    parser.add_argument("--max-prefix-rotation-step-deg", type=float, default=5.0)
    parser.add_argument("--max-peak-allocated-mb", type=float, default=10240.0)
    args = parser.parse_args()

    with open(args.input, newline="") as handle:
        rows = list(csv.DictReader(handle))
    finals = {
        row["method"]: row for row in rows if row.get("prefix_frames") in (None, "")
    }
    if args.reference not in finals:
        raise RuntimeError(f"missing reference row {args.reference}")
    reference_ate = number(finals[args.reference], "ate")
    reference_rotation = number(finals[args.reference], "rpe_rot_deg")
    if reference_ate is None or reference_rotation is None:
        raise RuntimeError("reference is missing ATE or rotation RPE")
    ate_limit = reference_ate * args.ate_factor
    rotation_limit = reference_rotation * args.rotation_factor

    decisions = []
    for method in args.candidates:
        row = finals.get(method)
        if row is None:
            missing = {key: "" for key in FIELDS}
            missing.update(
                {
                    "method": method,
                    "run_ok": "no",
                    "ate_ok": "no",
                    "rotation_ok": "no",
                    "prefix_stable": "no",
                    "peak_memory_ok": "no",
                    "bounded_window_ok": "no",
                    "eligible_for_stage3_3a": "no",
                    "decision": "missing result",
                }
            )
            decisions.append(missing)
            continue

        ate = number(row, "ate")
        rpe_trans = number(row, "rpe_trans")
        rotation = number(row, "rpe_rot_deg")
        peak = number(row, "peak_allocated_mb")
        window = number(row, "window_size")
        overlap = number(row, "overlap")
        prefix_rows = sorted(
            (
                int(item["prefix_frames"]),
                item.get("status"),
                number(item, "rpe_rot_deg"),
            )
            for item in rows
            if item.get("method") == method
            and item.get("prefix_frames") not in (None, "")
        )
        prefix_rotations = [value for _, status, value in prefix_rows if status == "ok" and value is not None]
        max_prefix_rotation = max(prefix_rotations, default=None)
        max_rotation_step = max(
            (
                current - previous
                for previous, current in zip(prefix_rotations, prefix_rotations[1:])
            ),
            default=0.0,
        )

        run_ok = row.get("status") == "ok"
        ate_ok = ate is not None and ate <= ate_limit
        rotation_ok = rotation is not None and rotation <= rotation_limit
        prefix_stable = (
            bool(prefix_rows)
            and len(prefix_rotations) == len(prefix_rows)
            and max_prefix_rotation <= rotation_limit
            and max_rotation_step <= args.max_prefix_rotation_step_deg
        )
        peak_ok = peak is not None and peak < args.max_peak_allocated_mb
        bounded = (
            row.get("mode") == "window_stitch"
            and window is not None
            and overlap is not None
            and 3 <= overlap < window
        )
        eligible = all(
            (run_ok, ate_ok, rotation_ok, prefix_stable, peak_ok, bounded)
        )
        failures = [
            name
            for name, value in (
                ("run", run_ok),
                ("ATE", ate_ok),
                ("rotation", rotation_ok),
                ("prefix", prefix_stable),
                ("peak_memory", peak_ok),
                ("bounded_window", bounded),
            )
            if not value
        ]
        decisions.append(
            {
                "method": method,
                "run_ok": yes(run_ok),
                "ate_ok": yes(ate_ok),
                "rotation_ok": yes(rotation_ok),
                "prefix_stable": yes(prefix_stable),
                "peak_memory_ok": yes(peak_ok),
                "bounded_window_ok": yes(bounded),
                "eligible_for_stage3_3a": yes(eligible),
                "ate": ate,
                "ate_limit": ate_limit,
                "rpe_trans": rpe_trans,
                "rpe_rot_deg": rotation,
                "rotation_limit_deg": rotation_limit,
                "max_prefix_rotation_deg": max_prefix_rotation,
                "max_prefix_rotation_step_deg": max_rotation_step,
                "peak_allocated_mb": peak,
                "window_size": window,
                "overlap": overlap,
                "recompute_factor": number(row, "recompute_factor"),
                "fps_unique_inference": number(row, "fps_unique_inference"),
                "max_overlap_translation_rmse": number(
                    row, "max_overlap_translation_rmse"
                ),
                "max_overlap_rotation_deg": number(
                    row, "max_overlap_rotation_deg"
                ),
                "decision": "PASS" if eligible else "FAIL: " + ", ".join(failures),
            }
        )

    with open(args.output, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(decisions)
    passing = [
        row["method"] for row in decisions if row["eligible_for_stage3_3a"] == "yes"
    ]
    print(f"Wrote {len(decisions)} Stage 3.6A decisions to {args.output}")
    print("Eligible for Stage 3.3A:", passing or "none")


if __name__ == "__main__":
    main()
