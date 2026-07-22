#!/usr/bin/env python3
"""Apply the predeclared Stage 3.7 geometry/pose backtest gates."""

import argparse
import csv
import json


BASELINES = ("stage3_2_k4", "old_dino_k6")
CANDIDATE = "temporal_binned_dino_k8"
EXPECTED = {
    "pose": {"sintel": 14, "scannet": 6, "tum": 8},
    "static_recon": {"7scenes": 12, "nrgbd": 9, "eth3d": 13},
    "dynamic_recon": {"tum": 8},
}
FIELDS = (
    "candidate",
    "pose_coverage_ok",
    "static_coverage_ok",
    "dynamic_coverage_ok",
    "peak_memory_ok",
    "static_quality_ok",
    "static_nc_ok",
    "dynamic_quality_ok",
    "dynamic_nc_ok",
    "pose_catastrophe_guard_ok",
    "geometry_backtest_pass",
    "eligible_final_geometry_claim",
    "max_candidate_peak_allocated_mb",
    "quality_checks_json",
    "pose_checks_json",
    "decision",
)


def number(row, key):
    if row is None or row.get(key) in (None, ""):
        return None
    return float(row[key])


def yes(value):
    return "yes" if value else "no"


def index_rows(rows):
    indexed = {}
    for row in rows:
        key = (row["task"], row["dataset"], row["method"])
        if key in indexed:
            raise RuntimeError(f"duplicate comparison row: {key}")
        indexed[key] = row
    return indexed


def coverage_ok(indexed, task):
    for dataset, expected_successes in EXPECTED[task].items():
        row = indexed.get((task, dataset, CANDIDATE))
        if row is None:
            return False
        if number(row, "num_sequences") != expected_successes:
            return False
        if number(row, "num_successful") != expected_successes:
            return False
        if number(row, "num_failed") != 0:
            return False
    return True


def best_baseline(indexed, task, dataset, metric, operation):
    values = []
    for method in BASELINES:
        value = number(indexed.get((task, dataset, method)), metric)
        if value is None:
            raise RuntimeError(f"missing {task}/{dataset}/{method}/{metric} baseline")
        values.append(value)
    return operation(values)


def reconstruction_checks(indexed, task, datasets, quality_ratio, nc_drop):
    checks = {}
    quality_ok = True
    nc_ok = True
    for dataset in datasets:
        row = indexed.get((task, dataset, CANDIDATE))
        candidate_overall = number(row, "mean_overall")
        candidate_nc = number(row, "mean_nc")
        best_overall = best_baseline(indexed, task, dataset, "mean_overall", min)
        best_nc = best_baseline(indexed, task, dataset, "mean_nc", max)
        dataset_quality_ok = bool(
            candidate_overall is not None
            and candidate_overall <= best_overall * quality_ratio
        )
        dataset_nc_ok = bool(candidate_nc is not None and candidate_nc >= best_nc - nc_drop)
        quality_ok &= dataset_quality_ok
        nc_ok &= dataset_nc_ok
        checks[dataset] = {
            "candidate_overall": candidate_overall,
            "overall_limit": best_overall * quality_ratio,
            "overall_ok": dataset_quality_ok,
            "candidate_nc": candidate_nc,
            "nc_limit": best_nc - nc_drop,
            "nc_ok": dataset_nc_ok,
        }
    return quality_ok, nc_ok, checks


def pose_checks(indexed, multiplier):
    checks = {}
    all_ok = True
    for dataset in EXPECTED["pose"]:
        row = indexed.get(("pose", dataset, CANDIDATE))
        candidate_ate = number(row, "mean_ate")
        candidate_rotation = number(row, "mean_rpe_rot_deg")
        best_ate = best_baseline(indexed, "pose", dataset, "mean_ate", min)
        best_rotation = best_baseline(
            indexed, "pose", dataset, "mean_rpe_rot_deg", min
        )
        ate_ok = bool(candidate_ate is not None and candidate_ate <= best_ate * multiplier)
        rotation_ok = bool(
            candidate_rotation is not None
            and candidate_rotation <= best_rotation * multiplier
        )
        all_ok &= ate_ok and rotation_ok
        checks[dataset] = {
            "candidate_ate": candidate_ate,
            "ate_limit": best_ate * multiplier,
            "ate_ok": ate_ok,
            "candidate_rpe_rot_deg": candidate_rotation,
            "rpe_rot_deg_limit": best_rotation * multiplier,
            "rpe_rot_deg_ok": rotation_ok,
        }
    return all_ok, checks


def main():
    parser = argparse.ArgumentParser("Check the Stage 3.7 backtest gates")
    parser.add_argument("--input", default="stage3_7_comparison.csv")
    parser.add_argument("--output", default="stage3_7_gate.csv")
    parser.add_argument("--max-peak-allocated-mb", type=float, default=10240.0)
    parser.add_argument("--quality-ratio", type=float, default=1.10)
    parser.add_argument("--max-nc-drop", type=float, default=0.03)
    parser.add_argument("--pose-catastrophe-multiplier", type=float, default=2.0)
    args = parser.parse_args()

    with open(args.input, newline="") as handle:
        indexed = index_rows(csv.DictReader(handle))

    pose_coverage = coverage_ok(indexed, "pose")
    static_coverage = coverage_ok(indexed, "static_recon")
    dynamic_coverage = coverage_ok(indexed, "dynamic_recon")
    candidate_rows = [
        row for (*_, method), row in indexed.items() if method == CANDIDATE
    ]
    peaks = [number(row, "max_peak_allocated_mb") for row in candidate_rows]
    memory_ok = bool(
        candidate_rows
        and all(peak is not None and peak < args.max_peak_allocated_mb for peak in peaks)
    )
    max_peak = max((peak for peak in peaks if peak is not None), default=None)

    static_quality, static_nc, static_checks = reconstruction_checks(
        indexed,
        "static_recon",
        EXPECTED["static_recon"],
        args.quality_ratio,
        args.max_nc_drop,
    )
    dynamic_quality, dynamic_nc, dynamic_checks = reconstruction_checks(
        indexed,
        "dynamic_recon",
        EXPECTED["dynamic_recon"],
        args.quality_ratio,
        args.max_nc_drop,
    )
    pose_ok, pose_detail = pose_checks(indexed, args.pose_catastrophe_multiplier)

    complete = pose_coverage and static_coverage and dynamic_coverage
    geometry_pass = bool(
        static_coverage
        and dynamic_coverage
        and memory_ok
        and static_quality
        and static_nc
        and dynamic_quality
        and dynamic_nc
    )
    eligible = complete and geometry_pass
    if not complete:
        decision = "FAIL_INCOMPLETE"
    elif not geometry_pass:
        decision = "FAIL_GEOMETRY"
    elif not pose_ok:
        decision = "PASS_GEOMETRY_WITH_POSE_LIMITATION"
    else:
        decision = "PASS_ALL_BACKTESTS"

    output_row = {
        "candidate": CANDIDATE,
        "pose_coverage_ok": yes(pose_coverage),
        "static_coverage_ok": yes(static_coverage),
        "dynamic_coverage_ok": yes(dynamic_coverage),
        "peak_memory_ok": yes(memory_ok),
        "static_quality_ok": yes(static_quality),
        "static_nc_ok": yes(static_nc),
        "dynamic_quality_ok": yes(dynamic_quality),
        "dynamic_nc_ok": yes(dynamic_nc),
        "pose_catastrophe_guard_ok": yes(pose_ok),
        "geometry_backtest_pass": yes(geometry_pass),
        "eligible_final_geometry_claim": yes(eligible),
        "max_candidate_peak_allocated_mb": max_peak,
        "quality_checks_json": json.dumps(
            {"static": static_checks, "dynamic": dynamic_checks}, sort_keys=True
        ),
        "pose_checks_json": json.dumps(pose_detail, sort_keys=True),
        "decision": decision,
    }
    with open(args.output, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerow(output_row)
    print(f"Stage 3.7 decision: {decision}")
    print(f"Wrote Stage 3.7 gate to {args.output}")


if __name__ == "__main__":
    main()
