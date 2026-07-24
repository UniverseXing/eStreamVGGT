#!/usr/bin/env python3
"""Apply the frozen Stage 4A outdoor/cross-domain VideoDepth gate."""

import argparse
import csv
import json


CANDIDATES = ("stage3_2_k4", "old_dino_k6", "temporal_binned_dino_k8")
FIELDS = (
    "candidate",
    "kitti_coverage_ok",
    "kitti_quality_ok",
    "kitti_delta1_ok",
    "kitti_memory_below_full",
    "temporal_cross_domain_coverage_ok",
    "temporal_cross_domain_quality_ok",
    "eligible_for_stage4b",
    "checks_json",
    "decision",
)


def number(row, key):
    if row is None or row.get(key) in (None, ""):
        return None
    return float(row[key])


def yes(value):
    return "yes" if value else "no"


def main():
    parser = argparse.ArgumentParser("Check Stage 4A VideoDepth gate")
    parser.add_argument("--input", default="stage4a_video_depth_results.csv")
    parser.add_argument("--output", default="stage4a_gate.csv")
    parser.add_argument("--expected-kitti-sequences", type=int, default=13)
    parser.add_argument("--kitti-max-abs-rel-ratio", type=float, default=1.15)
    parser.add_argument("--cross-domain-max-abs-rel-ratio", type=float, default=1.10)
    parser.add_argument("--max-delta1-drop", type=float, default=0.03)
    args = parser.parse_args()

    with open(args.input, newline="") as handle:
        rows = list(csv.DictReader(handle))
    indexed = {(row["dataset"], row["method"]): row for row in rows}
    kitti_full = indexed.get(("kitti", "full_cache"))
    full_frames = number(kitti_full, "total_frames")
    outputs = []
    for candidate in CANDIDATES:
        row = indexed.get(("kitti", candidate))
        coverage_ok = bool(
            row
            and number(row, "num_sequences") == args.expected_kitti_sequences
            and number(row, "num_ok") == args.expected_kitti_sequences
            and number(row, "num_oom") == 0
            and number(row, "total_frames") == full_frames
        )
        kitti_abs_limit = (
            number(kitti_full, "abs_rel") * args.kitti_max_abs_rel_ratio
            if number(kitti_full, "abs_rel") is not None
            else None
        )
        kitti_delta_limit = (
            number(kitti_full, "delta_1") - args.max_delta1_drop
            if number(kitti_full, "delta_1") is not None
            else None
        )
        quality_ok = bool(
            number(row, "abs_rel") is not None
            and kitti_abs_limit is not None
            and number(row, "abs_rel") <= kitti_abs_limit
        )
        delta_ok = bool(
            number(row, "delta_1") is not None
            and kitti_delta_limit is not None
            and number(row, "delta_1") >= kitti_delta_limit
        )
        memory_ok = bool(
            number(row, "max_peak_allocated_mb") is not None
            and number(kitti_full, "max_peak_allocated_mb") is not None
            and number(row, "max_peak_reserved_mb") is not None
            and number(kitti_full, "max_peak_reserved_mb") is not None
            and number(row, "max_peak_allocated_mb")
            < number(kitti_full, "max_peak_allocated_mb")
            and number(row, "max_peak_reserved_mb")
            < number(kitti_full, "max_peak_reserved_mb")
        )

        cross_checks = {}
        cross_coverage = True
        cross_quality = True
        if candidate == "temporal_binned_dino_k8":
            for dataset, expected_sequences in (("bonn", 5), ("sintel", 23)):
                candidate_row = indexed.get((dataset, candidate))
                baselines = [
                    indexed.get((dataset, method))
                    for method in ("stage3_2_k4", "old_dino_k6")
                ]
                valid_baselines = [item for item in baselines if item is not None]
                if not valid_baselines:
                    raise RuntimeError(
                        f"missing K4/old-K6 frozen baselines for {dataset}"
                    )
                expected_frames = None
                full_row = indexed.get((dataset, "full_cache"))
                if full_row is not None:
                    expected_frames = number(full_row, "total_frames")
                dataset_coverage = bool(
                    candidate_row
                    and number(candidate_row, "num_sequences") == expected_sequences
                    and number(candidate_row, "num_ok") == expected_sequences
                    and number(candidate_row, "num_oom") == 0
                    and (
                        expected_frames is None
                        or number(candidate_row, "total_frames") == expected_frames
                    )
                )
                best_abs_rel = min(number(item, "abs_rel") for item in valid_baselines)
                best_delta1 = max(number(item, "delta_1") for item in valid_baselines)
                abs_limit = best_abs_rel * args.cross_domain_max_abs_rel_ratio
                delta_limit = best_delta1 - args.max_delta1_drop
                dataset_quality = bool(
                    number(candidate_row, "abs_rel") is not None
                    and number(candidate_row, "abs_rel") <= abs_limit
                    and number(candidate_row, "delta_1") is not None
                    and number(candidate_row, "delta_1") >= delta_limit
                )
                cross_coverage &= dataset_coverage
                cross_quality &= dataset_quality
                cross_checks[dataset] = {
                    "coverage_ok": dataset_coverage,
                    "candidate_abs_rel": number(candidate_row, "abs_rel"),
                    "abs_rel_limit": abs_limit,
                    "candidate_delta1": number(candidate_row, "delta_1"),
                    "delta1_limit": delta_limit,
                    "quality_ok": dataset_quality,
                }

        eligible = bool(
            coverage_ok
            and quality_ok
            and delta_ok
            and memory_ok
            and cross_coverage
            and cross_quality
        )
        checks = {
            "kitti": {
                "candidate_abs_rel": number(row, "abs_rel"),
                "abs_rel_limit": kitti_abs_limit,
                "candidate_delta1": number(row, "delta_1"),
                "delta1_limit": kitti_delta_limit,
            },
            "cross_domain": cross_checks,
        }
        outputs.append(
            {
                "candidate": candidate,
                "kitti_coverage_ok": yes(coverage_ok),
                "kitti_quality_ok": yes(quality_ok),
                "kitti_delta1_ok": yes(delta_ok),
                "kitti_memory_below_full": yes(memory_ok),
                "temporal_cross_domain_coverage_ok": yes(cross_coverage),
                "temporal_cross_domain_quality_ok": yes(cross_quality),
                "eligible_for_stage4b": yes(eligible),
                "checks_json": json.dumps(checks, sort_keys=True),
                "decision": "PASS" if eligible else "FAIL",
            }
        )

    with open(args.output, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(outputs)
    print(f"Wrote {len(outputs)} Stage 4A gate rows to {args.output}")


if __name__ == "__main__":
    main()
