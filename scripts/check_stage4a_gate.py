#!/usr/bin/env python3
"""Apply the frozen same-GPU Stage 4A VideoDepth gate."""

import argparse
import csv
import json


DATASETS = {"bonn": 5, "kitti": 13, "sintel": 23}
CANDIDATES = ("stage3_2_k4", "old_dino_k6", "temporal_binned_dino_k8")
FIELDS = (
    "candidate",
    "same_gpu_ok",
    "bonn_coverage_ok",
    "bonn_quality_ok",
    "bonn_memory_below_full",
    "kitti_coverage_ok",
    "kitti_quality_ok",
    "kitti_memory_below_full",
    "sintel_coverage_ok",
    "sintel_quality_ok",
    "sintel_memory_below_full",
    "all_coverage_ok",
    "all_quality_ok",
    "all_memory_below_full",
    "temporal_vs_best_bounded_ok",
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
    parser.add_argument("--max-abs-rel-ratio", type=float, default=1.15)
    parser.add_argument("--max-delta1-drop", type=float, default=0.03)
    parser.add_argument("--temporal-max-best-ratio", type=float, default=1.10)
    parser.add_argument("--expected-gpu-substring", default="6000 Ada")
    args = parser.parse_args()

    with open(args.input, newline="") as handle:
        rows = list(csv.DictReader(handle))
    indexed = {(row["dataset"], row["method"]): row for row in rows}

    gpu_names = {
        row.get("gpu_name", "").strip()
        for row in rows
        if row.get("gpu_name", "").strip()
    }
    same_gpu_ok = bool(
        len(rows) == 12
        and len(gpu_names) == 1
        and args.expected_gpu_substring.lower() in next(iter(gpu_names)).lower()
    )

    outputs = []
    for candidate in CANDIDATES:
        dataset_checks = {}
        coverage_values = []
        quality_values = []
        memory_values = []
        temporal_values = []
        for dataset, expected_sequences in DATASETS.items():
            full = indexed.get((dataset, "full_cache"))
            row = indexed.get((dataset, candidate))
            full_frames = number(full, "total_frames")
            coverage_ok = bool(
                full
                and row
                and number(full, "num_sequences") == expected_sequences
                and number(full, "num_ok") == expected_sequences
                and number(full, "num_oom") == 0
                and number(row, "num_sequences") == expected_sequences
                and number(row, "num_ok") == expected_sequences
                and number(row, "num_oom") == 0
                and full_frames is not None
                and number(row, "total_frames") == full_frames
            )

            full_abs_rel = number(full, "abs_rel")
            full_delta1 = number(full, "delta_1")
            abs_limit = (
                full_abs_rel * args.max_abs_rel_ratio
                if full_abs_rel is not None
                else None
            )
            delta_limit = (
                full_delta1 - args.max_delta1_drop
                if full_delta1 is not None
                else None
            )
            quality_ok = bool(
                number(row, "abs_rel") is not None
                and abs_limit is not None
                and number(row, "abs_rel") <= abs_limit
                and number(row, "delta_1") is not None
                and delta_limit is not None
                and number(row, "delta_1") >= delta_limit
            )
            memory_ok = bool(
                number(row, "max_peak_allocated_mb") is not None
                and number(full, "max_peak_allocated_mb") is not None
                and number(row, "max_peak_reserved_mb") is not None
                and number(full, "max_peak_reserved_mb") is not None
                and number(row, "max_peak_allocated_mb")
                < number(full, "max_peak_allocated_mb")
                and number(row, "max_peak_reserved_mb")
                < number(full, "max_peak_reserved_mb")
            )

            temporal_ok = True
            best_abs_rel = None
            best_delta1 = None
            temporal_abs_limit = None
            temporal_delta_limit = None
            if candidate == "temporal_binned_dino_k8":
                bounded = [
                    indexed.get((dataset, method))
                    for method in ("stage3_2_k4", "old_dino_k6")
                ]
                if all(item is not None for item in bounded):
                    best_abs_rel = min(number(item, "abs_rel") for item in bounded)
                    best_delta1 = max(number(item, "delta_1") for item in bounded)
                    temporal_abs_limit = (
                        best_abs_rel * args.temporal_max_best_ratio
                    )
                    temporal_delta_limit = best_delta1 - args.max_delta1_drop
                    temporal_ok = bool(
                        number(row, "abs_rel") is not None
                        and number(row, "abs_rel") <= temporal_abs_limit
                        and number(row, "delta_1") is not None
                        and number(row, "delta_1") >= temporal_delta_limit
                    )
                else:
                    temporal_ok = False
                temporal_values.append(temporal_ok)

            coverage_values.append(coverage_ok)
            quality_values.append(quality_ok)
            memory_values.append(memory_ok)
            dataset_checks[dataset] = {
                "coverage_ok": coverage_ok,
                "quality_ok": quality_ok,
                "memory_below_full": memory_ok,
                "candidate_abs_rel": number(row, "abs_rel"),
                "full_abs_rel": full_abs_rel,
                "abs_rel_limit": abs_limit,
                "candidate_delta1": number(row, "delta_1"),
                "full_delta1": full_delta1,
                "delta1_limit": delta_limit,
                "best_bounded_abs_rel": best_abs_rel,
                "temporal_abs_rel_limit": temporal_abs_limit,
                "best_bounded_delta1": best_delta1,
                "temporal_delta1_limit": temporal_delta_limit,
                "temporal_vs_best_bounded_ok": temporal_ok,
            }

        all_coverage_ok = all(coverage_values)
        all_quality_ok = all(quality_values)
        all_memory_ok = all(memory_values)
        temporal_ok = all(temporal_values) if temporal_values else True
        base_pass = bool(
            same_gpu_ok and all_coverage_ok and all_quality_ok and all_memory_ok
        )
        eligible = base_pass
        if not base_pass:
            decision = "FAIL"
        elif candidate == "temporal_binned_dino_k8" and not temporal_ok:
            decision = "PASS_SPECIALIST_ONLY"
        else:
            decision = "PASS"

        output = {
            "candidate": candidate,
            "same_gpu_ok": yes(same_gpu_ok),
            "all_coverage_ok": yes(all_coverage_ok),
            "all_quality_ok": yes(all_quality_ok),
            "all_memory_below_full": yes(all_memory_ok),
            "temporal_vs_best_bounded_ok": (
                yes(temporal_ok)
                if candidate == "temporal_binned_dino_k8"
                else "n/a"
            ),
            "eligible_for_stage4b": yes(eligible),
            "checks_json": json.dumps(
                {
                    "gpu_names": sorted(gpu_names),
                    "datasets": dataset_checks,
                },
                sort_keys=True,
            ),
            "decision": decision,
        }
        for dataset in DATASETS:
            output[f"{dataset}_coverage_ok"] = yes(
                dataset_checks[dataset]["coverage_ok"]
            )
            output[f"{dataset}_quality_ok"] = yes(
                dataset_checks[dataset]["quality_ok"]
            )
            output[f"{dataset}_memory_below_full"] = yes(
                dataset_checks[dataset]["memory_below_full"]
            )
        outputs.append(output)

    with open(args.output, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(outputs)
    print(f"Wrote {len(outputs)} Stage 4A gate rows to {args.output}")


if __name__ == "__main__":
    main()
