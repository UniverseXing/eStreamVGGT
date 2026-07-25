#!/usr/bin/env python3
"""Freeze cross-task roles from existing Stage 3/4 evidence."""

import argparse
import csv
import json
import math
import statistics
import tarfile
from collections import defaultdict


METHODS = (
    "full_cache",
    "stage3_2_k4",
    "old_dino_k6",
    "temporal_binned_dino_k8",
)
BOUNDED_METHODS = METHODS[1:]
METHOD_ORDER = {method: index for index, method in enumerate(METHODS)}
EXPECTED_DATASETS = {
    "video_depth": {"bonn": 5, "kitti": 13, "sintel": 23},
    "pose": {"scannet": 1, "sintel": 1, "tum": 1},
    "static_recon": {"7scenes": 12, "nrgbd": 9, "eth3d": 13},
    "dynamic_recon": {"tum": 8},
}

SUMMARY_FIELDS = (
    "task",
    "dataset",
    "method",
    "evaluation_unit",
    "num_units",
    "total_frames",
    "primary_metric",
    "primary_direction",
    "primary_value",
    "secondary_metric",
    "secondary_direction",
    "secondary_value",
    "total_inference_sec",
    "fps_inference",
    "max_peak_allocated_mb",
    "max_peak_reserved_mb",
    "coverage_ok",
    "source",
)
REGRET_FIELDS = (
    "task",
    "dataset",
    "evaluation_unit",
    "metric",
    "direction",
    "oracle_scope",
    "method",
    "n_units",
    "mean_normalized_regret",
    "median_normalized_regret",
    "max_normalized_regret",
    "oracle_wins",
)
ROLE_FIELDS = (
    "method",
    "bounded",
    "stage4a_eligible",
    "num_benchmarks",
    "primary_oracle_wins",
    "mean_macro_primary_regret",
    "median_macro_primary_regret",
    "max_macro_primary_regret",
    "video_depth_mean_regret",
    "pose_mean_regret",
    "static_recon_mean_regret",
    "dynamic_recon_mean_regret",
    "video_depth_pareto_datasets",
    "max_peak_allocated_mb",
    "final_role",
    "status",
    "rationale",
)
CLAIM_FIELDS = (
    "claim_id",
    "status",
    "claim",
    "evidence",
    "allowed_wording",
    "forbidden_wording",
    "next_action",
)


def read_csv(path):
    with open(path, newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path, fields, rows):
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} rows to {path}")


def as_float(value):
    if value in (None, ""):
        return None
    return float(value)


def as_int(value):
    if value in (None, ""):
        return None
    return int(float(value))


def close(a, b):
    return math.isclose(a, b, rel_tol=1e-8, abs_tol=1e-10)


def normalize_method(row):
    policy = row.get("cache_policy", "")
    window = str(row.get("cache_window_size", "") or "")
    if policy == "full_cache" and not window:
        return "full_cache"
    if policy == "anchor_recent_dino_diverse_2old_1recent" and window == "4":
        return "stage3_2_k4"
    if policy == "anchor_recent_dino_diverse" and window == "6":
        return "old_dino_k6"
    if policy == "temporal_binned_dino_k8" and window == "8":
        return "temporal_binned_dino_k8"
    method = row.get("method", "")
    if method.startswith("dense_"):
        method = method[len("dense_") :]
    return method if method in METHODS else None


def quality_unit(
    task,
    dataset,
    unit,
    method,
    num_frames,
    primary_metric,
    primary_direction,
    primary_value,
    secondary_metric,
    secondary_direction,
    secondary_value,
    inference_sec,
    peak_allocated_mb,
    peak_reserved_mb,
    source,
):
    return {
        "task": task,
        "dataset": dataset,
        "unit": unit,
        "method": method,
        "num_frames": num_frames,
        "primary_metric": primary_metric,
        "primary_direction": primary_direction,
        "primary_value": primary_value,
        "secondary_metric": secondary_metric,
        "secondary_direction": secondary_direction,
        "secondary_value": secondary_value,
        "inference_sec": inference_sec,
        "peak_allocated_mb": peak_allocated_mb,
        "peak_reserved_mb": peak_reserved_mb,
        "source": source,
    }


def load_video_depth(path):
    rows = read_csv(path)
    units = []
    for row in rows:
        method = row["method"]
        if method not in METHODS:
            continue
        units.append(
            quality_unit(
                "video_depth",
                row["dataset"],
                row["sequence"],
                method,
                as_int(row["num_frames"]),
                "abs_rel",
                "lower",
                float(row["abs_rel"]),
                "delta_1",
                "higher",
                float(row["delta_1"]),
                float(row["inference_sec"]),
                float(row["peak_allocated_mb"]),
                float(row["peak_reserved_mb"]),
                path,
            )
        )
    return units


def load_pose(baseline_path, k8_path):
    units = []
    for path in (baseline_path, k8_path):
        for row in read_csv(path):
            method = normalize_method(row)
            if method not in METHODS:
                continue
            dataset = row["dataset"]
            if dataset not in EXPECTED_DATASETS["pose"]:
                continue
            expected = {
                "scannet": 6,
                "sintel": 14,
                "tum": 8,
            }[dataset]
            if (
                as_int(row["num_successful"]) != expected
                or as_int(row["num_failed"]) != 0
            ):
                raise RuntimeError(
                    f"incomplete pose result: {dataset}/{method}"
                )
            units.append(
                quality_unit(
                    "pose",
                    dataset,
                    "dataset_aggregate",
                    method,
                    as_int(row["total_frames"]),
                    "ate",
                    "lower",
                    float(row["mean_ate"]),
                    "rpe_rot_deg",
                    "lower",
                    float(row["mean_rpe_rot_deg"]),
                    float(row["total_inference_sec"]),
                    float(row["max_peak_allocated_mb"]),
                    float(row["max_peak_reserved_mb"]),
                    path,
                )
            )
    return units


def recon_method(payload):
    return normalize_method(
        {
            "cache_policy": payload.get("cache_policy", ""),
            "cache_window_size": payload.get("cache_window_size", ""),
        }
    )


def load_reconstruction(archive_path):
    payloads = {}
    sources = {}
    with tarfile.open(archive_path, "r:gz") as archive:
        for member in archive.getmembers():
            if not member.isfile() or not member.name.endswith(
                "reconstruction_metrics.json"
            ):
                continue
            payload = json.load(archive.extractfile(member))
            method = recon_method(payload)
            if method not in METHODS:
                continue
            protocol = payload.get("protocol")
            key = (protocol, method)
            if key in payloads:
                raise RuntimeError(f"duplicate reconstruction payload: {key}")
            payloads[key] = payload
            sources[key] = f"{archive_path}:{member.name}"

    expected_payloads = {
        (protocol, method)
        for protocol in ("dense", "paper")
        for method in METHODS
    }
    if set(payloads) != expected_payloads:
        raise RuntimeError(
            f"reconstruction archive mismatch: "
            f"missing={sorted(expected_payloads-set(payloads))}, "
            f"unexpected={sorted(set(payloads)-expected_payloads)}"
        )

    units = []
    task_protocols = {
        "static_recon": ("dense", ("7scenes", "nrgbd", "eth3d")),
        "dynamic_recon": ("paper", ("tum",)),
    }
    for task, (protocol, datasets) in task_protocols.items():
        for dataset in datasets:
            sequences_by_method = {}
            for method in METHODS:
                dataset_payload = payloads[(protocol, method)]["datasets"][
                    dataset
                ]
                sequences_by_method[method] = {
                    item["sequence"]: item
                    for item in dataset_payload["sequences"]
                    if item.get("status") == "ok"
                }
            common = set.intersection(
                *(set(items) for items in sequences_by_method.values())
            )
            expected_count = EXPECTED_DATASETS[task][dataset]
            if len(common) != expected_count:
                raise RuntimeError(
                    f"{task}/{dataset}: expected {expected_count} common "
                    f"sequences, found {len(common)}"
                )
            for sequence in sorted(common):
                frame_counts = {
                    int(sequences_by_method[method][sequence]["num_frames"])
                    for method in METHODS
                }
                if len(frame_counts) != 1:
                    raise RuntimeError(
                        f"{task}/{dataset}/{sequence}: frame mismatch"
                    )
                for method in METHODS:
                    item = sequences_by_method[method][sequence]
                    units.append(
                        quality_unit(
                            task,
                            dataset,
                            sequence,
                            method,
                            int(item["num_frames"]),
                            "overall",
                            "lower",
                            float(item["overall"]),
                            "nc",
                            "higher",
                            float(item["nc"]),
                            float(item["inference_sec"]),
                            float(item["peak_allocated_mb"]),
                            float(item["peak_reserved_mb"]),
                            sources[(protocol, method)],
                        )
                    )
    return units


def validate_quality_units(units):
    indexed = defaultdict(list)
    for unit in units:
        indexed[(unit["task"], unit["dataset"], unit["method"])].append(unit)

    expected_keys = {
        (task, dataset, method)
        for task, datasets in EXPECTED_DATASETS.items()
        for dataset in datasets
        for method in METHODS
    }
    if set(indexed) != expected_keys:
        raise RuntimeError(
            f"cross-task matrix mismatch: "
            f"missing={sorted(expected_keys-set(indexed))}, "
            f"unexpected={sorted(set(indexed)-expected_keys)}"
        )

    for task, datasets in EXPECTED_DATASETS.items():
        for dataset, expected_count in datasets.items():
            reference = {
                item["unit"]
                for item in indexed[(task, dataset, "full_cache")]
            }
            if len(reference) != expected_count:
                raise RuntimeError(
                    f"{task}/{dataset}: expected {expected_count} units, "
                    f"found {len(reference)}"
                )
            for method in METHODS:
                method_units = {
                    item["unit"]
                    for item in indexed[(task, dataset, method)]
                }
                if method_units != reference:
                    raise RuntimeError(
                        f"{task}/{dataset}: coverage mismatch for {method}"
                    )
            for unit_name in reference:
                frame_counts = {
                    item["num_frames"]
                    for item in units
                    if item["task"] == task
                    and item["dataset"] == dataset
                    and item["unit"] == unit_name
                }
                if len(frame_counts) != 1:
                    raise RuntimeError(
                        f"{task}/{dataset}/{unit_name}: frame mismatch"
                    )


def load_video_depth_aggregates(pareto_path):
    rows = {}
    for row in read_csv(pareto_path):
        key = (row["dataset"], row["method"])
        rows[key] = {
            "primary_value": float(row["abs_rel"]),
            "secondary_value": float(row["delta_1"]),
            "total_inference_sec": float(row["total_inference_sec"]),
            "fps_inference": float(row["fps_inference"]),
            "max_peak_allocated_mb": float(row["max_peak_allocated_mb"]),
            "max_peak_reserved_mb": float(row["max_peak_reserved_mb"]),
        }
    expected = {
        (dataset, method)
        for dataset in EXPECTED_DATASETS["video_depth"]
        for method in METHODS
    }
    if set(rows) != expected:
        raise RuntimeError("Stage 4B VideoDepth Pareto coverage mismatch")
    return rows


def build_summary(units, video_depth_aggregates, long_results, long_gate):
    grouped = defaultdict(list)
    for unit in units:
        grouped[(unit["task"], unit["dataset"], unit["method"])].append(unit)

    outputs = []
    for key in sorted(
        grouped,
        key=lambda item: (
            list(EXPECTED_DATASETS).index(item[0]),
            item[1],
            METHOD_ORDER[item[2]],
        ),
    ):
        task, dataset, method = key
        items = grouped[key]
        total_frames = sum(item["num_frames"] for item in items)
        total_inference = sum(item["inference_sec"] for item in items)
        row = {
            "task": task,
            "dataset": dataset,
            "method": method,
            "evaluation_unit": (
                "dataset_aggregate" if task == "pose" else "sequence"
            ),
            "num_units": len(items),
            "total_frames": total_frames,
            "primary_metric": items[0]["primary_metric"],
            "primary_direction": items[0]["primary_direction"],
            "primary_value": statistics.fmean(
                item["primary_value"] for item in items
            ),
            "secondary_metric": items[0]["secondary_metric"],
            "secondary_direction": items[0]["secondary_direction"],
            "secondary_value": statistics.fmean(
                item["secondary_value"] for item in items
            ),
            "total_inference_sec": total_inference,
            "fps_inference": total_frames / total_inference,
            "max_peak_allocated_mb": max(
                item["peak_allocated_mb"] for item in items
            ),
            "max_peak_reserved_mb": max(
                item["peak_reserved_mb"] for item in items
            ),
            "coverage_ok": "yes",
            "source": " ".join(sorted({item["source"] for item in items})),
        }
        if task == "video_depth":
            row.update(video_depth_aggregates[(dataset, method)])
        outputs.append(row)

    gate_rows = read_csv(long_gate)
    if len(gate_rows) != 1 or gate_rows[0].get("decision") != "PASS":
        raise RuntimeError("Stage 3.6B long-sequence platform gate did not pass")
    gate = gate_rows[0]
    result_rows = read_csv(long_results)
    long_1000 = next(
        row
        for row in result_rows
        if row["method"] == "7scenes_stream_1000"
    )
    outputs.append(
        {
            "task": "long_sequence_platform",
            "dataset": "7scenes_raw_500_to_1000",
            "method": "temporal_binned_dino_k8",
            "evaluation_unit": "length_scaling",
            "num_units": 2,
            "total_frames": 1000,
            "primary_metric": "gpu_peak_1000_minus_500_mb",
            "primary_direction": "lower",
            "primary_value": float(gate["gpu_peak_1000_minus_500_mb"]),
            "secondary_metric": "rss_peak_1000_minus_500_mib",
            "secondary_direction": "lower",
            "secondary_value": float(gate["rss_peak_1000_minus_500_mib"]),
            "total_inference_sec": float(long_1000["inference_sec"]),
            "fps_inference": float(long_1000["fps_inference"]),
            "max_peak_allocated_mb": float(
                long_1000["peak_allocated_mb"]
            ),
            "max_peak_reserved_mb": float(long_1000["peak_reserved_mb"]),
            "coverage_ok": "yes",
            "source": f"{long_results} {long_gate}",
        }
    )
    return outputs


def regret(value, oracle, direction):
    denominator = max(abs(oracle), 1e-12)
    if direction == "lower":
        return (value - oracle) / denominator
    return (oracle - value) / denominator


def regret_rows_for_items(task, dataset, evaluation_unit, items):
    indexed = {
        (item["unit"], item["method"]): item for item in items
    }
    unit_names = sorted({item["unit"] for item in items})
    outputs = []
    for metric_kind in ("primary", "secondary"):
        metric_key = f"{metric_kind}_metric"
        direction_key = f"{metric_kind}_direction"
        value_key = f"{metric_kind}_value"
        metric_names = {item[metric_key] for item in items}
        directions = {item[direction_key] for item in items}
        if len(metric_names) != 1 or len(directions) != 1:
            raise RuntimeError(
                f"incompatible regret metrics for {task}/{dataset}"
            )
        metric = next(iter(metric_names))
        direction = next(iter(directions))
        method_regrets = {method: [] for method in BOUNDED_METHODS}
        method_wins = {method: 0 for method in BOUNDED_METHODS}
        for unit_name in unit_names:
            values = {
                method: indexed[(unit_name, method)][value_key]
                for method in BOUNDED_METHODS
            }
            oracle = (
                min(values.values())
                if direction == "lower"
                else max(values.values())
            )
            for method, value in values.items():
                method_regrets[method].append(
                    regret(value, oracle, direction)
                )
                if close(value, oracle):
                    method_wins[method] += 1
        for method in BOUNDED_METHODS:
            values = method_regrets[method]
            outputs.append(
                {
                    "task": task,
                    "dataset": dataset,
                    "evaluation_unit": evaluation_unit,
                    "metric": metric,
                    "direction": direction,
                    "oracle_scope": "bounded_only",
                    "method": method,
                    "n_units": len(values),
                    "mean_normalized_regret": statistics.fmean(values),
                    "median_normalized_regret": statistics.median(values),
                    "max_normalized_regret": max(values),
                    "oracle_wins": method_wins[method],
                }
            )
    return outputs


def build_regret(units, summary_rows):
    outputs = []
    for task, datasets in EXPECTED_DATASETS.items():
        task_items = [item for item in units if item["task"] == task]
        for dataset in datasets:
            items = [
                item for item in task_items if item["dataset"] == dataset
            ]
            outputs.extend(
                regret_rows_for_items(
                    task,
                    dataset,
                    "dataset_aggregate" if task == "pose" else "sequence",
                    items,
                )
            )
        if len(datasets) > 1:
            combined = []
            for item in task_items:
                combined_item = dict(item)
                combined_item["unit"] = (
                    f"{item['dataset']}::{item['unit']}"
                )
                combined.append(combined_item)
            outputs.extend(
                regret_rows_for_items(
                    task,
                    "all",
                    "mixed_units",
                    combined,
                )
            )

    core_summaries = [
        row
        for row in summary_rows
        if row["task"] in EXPECTED_DATASETS
    ]
    benchmark_keys = sorted(
        {
            (row["task"], row["dataset"])
            for row in core_summaries
        }
    )
    values_by_method = {method: [] for method in BOUNDED_METHODS}
    wins_by_method = {method: 0 for method in BOUNDED_METHODS}
    for task, dataset in benchmark_keys:
        values = {
            method: float(
                next(
                    row["primary_value"]
                    for row in core_summaries
                    if row["task"] == task
                    and row["dataset"] == dataset
                    and row["method"] == method
                )
            )
            for method in BOUNDED_METHODS
        }
        direction = next(
            row["primary_direction"]
            for row in core_summaries
            if row["task"] == task and row["dataset"] == dataset
        )
        oracle = (
            min(values.values())
            if direction == "lower"
            else max(values.values())
        )
        for method, value in values.items():
            values_by_method[method].append(
                regret(value, oracle, direction)
            )
            if close(value, oracle):
                wins_by_method[method] += 1
    for method in BOUNDED_METHODS:
        values = values_by_method[method]
        outputs.append(
            {
                "task": "cross_task_macro",
                "dataset": "all",
                "evaluation_unit": "benchmark_dataset",
                "metric": "primary_quality",
                "direction": "lower_regret",
                "oracle_scope": "bounded_only",
                "method": method,
                "n_units": len(values),
                "mean_normalized_regret": statistics.fmean(values),
                "median_normalized_regret": statistics.median(values),
                "max_normalized_regret": max(values),
                "oracle_wins": wins_by_method[method],
            }
        )
    return outputs


def build_roles(
    summary_rows,
    regret_rows,
    pareto_path,
    long_gate,
    stage4a_gate,
):
    macro = {
        row["method"]: row
        for row in regret_rows
        if row["task"] == "cross_task_macro"
    }
    stage4a_rows = {
        row["candidate"]: row for row in read_csv(stage4a_gate)
    }
    missing_gates = set(BOUNDED_METHODS) - set(stage4a_rows)
    if missing_gates:
        raise ValueError(
            f"Stage 4A gate is missing candidates: {sorted(missing_gates)}"
        )
    eligible = [
        method
        for method in BOUNDED_METHODS
        if stage4a_rows[method].get("eligible_for_stage4b") == "yes"
        and stage4a_rows[method].get("decision") == "PASS"
    ]
    if not eligible:
        raise ValueError("Stage 4A has no eligible bounded candidate")

    # Role selection is deliberately gate-first. Cross-task normalized regret
    # mixes heterogeneous metrics and is descriptive; it must not promote a
    # candidate that failed a pre-registered gate. Among eligible candidates,
    # select the primary by benchmark-level oracle wins, with mean regret only
    # as a deterministic tie-breaker.
    primary = max(
        eligible,
        key=lambda method: (
            int(macro[method]["oracle_wins"]),
            -float(macro[method]["mean_normalized_regret"]),
        ),
    )
    robust_candidates = [
        method for method in eligible if method != primary
    ]
    robust = (
        min(
            robust_candidates,
            key=lambda method: float(
                next(
                    row["mean_normalized_regret"]
                    for row in regret_rows
                    if row["task"] == "static_recon"
                    and row["dataset"] == "all"
                    and row["method"] == method
                )
            ),
        )
        if robust_candidates
        else None
    )
    long_pass = (
        len(read_csv(long_gate)) == 1
        and read_csv(long_gate)[0].get("decision") == "PASS"
    )
    pareto_rows = read_csv(pareto_path)
    core_summaries = [
        row for row in summary_rows if row["task"] in EXPECTED_DATASETS
    ]

    task_regret = {}
    for row in regret_rows:
        if (
            row["dataset"] == "all"
            and row["metric"]
            in {"abs_rel", "ate", "overall"}
            and row["task"] in EXPECTED_DATASETS
        ):
            task_regret[(row["task"], row["method"])] = row[
                "mean_normalized_regret"
            ]
        elif (
            row["task"] == "dynamic_recon"
            and row["dataset"] == "tum"
            and row["metric"] == "overall"
        ):
            task_regret[(row["task"], row["method"])] = row[
                "mean_normalized_regret"
            ]

    outputs = []
    for method in METHODS:
        max_allocated = max(
            float(row["max_peak_allocated_mb"])
            for row in core_summaries
            if row["method"] == method
        )
        pareto_count = sum(
            row["method"] == method
            and row["pareto_absrel_allocated_time"] == "yes"
            for row in pareto_rows
        )
        if method == "full_cache":
            outputs.append(
                {
                    "method": method,
                    "bounded": "no",
                    "stage4a_eligible": "n/a",
                    "num_benchmarks": len(EXPECTED_DATASETS["video_depth"])
                    + len(EXPECTED_DATASETS["pose"])
                    + len(EXPECTED_DATASETS["static_recon"])
                    + len(EXPECTED_DATASETS["dynamic_recon"]),
                    "primary_oracle_wins": "",
                    "mean_macro_primary_regret": "",
                    "median_macro_primary_regret": "",
                    "max_macro_primary_regret": "",
                    "video_depth_mean_regret": "",
                    "pose_mean_regret": "",
                    "static_recon_mean_regret": "",
                    "dynamic_recon_mean_regret": "",
                    "video_depth_pareto_datasets": pareto_count,
                    "max_peak_allocated_mb": max_allocated,
                    "final_role": "quality_resource_reference",
                    "status": "REFERENCE_ONLY",
                    "rationale": (
                        "Unbounded cache is retained as a quality/resource "
                        "reference, not a deployment candidate."
                    ),
                }
            )
            continue

        m = macro[method]
        role = "secondary_bounded"
        status = "FROZEN_SECONDARY"
        rationale = "Retained for complete bounded-method reporting."
        if method == primary:
            role = "primary_bounded_deployment"
            status = "FROZEN_PRIMARY"
            rationale = (
                f"Stage 4A eligible and wins {m['oracle_wins']}/10 "
                "bounded-only benchmark primary oracles, the largest count "
                "among eligible candidates."
            )
        elif method == robust:
            role = "robust_bounded_alternative"
            status = "FROZEN_ROBUST"
            rationale = (
                "Stage 4A eligible; retained as the reconstruction/tail-risk "
                "alternative after the primary was frozen."
            )
        elif (
            method == "temporal_binned_dino_k8"
            and stage4a_rows[method].get("eligible_for_stage4b") != "yes"
            and long_pass
        ):
            role = "long_sequence_pose_specialist"
            status = "FROZEN_SPECIALIST"
            rationale = (
                "Passed the 1000-frame streaming plateau gate but failed "
                "the unified VideoDepth/reconstruction gate."
            )
        outputs.append(
            {
                "method": method,
                "bounded": "yes",
                "stage4a_eligible": stage4a_rows[method].get(
                    "eligible_for_stage4b", ""
                ),
                "num_benchmarks": m["n_units"],
                "primary_oracle_wins": m["oracle_wins"],
                "mean_macro_primary_regret": m[
                    "mean_normalized_regret"
                ],
                "median_macro_primary_regret": m[
                    "median_normalized_regret"
                ],
                "max_macro_primary_regret": m[
                    "max_normalized_regret"
                ],
                "video_depth_mean_regret": task_regret.get(
                    ("video_depth", method), ""
                ),
                "pose_mean_regret": task_regret.get(("pose", method), ""),
                "static_recon_mean_regret": task_regret.get(
                    ("static_recon", method), ""
                ),
                "dynamic_recon_mean_regret": task_regret.get(
                    ("dynamic_recon", method), ""
                ),
                "video_depth_pareto_datasets": pareto_count,
                "max_peak_allocated_mb": max_allocated,
                "final_role": role,
                "status": status,
                "rationale": rationale,
            }
        )
    return outputs


def k4_vs_full_significance(paired_rows, dataset, metric):
    row = next(
        item
        for item in paired_rows
        if item["dataset"] == dataset
        and item["metric"] == metric
        and {item["method_a"], item["method_b"]}
        == {"stage3_2_k4", "full_cache"}
    )
    significance = row["significance"]
    if row["method_a"] == "stage3_2_k4":
        return significance
    return {
        "A_BETTER": "B_BETTER",
        "B_BETTER": "A_BETTER",
        "NO_CLEAR_DIFFERENCE": "NO_CLEAR_DIFFERENCE",
    }[significance]


def build_claims(summary_rows, role_rows, paired_path, pareto_path, long_gate):
    core = [row for row in summary_rows if row["task"] in EXPECTED_DATASETS]
    coverage_ok = len(core) == 40 and all(
        row["coverage_ok"] == "yes" for row in core
    )
    paired = read_csv(paired_path)
    pareto = read_csv(pareto_path)
    k4_pareto = sum(
        row["method"] == "stage3_2_k4"
        and row["pareto_absrel_allocated_time"] == "yes"
        for row in pareto
    )
    kitti_abs = k4_vs_full_significance(paired, "kitti", "abs_rel")
    bonn_abs = k4_vs_full_significance(paired, "bonn", "abs_rel")
    sintel_abs = k4_vs_full_significance(paired, "sintel", "abs_rel")
    k4_resource_ok = all(
        next(
            float(row["max_peak_allocated_mb"])
            for row in core
            if row["task"] == task
            and row["dataset"] == dataset
            and row["method"] == "stage3_2_k4"
        )
        < next(
            float(row["max_peak_allocated_mb"])
            for row in core
            if row["task"] == task
            and row["dataset"] == dataset
            and row["method"] == "full_cache"
        )
        for task, datasets in EXPECTED_DATASETS.items()
        for dataset in datasets
    )
    roles = {row["method"]: row for row in role_rows}
    bounded_macro = {
        method: float(roles[method]["mean_macro_primary_regret"])
        for method in BOUNDED_METHODS
    }
    long_pass = (
        len(read_csv(long_gate)) == 1
        and read_csv(long_gate)[0].get("decision") == "PASS"
    )

    return [
        {
            "claim_id": "coverage_complete",
            "status": "PASS" if coverage_ok else "FAIL",
            "claim": "The frozen four-method cross-task matrix is complete.",
            "evidence": (
                f"{len(core)}/40 task-dataset-method aggregate cells; "
                "VideoDepth and reconstruction retain paired sequence coverage."
            ),
            "allowed_wording": (
                "All frozen methods were evaluated on identical coverage "
                "within each benchmark."
            ),
            "forbidden_wording": "Do not compare incomplete method means.",
            "next_action": "Proceed only if PASS.",
        },
        {
            "claim_id": "k4_video_depth_default",
            "status": (
                "PASS"
                if k4_pareto == 3
                and kitti_abs == "A_BETTER"
                and bonn_abs != "B_BETTER"
                and sintel_abs != "B_BETTER"
                else "FAIL"
            ),
            "claim": "K4 is the frozen VideoDepth/compact default.",
            "evidence": (
                f"K4 Pareto datasets={k4_pareto}/3; paired AbsRel "
                f"KITTI={kitti_abs}, Bonn={bonn_abs}, Sintel={sintel_abs}."
            ),
            "allowed_wording": (
                "K4 significantly improves KITTI and maintains Bonn/Sintel "
                "AbsRel while reducing memory."
            ),
            "forbidden_wording": (
                "Do not claim significant quality improvement on Bonn or "
                "Sintel."
            ),
            "next_action": "Use K4 as the primary bounded VideoDepth row.",
        },
        {
            "claim_id": "k4_cross_task_primary",
            "status": (
                "PASS"
                if roles["stage3_2_k4"]["status"] == "FROZEN_PRIMARY"
                else "FAIL"
            ),
            "claim": "K4 is the primary bounded deployment configuration.",
            "evidence": roles["stage3_2_k4"]["rationale"],
            "allowed_wording": (
                "K4 wins the largest number of benchmark primary oracles "
                "among Stage 4A-eligible bounded candidates."
            ),
            "forbidden_wording": (
                "Do not call K4 the winner on every sequence or every pose "
                "metric."
            ),
            "next_action": "Carry K4 into Stage 4C.",
        },
        {
            "claim_id": "old_k6_robust_alternative",
            "status": (
                "PASS"
                if roles["old_dino_k6"]["status"] == "FROZEN_ROBUST"
                else "FAIL"
            ),
            "claim": "Old DINO K6 is the frozen robust bounded alternative.",
            "evidence": roles["old_dino_k6"]["rationale"],
            "allowed_wording": (
                "Old K6 is retained for lower tail risk on reconstruction-"
                "heavy workloads."
            ),
            "forbidden_wording": (
                "Do not present old K6 as the VideoDepth Pareto winner."
            ),
            "next_action": "Carry old K6 into Stage 4C.",
        },
        {
            "claim_id": "temporal_k8_specialist",
            "status": (
                "PASS_LIMITED"
                if roles["temporal_binned_dino_k8"]["status"]
                == "FROZEN_SPECIALIST"
                and long_pass
                else "FAIL"
            ),
            "claim": "Temporal K8 is a long-sequence/pose specialist only.",
            "evidence": (
                "Stage 3.6B plateau gate PASS; Stage 3.7 geometry and Stage "
                "4A KITTI VideoDepth gates failed."
            ),
            "allowed_wording": (
                "Temporal K8 demonstrates bounded 1000-frame streaming and "
                "strong selected pose cases."
            ),
            "forbidden_wording": (
                "Do not present temporal K8 as a unified default."
            ),
            "next_action": "Carry K8 into Stage 4C as a specialist.",
        },
        {
            "claim_id": "macro_regret_interpretation",
            "status": "LIMITED",
            "claim": (
                "Cross-task macro normalized regret is descriptive and does "
                "not override pre-registered eligibility gates."
            ),
            "evidence": (
                "Naive macro mean regrets: "
                + ", ".join(
                    f"{method}={bounded_macro[method]:.6f}"
                    for method in BOUNDED_METHODS
                )
                + "; temporal K8 nevertheless failed the Stage 4A gate."
            ),
            "allowed_wording": (
                "Report macro regret as a heterogeneous cross-task risk "
                "summary alongside task-specific gates."
            ),
            "forbidden_wording": (
                "Do not choose the unified default solely from the lowest "
                "post-hoc macro mean."
            ),
            "next_action": "Keep the frozen gate-first method roles.",
        },
        {
            "claim_id": "dino_causal_attribution",
            "status": "LIMITED",
            "claim": "The final results establish an integrated DINO-guided system.",
            "evidence": (
                "Historical uniform-K6 evidence exists, but no same-K "
                "non-DINO control exists for K4 or temporal K8."
            ),
            "allowed_wording": (
                "The integrated DINO-guided bounded-cache configurations "
                "achieve the reported trade-offs."
            ),
            "forbidden_wording": (
                "Do not attribute every K4/K8 gain solely to DINO."
            ),
            "next_action": "No new attribution experiment; preserve limitation.",
        },
        {
            "claim_id": "stage4c_ready",
            "status": (
                "PASS"
                if coverage_ok
                and k4_resource_ok
                and long_pass
                else "FAIL"
            ),
            "claim": "Frozen candidates are ready for held-out long-sequence validation.",
            "evidence": (
                f"coverage={coverage_ok}, K4 allocated below full on all "
                f"10 benchmarks={k4_resource_ok}, Stage3.6B={long_pass}."
            ),
            "allowed_wording": "Proceed to Stage 4C without selector tuning.",
            "forbidden_wording": "Do not reopen K/bank/DINO threshold search.",
            "next_action": "Implement Stage 4C held-out real-sequence protocol.",
        },
    ]


def main():
    parser = argparse.ArgumentParser("Summarize Stage 4B cross-task evidence")
    parser.add_argument(
        "--video-depth-sequences",
        default="stage4b_video_depth_sequence_results.csv",
    )
    parser.add_argument(
        "--video-depth-paired",
        default="stage4b_video_depth_paired_comparison.csv",
    )
    parser.add_argument(
        "--video-depth-pareto", default="stage4b_pareto.csv"
    )
    parser.add_argument("--pose-baseline", default="stage3_3_pose_results.csv")
    parser.add_argument("--pose-k8", default="stage3_7_pose_results.csv")
    parser.add_argument(
        "--recon-archive", default="stage3_7_sequence_metrics.tar.gz"
    )
    parser.add_argument("--long-results", default="stage3_6b_results.csv")
    parser.add_argument("--long-gate", default="stage3_6b_gate.csv")
    parser.add_argument("--stage4a-gate", default="stage4a_gate.csv")
    parser.add_argument(
        "--summary-output", default="stage4b_cross_task_summary.csv"
    )
    parser.add_argument(
        "--regret-output", default="stage4b_cross_task_regret.csv"
    )
    parser.add_argument(
        "--roles-output", default="stage4b_method_roles.csv"
    )
    parser.add_argument(
        "--claims-output", default="stage4b_claim_audit.csv"
    )
    args = parser.parse_args()

    units = []
    units.extend(load_video_depth(args.video_depth_sequences))
    units.extend(load_pose(args.pose_baseline, args.pose_k8))
    units.extend(load_reconstruction(args.recon_archive))
    validate_quality_units(units)

    vd_aggregates = load_video_depth_aggregates(
        args.video_depth_pareto
    )
    summary_rows = build_summary(
        units, vd_aggregates, args.long_results, args.long_gate
    )
    regret_rows = build_regret(units, summary_rows)
    role_rows = build_roles(
        summary_rows,
        regret_rows,
        args.video_depth_pareto,
        args.long_gate,
        args.stage4a_gate,
    )
    claim_rows = build_claims(
        summary_rows,
        role_rows,
        args.video_depth_paired,
        args.video_depth_pareto,
        args.long_gate,
    )

    write_csv(args.summary_output, SUMMARY_FIELDS, summary_rows)
    write_csv(args.regret_output, REGRET_FIELDS, regret_rows)
    write_csv(args.roles_output, ROLE_FIELDS, role_rows)
    write_csv(args.claims_output, CLAIM_FIELDS, claim_rows)


if __name__ == "__main__":
    main()
