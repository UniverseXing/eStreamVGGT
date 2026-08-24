#!/usr/bin/env python3
"""Summarize the conference Stage 5A same-budget VideoDepth experiment."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))
from summarize_stage6a import (  # noqa: E402
    COMMON_FIELDS,
    SEQUENCE_FIELDS,
    load_video_depth,
    write_csv,
)

CORE_METHODS = ("full_cache", "recent4", "anchor_recent4", "proposed_k4")
OPTIONAL_METHODS = (
    "anchor_uniform4",
    "random4_seed0",
    "random4_seed1",
    "random4_seed2",
)
DATASETS = {"bonn": 5, "sintel": 23, "kitti": 13}
PAIRED_FIELDS = (
    "dataset", "metric", "proposed", "control", "n_pairs",
    "mean_proposed", "mean_control", "mean_advantage_proposed",
    "ci95_low", "ci95_high", "wins_proposed", "ties", "losses_proposed",
    "bootstrap_samples",
)


def read_csv(path):
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def load_frozen_references(root):
    aggregate_candidates = (
        root / "stage4a_video_depth_results(1).csv",
        root / "stage4a_video_depth_results.csv",
    )
    aggregate_path = next(
        (path for path in aggregate_candidates if path.is_file()),
        aggregate_candidates[0],
    )
    sequence_path = root / "stage4b_video_depth_sequence_results.csv"
    if not aggregate_path.is_file() or not sequence_path.is_file():
        raise FileNotFoundError(
            "Stage 5A reference reuse requires stage4a_video_depth_results(1).csv "
            "and stage4b_video_depth_sequence_results.csv"
        )
    aggregate_source = read_csv(aggregate_path)
    sequence_source = read_csv(sequence_path)
    method_map = {"full_cache": "full_cache", "stage3_2_k4": "proposed_k4"}
    aggregates, sequences = [], []
    for source in aggregate_source:
        if source["method"] not in method_map:
            continue
        method = method_map[source["method"]]
        row = {field: "" for field in COMMON_FIELDS}
        row.update(
            task="video_depth", dataset=source["dataset"], method=method,
            random_seed=0,
            cache_policy=(
                "full_cache" if method == "full_cache"
                else "anchor_recent_dino_diverse_k4"
            ),
            cache_window_size=("" if method == "full_cache" else 4),
            num_sequences=source["num_sequences"],
            num_successful=source["num_ok"], num_failed=source["num_oom"],
            total_frames=source["total_frames"], abs_rel=source["abs_rel"],
            rmse=source["rmse"], delta_1=source["delta_1"],
            fps_inference=source["fps_inference"],
            peak_allocated_mb=source["max_peak_allocated_mb"],
            peak_reserved_mb=source["max_peak_reserved_mb"],
            gpu_name=source["gpu_name"], torch_version=source["torch_version"],
            cuda_version=source["cuda_version"], slurm_job_id=source["slurm_job_id"],
            source=str(aggregate_path),
        )
        aggregates.append(row)
    for source in sequence_source:
        if source["method"] not in method_map:
            continue
        method = method_map[source["method"]]
        row = {field: "" for field in SEQUENCE_FIELDS}
        row.update(
            task="video_depth", dataset=source["dataset"],
            sequence=source["sequence"], method=method, random_seed=0,
            num_frames=source["num_frames"], abs_rel=source["abs_rel"],
            rmse=source["rmse"], delta_1=source["delta_1"],
            inference_sec=source["inference_sec"],
            fps_inference=source["fps_inference"],
            peak_allocated_mb=source["peak_allocated_mb"],
            peak_reserved_mb=source["peak_reserved_mb"],
            source=str(sequence_path),
        )
        sequences.append(row)
    expected = len(DATASETS) * 2
    if len(aggregates) != expected:
        raise RuntimeError(f"frozen Stage 4 reference aggregate is incomplete: {len(aggregates)}/{expected}")
    return aggregates, sequences


def method_values(rows, dataset, method, metric):
    selected = [row for row in rows if row["dataset"] == dataset]
    if method == "random4_mean":
        random_names = {name for name in OPTIONAL_METHODS if name.startswith("random4_seed")}
        grouped = {}
        for row in selected:
            if row["method"] in random_names:
                grouped.setdefault(row["sequence"], []).append(float(row[metric]))
        if not grouped or any(len(values) != 3 for values in grouped.values()):
            return {}
        return {name: float(np.mean(values)) for name, values in grouped.items()}
    return {
        row["sequence"]: float(row[metric])
        for row in selected
        if row["method"] == method
    }


def paired_statistics(rows, samples):
    rng = np.random.default_rng(20260824)
    outputs = []
    controls = ["recent4", "anchor_recent4"]
    if any(row["method"] == "anchor_uniform4" for row in rows):
        controls.append("anchor_uniform4")
    if any(row["method"].startswith("random4_seed") for row in rows):
        controls.append("random4_mean")
    for dataset in DATASETS:
        proposed = method_values(rows, dataset, "proposed_k4", "abs_rel")
        for control in controls:
            baseline = method_values(rows, dataset, control, "abs_rel")
            if not proposed or set(proposed) != set(baseline):
                continue
            names = sorted(proposed)
            a = np.asarray([proposed[name] for name in names])
            b = np.asarray([baseline[name] for name in names])
            advantage = b - a
            indices = rng.integers(0, len(names), size=(samples, len(names)))
            bootstrap = advantage[indices].mean(axis=1)
            outputs.append({
                "dataset": dataset, "metric": "abs_rel",
                "proposed": "proposed_k4", "control": control,
                "n_pairs": len(names), "mean_proposed": float(a.mean()),
                "mean_control": float(b.mean()),
                "mean_advantage_proposed": float(advantage.mean()),
                "ci95_low": float(np.percentile(bootstrap, 2.5)),
                "ci95_high": float(np.percentile(bootstrap, 97.5)),
                "wins_proposed": int(np.sum(advantage > 1e-12)),
                "ties": int(np.sum(np.abs(advantage) <= 1e-12)),
                "losses_proposed": int(np.sum(advantage < -1e-12)),
                "bootstrap_samples": samples,
            })
    return outputs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--include-optional", action="store_true")
    parser.add_argument("--rerun-references", action="store_true")
    args = parser.parse_args()
    root = args.repo_root.resolve()
    methods_to_load = ["recent4", "anchor_recent4"]
    if args.rerun_references:
        methods_to_load = list(CORE_METHODS)
    if args.include_optional:
        methods_to_load.extend(OPTIONAL_METHODS)
    aggregate_rows, sequence_rows = (
        ([], []) if args.rerun_references else load_frozen_references(root)
    )
    for method in methods_to_load:
        for dataset, expected_count in DATASETS.items():
            aggregate, sequences = load_video_depth(
                root, dataset, method, False, stage_tag="stage5a"
            )
            if int(aggregate["num_successful"]) != expected_count or int(aggregate["num_failed"]) != 0:
                raise RuntimeError(f"incomplete Stage 5A result: {dataset}/{method}")
            aggregate_rows.append(aggregate)
            sequence_rows.extend(sequences)
    methods = (*CORE_METHODS, *(OPTIONAL_METHODS if args.include_optional else ()))
    for dataset, expected_count in DATASETS.items():
        reference = None
        for method in methods:
            signature = {
                (row["sequence"], int(row["num_frames"]))
                for row in sequence_rows
                if row["dataset"] == dataset and row["method"] == method
            }
            if len(signature) != expected_count:
                raise RuntimeError(f"wrong sequence count for {dataset}/{method}")
            if reference is None:
                reference = signature
            elif signature != reference:
                raise RuntimeError(f"coverage mismatch for {dataset}/{method}")
    gpu_names = {row["gpu_name"] for row in aggregate_rows}
    if len(gpu_names) != 1 or "6000 ada" not in next(iter(gpu_names)).lower():
        raise RuntimeError(f"formal Stage 5A requires one RTX 6000 Ada model: {gpu_names}")
    for field in ("torch_version", "cuda_version"):
        values = {str(row[field]) for row in aggregate_rows}
        if len(values) != 1:
            raise RuntimeError(f"Stage 5A mixes {field}: {sorted(values)}")
    order = {method: index for index, method in enumerate(methods)}
    aggregate_rows.sort(key=lambda row: (row["dataset"], order[row["method"]]))
    sequence_rows.sort(key=lambda row: (row["dataset"], row["sequence"], order[row["method"]]))
    write_csv(root / "stage5a_same_budget_results.csv", COMMON_FIELDS, aggregate_rows)
    write_csv(root / "stage5a_same_budget_sequence_results.csv", SEQUENCE_FIELDS, sequence_rows)
    paired = paired_statistics(sequence_rows, args.bootstrap_samples)
    with (root / "stage5a_paired_statistics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=PAIRED_FIELDS)
        writer.writeheader(); writer.writerows(paired)
    print(f"Wrote {len(paired)} paired rows to {root / 'stage5a_paired_statistics.csv'}")


if __name__ == "__main__":
    main()
