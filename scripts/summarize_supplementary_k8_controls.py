#!/usr/bin/env python3
"""Summarize optional matched-budget K8 VideoDepth controls."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from summarize_stage6a import COMMON_FIELDS, SEQUENCE_FIELDS, load_video_depth, write_csv  # noqa: E402


DATASETS = {"bonn": 5, "sintel": 23, "kitti": 13}
CONTROLS = ("recent8", "nonhierarchical_dino8")
DEPTH_FIELDS = ("abs_rel", "sq_rel", "rmse", "log_rmse", "delta_1", "delta_2", "delta_3")
PAIRED_FIELDS = (
    "dataset", "metric", "proposed", "control", "n_pairs", "mean_proposed",
    "mean_control", "mean_advantage_proposed", "ci95_low", "ci95_high",
    "wins_proposed", "ties", "losses_proposed", "bootstrap_samples", "bootstrap_seed",
)


def read_csv(path: Path):
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def frozen_k8(root: Path):
    aggregate_path = next(
        path for path in (
            root / "stage4a_video_depth_results.csv",
            root / "stage4a_video_depth_results(1).csv",
        ) if path.is_file()
    )
    sequence_path = root / "stage4b_video_depth_sequence_results.csv"
    aggregates, sequences = [], []
    for source in read_csv(aggregate_path):
        if source["method"] != "temporal_binned_dino_k8":
            continue
        row = {field: "" for field in COMMON_FIELDS}
        row.update(
            task="video_depth", dataset=source["dataset"], method="proposed_k8",
            random_seed=0, cache_policy="anchor_recent_dino_diverse_k8",
            cache_window_size=8, num_sequences=source["num_sequences"],
            num_successful=source["num_ok"], num_failed=source["num_oom"],
            total_frames=source["total_frames"], fps_inference=source["fps_inference"],
            peak_allocated_mb=source["max_peak_allocated_mb"],
            peak_reserved_mb=source["max_peak_reserved_mb"], gpu_name=source["gpu_name"],
            torch_version=source["torch_version"], cuda_version=source["cuda_version"],
            slurm_job_id=source["slurm_job_id"], source=str(aggregate_path),
            **{field: source[field] for field in DEPTH_FIELDS},
        )
        aggregates.append(row)
    for source in read_csv(sequence_path):
        if source["method"] != "temporal_binned_dino_k8":
            continue
        row = {field: "" for field in SEQUENCE_FIELDS}
        row.update(
            task="video_depth", dataset=source["dataset"], sequence=source["sequence"],
            method="proposed_k8", random_seed=0, num_frames=source["num_frames"],
            inference_sec=source["inference_sec"], fps_inference=source["fps_inference"],
            peak_allocated_mb=source["peak_allocated_mb"],
            peak_reserved_mb=source["peak_reserved_mb"], source=str(sequence_path),
            **{field: source[field] for field in DEPTH_FIELDS},
        )
        sequences.append(row)
    return aggregates, sequences


def paired(rows, samples: int, seed: int):
    rng = np.random.default_rng(seed)
    output = []
    for dataset in DATASETS:
        proposed = {
            row["sequence"]: float(row["abs_rel"])
            for row in rows if row["dataset"] == dataset and row["method"] == "proposed_k8"
        }
        for control in CONTROLS:
            baseline = {
                row["sequence"]: float(row["abs_rel"])
                for row in rows if row["dataset"] == dataset and row["method"] == control
            }
            if set(proposed) != set(baseline) or len(proposed) != DATASETS[dataset]:
                raise RuntimeError(f"coverage mismatch for {dataset}/{control}")
            names = sorted(proposed)
            a = np.asarray([proposed[name] for name in names])
            b = np.asarray([baseline[name] for name in names])
            advantage = b - a
            indices = rng.integers(0, len(names), size=(samples, len(names)))
            boot = advantage[indices].mean(axis=1)
            output.append({
                "dataset": dataset, "metric": "abs_rel", "proposed": "proposed_k8",
                "control": control, "n_pairs": len(names), "mean_proposed": a.mean(),
                "mean_control": b.mean(), "mean_advantage_proposed": advantage.mean(),
                "ci95_low": np.percentile(boot, 2.5), "ci95_high": np.percentile(boot, 97.5),
                "wins_proposed": int(np.sum(advantage > 1e-12)),
                "ties": int(np.sum(np.abs(advantage) <= 1e-12)),
                "losses_proposed": int(np.sum(advantage < -1e-12)),
                "bootstrap_samples": samples, "bootstrap_seed": seed,
            })
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=ROOT)
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260824)
    args = parser.parse_args()
    root = args.repo_root.resolve()
    aggregates, sequences = frozen_k8(root)
    for method in CONTROLS:
        for dataset, count in DATASETS.items():
            aggregate, sequence_rows = load_video_depth(
                root, dataset, method, False, stage_tag="supplementary_p1"
            )
            if int(aggregate["num_successful"]) != count:
                raise RuntimeError(f"incomplete {dataset}/{method}")
            aggregates.append(aggregate)
            sequences.extend(sequence_rows)
    gpu_names = {row["gpu_name"] for row in aggregates}
    if len(gpu_names) != 1 or "6000 ada" not in next(iter(gpu_names)).lower():
        raise RuntimeError(f"K8 matched controls require one RTX 6000 Ada: {gpu_names}")
    order = {name: index for index, name in enumerate(("proposed_k8", *CONTROLS))}
    aggregates.sort(key=lambda row: (row["dataset"], order[row["method"]]))
    sequences.sort(key=lambda row: (row["dataset"], row["sequence"], order[row["method"]]))
    output = root / "eval_results/supplementary_k8_controls"
    output.mkdir(parents=True, exist_ok=True)
    write_csv(output / "k8_controls_summary.csv", COMMON_FIELDS, aggregates)
    write_csv(output / "k8_controls_sequences.csv", SEQUENCE_FIELDS, sequences)
    with (output / "k8_controls_paired.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=PAIRED_FIELDS)
        writer.writeheader()
        writer.writerows(paired(sequences, args.bootstrap_samples, args.bootstrap_seed))


if __name__ == "__main__":
    main()
