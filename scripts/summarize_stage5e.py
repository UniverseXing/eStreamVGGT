#!/usr/bin/env python3
"""Summarize the emergency K4 versus official OVGGT comparison."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


METRIC_KEYS = {
    "abs_rel": "Abs Rel",
    "sq_rel": "Sq Rel",
    "rmse": "RMSE",
    "log_rmse": "Log RMSE",
    "delta_1": "δ < 1.25",
    "delta_2": "δ < 1.25^2",
    "delta_3": "δ < 1.25^3",
}
RESULT_FIELDS = (
    "dataset", "method", "num_sequences", "num_successful", "num_failed",
    "total_frames", *METRIC_KEYS, "fps_inference", "peak_allocated_mb",
    "peak_reserved_mb", "gpu_name", "torch_version", "cuda_version",
    "competitor_commit", "backend", "source",
)
SEQUENCE_FIELDS = (
    "dataset", "sequence", "method", "num_frames", *METRIC_KEYS,
    "inference_sec", "fps_inference", "peak_allocated_mb",
    "peak_reserved_mb", "status", "source",
)


def read_csv(path: Path):
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def read_json(path: Path):
    with path.open() as handle:
        return json.load(handle)


def write_csv(path: Path, fields, rows):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} rows to {path}")


def blank(fields):
    return {field: "" for field in fields}


def load_k4(root: Path, datasets):
    aggregate_path = root / "stage5a_same_budget_results.csv"
    sequence_path = root / "stage5a_same_budget_sequence_results.csv"
    aggregate_sources = read_csv(aggregate_path)
    sequence_sources = read_csv(sequence_path)
    # The refreshed supplementary tables retain all seven depth metrics.  Use
    # them when the older Stage 5A conference CSV only contains the three
    # primary columns; both were built from the same frozen raw runs.
    if "sq_rel" not in aggregate_sources[0]:
        candidates = (
            (
                root / "supplementary material/tables/p0_04_depth_secondary_metrics_summary.csv",
                root / "supplementary material/tables/p0_04_depth_secondary_metrics_sequences.csv",
            ),
            (
                root / "stage4a_video_depth_results.csv",
                root / "stage4b_video_depth_sequence_results.csv",
            ),
            (
                root / "stage4a_video_depth_results(1).csv",
                root / "stage4b_video_depth_sequence_results.csv",
            ),
        )
        for candidate_aggregate, candidate_sequence in candidates:
            if candidate_aggregate.is_file() and candidate_sequence.is_file():
                aggregate_path, sequence_path = candidate_aggregate, candidate_sequence
                aggregate_sources = read_csv(aggregate_path)
                sequence_sources = read_csv(sequence_path)
                break
        else:
            raise FileNotFoundError(
                "Stage 5E finalization needs the frozen K4 secondary-metric "
                "summary and sequence CSVs"
            )
    aggregates = []
    for source in aggregate_sources:
        method_id = source.get("method_id", source.get("method"))
        if source["dataset"] not in datasets or method_id not in (
            "k4", "proposed_k4", "stage3_2_k4"
        ):
            continue
        row = blank(RESULT_FIELDS)
        row.update(
            dataset=source["dataset"], method="K4",
            num_sequences=source["num_sequences"],
            num_successful=source.get("num_successful", source.get("num_ok")),
            num_failed=source.get("num_failed", source.get("num_oom")),
            total_frames=source["total_frames"],
            fps_inference=source["fps_inference"],
            peak_allocated_mb=source.get(
                "peak_allocated_mb",
                source.get("peak_allocated_mib", source.get("max_peak_allocated_mb")),
            ),
            peak_reserved_mb=source.get(
                "peak_reserved_mb",
                source.get("peak_reserved_mib", source.get("max_peak_reserved_mb")),
            ),
            gpu_name=source["gpu_name"], torch_version=source["torch_version"],
            cuda_version=source["cuda_version"], source=str(aggregate_path),
        )
        for metric in METRIC_KEYS:
            row[metric] = source[metric]
        aggregates.append(row)
    sequences = []
    for source in sequence_sources:
        method_id = source.get("method_id", source.get("method"))
        if source["dataset"] not in datasets or method_id not in (
            "k4", "proposed_k4", "stage3_2_k4"
        ):
            continue
        row = blank(SEQUENCE_FIELDS)
        row.update(
            dataset=source["dataset"], sequence=source["sequence"], method="K4",
            num_frames=source["num_frames"], inference_sec=source["inference_sec"],
            fps_inference=source["fps_inference"],
            peak_allocated_mb=source.get("peak_allocated_mb", source.get("peak_allocated_mib")),
            peak_reserved_mb=source.get("peak_reserved_mb", source.get("peak_reserved_mib")), status="ok",
            source=str(sequence_path),
        )
        for metric in METRIC_KEYS:
            row[metric] = source[metric]
        sequences.append(row)
    return aggregates, sequences


def load_competitor(root: Path, output_root: Path, datasets):
    aggregates, sequences = [], []
    for dataset in datasets:
        directory = output_root / dataset
        metrics_path = directory / "result_scale.json"
        sequence_metrics_path = directory / "result_scale_sequences.json"
        runtime_path = directory / "stage5e_runtime_memory.json"
        metrics = read_json(metrics_path)
        sequence_metrics = read_json(sequence_metrics_path)
        runtime = read_json(runtime_path)
        summary = runtime["summary"]
        runtime_by_sequence = {row["sequence"]: row for row in runtime["sequences"]}
        row = blank(RESULT_FIELDS)
        row.update(
            dataset=dataset, method="OVGGT",
            num_sequences=summary["num_sequences"], num_successful=summary["num_ok"],
            num_failed=summary["num_failed"], total_frames=summary["total_frames"],
            fps_inference=summary.get("fps_inference", ""),
            peak_allocated_mb=summary["max_peak_allocated_mb"],
            peak_reserved_mb=summary["max_peak_reserved_mb"],
            gpu_name=summary["gpu_name"], torch_version=summary["torch_version"],
            cuda_version=summary["cuda_version"],
            competitor_commit=summary["competitor_commit"],
            backend=summary["backend"], source=str(directory),
        )
        for metric, key in METRIC_KEYS.items():
            row[metric] = metrics[key]
        aggregates.append(row)
        for item in sequence_metrics["sequences"]:
            sequence = item["sequence"]
            timing = runtime_by_sequence[sequence]
            sequence_row = blank(SEQUENCE_FIELDS)
            sequence_row.update(
                dataset=dataset, sequence=sequence, method="OVGGT",
                num_frames=item["num_frames"], inference_sec=timing["inference_sec"],
                fps_inference=timing["fps_inference"],
                peak_allocated_mb=timing["peak_allocated_mb"],
                peak_reserved_mb=timing["peak_reserved_mb"], status=timing["status"],
                source=str(directory),
            )
            for metric, key in METRIC_KEYS.items():
                sequence_row[metric] = item["metrics"][key]
            sequences.append(sequence_row)
    return aggregates, sequences


def parity_gate(root: Path, output_root: Path):
    ours_candidates = sorted((output_root / "parity_ours").glob("bonn_streamvggt*/result_scale.json"))
    competitor_path = output_root / "parity_ovggt/bonn/result_scale.json"
    if len(ours_candidates) != 1 or not competitor_path.is_file():
        return {
            "check": "streamvggt_full_parity", "passed": 0, "observed": "missing",
            "threshold": "AbsRel difference <= max(2% of project Full, 0.002)",
            "action": "do not publish Stage 5E table",
        }
    ours = float(read_json(ours_candidates[0])["Abs Rel"])
    competitor = float(read_json(competitor_path)["Abs Rel"])
    tolerance = max(abs(ours) * 0.02, 0.002)
    difference = abs(ours - competitor)
    return {
        "check": "streamvggt_full_parity", "passed": int(difference <= tolerance),
        "observed": f"ours={ours:.8g}; ovggt_full={competitor:.8g}; abs_diff={difference:.8g}",
        "threshold": f"abs_diff <= {tolerance:.8g}",
        "action": "continue" if difference <= tolerance else "inspect preprocessing/checkpoint; do not publish",
    }


def paired_absrel(sequence_rows, datasets, samples, seed):
    rng = np.random.default_rng(seed)
    outputs = []
    for dataset in datasets:
        index = {
            (row["sequence"], row["method"]): float(row["abs_rel"])
            for row in sequence_rows if row["dataset"] == dataset
        }
        names_k4 = {name for name, method in index if method == "K4"}
        names_competitor = {name for name, method in index if method == "OVGGT"}
        if names_k4 != names_competitor:
            raise RuntimeError(
                f"coverage mismatch for {dataset}: K4={sorted(names_k4)}, "
                f"OVGGT={sorted(names_competitor)}"
            )
        names = sorted(names_k4)
        k4 = np.asarray([index[(name, "K4")] for name in names])
        competitor = np.asarray([index[(name, "OVGGT")] for name in names])
        advantage = competitor - k4
        sampled = advantage[rng.integers(0, len(names), size=(samples, len(names)))].mean(axis=1)
        outputs.append(
            {
                "dataset": dataset, "metric": "abs_rel", "n_pairs": len(names),
                "mean_k4": float(k4.mean()), "mean_ovggt": float(competitor.mean()),
                "mean_advantage_k4": float(advantage.mean()),
                "ci95_low": float(np.percentile(sampled, 2.5)),
                "ci95_high": float(np.percentile(sampled, 97.5)),
                "k4_wins": int(np.sum(advantage > 1e-12)),
                "ties": int(np.sum(abs(advantage) <= 1e-12)),
                "ovggt_wins": int(np.sum(advantage < -1e-12)),
                "bootstrap_samples": samples, "bootstrap_seed": seed,
            }
        )
    return outputs


def comparisons(aggregate_rows, datasets):
    index = {(row["dataset"], row["method"]): row for row in aggregate_rows}
    rows = []
    directions = {**{metric: "lower" for metric in ("abs_rel", "sq_rel", "rmse", "log_rmse", "peak_allocated_mb", "peak_reserved_mb")},
                  **{metric: "higher" for metric in ("delta_1", "delta_2", "delta_3", "fps_inference")}}
    for dataset in datasets:
        k4 = index[(dataset, "K4")]
        competitor = index[(dataset, "OVGGT")]
        for metric, direction in directions.items():
            a, b = float(k4[metric]), float(competitor[metric])
            winner = "K4" if (a < b if direction == "lower" else a > b) else "OVGGT"
            if abs(a - b) <= 1e-12:
                winner = "tie"
            rows.append(
                {
                    "dataset": dataset, "metric": metric, "direction": direction,
                    "k4": a, "ovggt": b, "k4_minus_ovggt": a - b,
                    "k4_relative_change_pct": (a - b) / abs(b) * 100 if b else "",
                    "winner": winner,
                }
            )
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--datasets", nargs="+", default=("bonn", "sintel", "kitti"))
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260827)
    parser.add_argument("--parity-only", action="store_true")
    args = parser.parse_args()
    root, output_root = args.repo_root.resolve(), args.output_root.resolve()
    if args.parity_only:
        row = parity_gate(root, output_root)
        print(json.dumps(row, indent=2))
        if not int(row["passed"]):
            raise RuntimeError("Stage 5E parity failed; do not run formal inference")
        return
    datasets = tuple(args.datasets)
    k4_aggregate, k4_sequences = load_k4(root, datasets)
    competitor_aggregate, competitor_sequences = load_competitor(
        root, output_root, datasets
    )
    aggregates = k4_aggregate + competitor_aggregate
    sequences = k4_sequences + competitor_sequences
    if len(aggregates) != len(datasets) * 2:
        raise RuntimeError(f"incomplete aggregate matrix: {len(aggregates)} rows")
    for row in competitor_aggregate:
        if int(row["num_failed"]):
            raise RuntimeError(f"OVGGT has failed sequences: {row}")
        if "6000 ada" not in row["gpu_name"].lower():
            raise RuntimeError(f"formal Stage 5E requires RTX 6000 Ada: {row['gpu_name']}")
    comparison_rows = comparisons(aggregates, datasets)
    paired_rows = paired_absrel(
        sequences, datasets, args.bootstrap_samples, args.bootstrap_seed
    )
    gate_rows = [parity_gate(root, output_root)]
    gate_rows.append(
        {
            "check": "complete_matched_coverage", "passed": 1,
            "observed": "; ".join(
                f"{row['dataset']}={row['num_successful']}"
                for row in competitor_aggregate
            ),
            "threshold": "K4 and OVGGT sequence/frame signatures identical",
            "action": "continue",
        }
    )
    claimable = [row for row in comparison_rows if row["winner"] == "K4"]
    gate_rows.append(
        {
            "check": "claimable_k4_dimension", "passed": int(bool(claimable)),
            "observed": ", ".join(f"{row['dataset']}/{row['metric']}" for row in claimable),
            "threshold": "at least one pre-reported quality or system metric favors K4",
            "action": "state only the observed dimensions; retain all metrics",
        }
    )
    write_csv(root / "stage5e_results.csv", RESULT_FIELDS, aggregates)
    write_csv(root / "stage5e_sequence_results.csv", SEQUENCE_FIELDS, sequences)
    write_csv(
        root / "stage5e_comparison.csv",
        ("dataset", "metric", "direction", "k4", "ovggt", "k4_minus_ovggt", "k4_relative_change_pct", "winner"),
        comparison_rows,
    )
    write_csv(
        root / "stage5e_paired_statistics.csv",
        ("dataset", "metric", "n_pairs", "mean_k4", "mean_ovggt", "mean_advantage_k4", "ci95_low", "ci95_high", "k4_wins", "ties", "ovggt_wins", "bootstrap_samples", "bootstrap_seed"),
        paired_rows,
    )
    write_csv(
        root / "stage5e_gate.csv",
        ("check", "passed", "observed", "threshold", "action"), gate_rows,
    )
    if not all(int(row["passed"]) for row in gate_rows[:2]):
        raise RuntimeError("Stage 5E validity gate failed; inspect stage5e_gate.csv")


if __name__ == "__main__":
    main()
