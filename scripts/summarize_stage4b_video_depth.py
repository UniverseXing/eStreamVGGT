#!/usr/bin/env python3
"""Build paired Stage 4B VideoDepth statistics from frozen Stage 4A outputs."""

import argparse
import csv
import glob
import itertools
import json
import os

import numpy as np


DATASETS = {"bonn": 5, "kitti": 13, "sintel": 23}
METHODS = (
    "full_cache",
    "stage3_2_k4",
    "old_dino_k6",
    "temporal_binned_dino_k8",
)
METHOD_ORDER = {method: index for index, method in enumerate(METHODS)}
METRIC_KEYS = {
    "abs_rel": "Abs Rel",
    "sq_rel": "Sq Rel",
    "rmse": "RMSE",
    "log_rmse": "Log RMSE",
    "delta_1": "δ < 1.25",
    "delta_2": "δ < 1.25^2",
    "delta_3": "δ < 1.25^3",
}
STATS_METRICS = (
    "abs_rel",
    "sq_rel",
    "rmse",
    "log_rmse",
    "delta_1",
    "delta_2",
    "delta_3",
    "inference_sec",
    "fps_inference",
    "peak_allocated_mb",
    "peak_reserved_mb",
)
PAIRED_METRICS = {
    "abs_rel": "lower",
    "rmse": "lower",
    "delta_1": "higher",
    "inference_sec": "lower",
    "peak_allocated_mb": "lower",
}
REGRET_METRICS = {"abs_rel": "lower", "delta_1": "higher"}

SEQUENCE_FIELDS = (
    "dataset",
    "sequence",
    "method",
    "num_frames",
    "valid_pixels",
    "abs_rel",
    "sq_rel",
    "rmse",
    "log_rmse",
    "delta_1",
    "delta_2",
    "delta_3",
    "inference_sec",
    "fps_inference",
    "peak_allocated_mb",
    "peak_reserved_mb",
    "gpu_name",
    "torch_version",
    "cuda_version",
    "slurm_job_id",
    "source",
)
STAT_FIELDS = (
    "dataset",
    "method",
    "metric",
    "n_sequences",
    "mean",
    "median",
    "std",
    "ci95_low",
    "ci95_high",
    "aggregation",
)
PAIRED_FIELDS = (
    "dataset",
    "metric",
    "better_direction",
    "method_a",
    "method_b",
    "n_pairs",
    "mean_a",
    "mean_b",
    "mean_advantage_a",
    "advantage_ci95_low",
    "advantage_ci95_high",
    "wins_a",
    "ties",
    "losses_a",
    "significance",
)
REGRET_FIELDS = (
    "dataset",
    "method",
    "metric",
    "n_sequences",
    "mean_normalized_regret",
    "median_normalized_regret",
    "max_normalized_regret",
    "oracle_wins",
)
PARETO_FIELDS = (
    "dataset",
    "method",
    "abs_rel",
    "delta_1",
    "total_inference_sec",
    "fps_inference",
    "max_peak_allocated_mb",
    "max_peak_reserved_mb",
    "pareto_absrel_allocated_time",
    "dominated_by",
    "source",
)


def method_from_directory(name):
    for method in (
        "temporal_binned_dino_k8",
        "stage3_2_k4",
        "old_dino_k6",
        "full_cache",
    ):
        if f"stage4a_{method}" in name:
            return method
    raise ValueError(f"cannot infer Stage 4A method from {name}")


def dataset_from_directory(name):
    marker = "_streamvggt"
    if marker not in name:
        raise ValueError(f"cannot infer dataset from {name}")
    return name.split(marker, 1)[0]


def write_csv(path, fields, rows):
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} rows to {path}")


def bootstrap_mean_ci(values, rng, samples):
    values = np.asarray(values, dtype=np.float64)
    indices = rng.integers(0, len(values), size=(samples, len(values)))
    bootstrap_means = values[indices].mean(axis=1)
    return tuple(np.percentile(bootstrap_means, (2.5, 97.5)))


def close(a, b):
    return bool(np.isclose(a, b, rtol=1e-8, atol=1e-10))


def load_rows(results_root):
    pattern = os.path.join(results_root, "*stage4a_*", "runtime_memory_rank0.json")
    runtime_paths = sorted(glob.glob(pattern))
    expected_matrix = {
        (dataset, method) for dataset in DATASETS for method in METHODS
    }
    seen_matrix = set()
    sequence_rows = []
    aggregate_rows = []

    for runtime_path in runtime_paths:
        result_dir = os.path.dirname(runtime_path)
        dirname = os.path.basename(result_dir)
        dataset = dataset_from_directory(dirname)
        method = method_from_directory(dirname)
        matrix_key = (dataset, method)
        if matrix_key in seen_matrix:
            raise RuntimeError(f"duplicate Stage 4A result: {matrix_key}")
        seen_matrix.add(matrix_key)

        with open(runtime_path) as handle:
            runtime_payload = json.load(handle)
        summary = runtime_payload["summary"]
        runtime_sequences = {
            item["seq"]: item for item in runtime_payload["sequences"]
        }
        if any(item.get("status") != "ok" for item in runtime_sequences.values()):
            raise RuntimeError(f"non-ok runtime sequence in {result_dir}")

        sequence_path = os.path.join(result_dir, "result_scale_sequences.json")
        if not os.path.isfile(sequence_path):
            raise FileNotFoundError(
                f"missing per-sequence metrics: {sequence_path}; "
                "run Stage 4B evaluation first"
            )
        with open(sequence_path) as handle:
            metrics_payload = json.load(handle)
        if metrics_payload.get("dataset") != dataset:
            raise RuntimeError(
                f"dataset mismatch in {sequence_path}: "
                f"{metrics_payload.get('dataset')} != {dataset}"
            )
        if metrics_payload.get("num_sequences") != DATASETS[dataset]:
            raise RuntimeError(
                f"{sequence_path}: expected {DATASETS[dataset]} sequences, "
                f"found {metrics_payload.get('num_sequences')}"
            )

        metric_sequences = {
            item["sequence"]: item for item in metrics_payload["sequences"]
        }
        if set(metric_sequences) != set(runtime_sequences):
            raise RuntimeError(
                f"metric/runtime sequence mismatch for {matrix_key}: "
                f"metrics-only={sorted(set(metric_sequences)-set(runtime_sequences))}, "
                f"runtime-only={sorted(set(runtime_sequences)-set(metric_sequences))}"
            )

        for sequence in sorted(metric_sequences):
            metric_item = metric_sequences[sequence]
            runtime_item = runtime_sequences[sequence]
            if int(metric_item["num_frames"]) != int(runtime_item["num_frames"]):
                raise RuntimeError(
                    f"frame mismatch for {matrix_key}/{sequence}: "
                    f"{metric_item['num_frames']} != {runtime_item['num_frames']}"
                )
            metrics = metric_item["metrics"]
            row = {
                "dataset": dataset,
                "sequence": sequence,
                "method": method,
                "num_frames": int(metric_item["num_frames"]),
                "valid_pixels": int(metrics["valid_pixels"]),
                "inference_sec": float(runtime_item["inference_sec"]),
                "fps_inference": float(runtime_item["fps_inference"]),
                "peak_allocated_mb": float(runtime_item["peak_allocated_mb"]),
                "peak_reserved_mb": float(runtime_item["peak_reserved_mb"]),
                "gpu_name": summary.get("gpu_name"),
                "torch_version": summary.get("torch_version"),
                "cuda_version": summary.get("cuda_version"),
                "slurm_job_id": summary.get("slurm_job_id"),
                "source": result_dir,
            }
            for output_key, input_key in METRIC_KEYS.items():
                row[output_key] = float(metrics[input_key])
            sequence_rows.append(row)

        aggregate_path = os.path.join(result_dir, "result_scale.json")
        with open(aggregate_path) as handle:
            aggregate_metrics = json.load(handle)
        for input_key in METRIC_KEYS.values():
            if not np.isclose(
                float(metrics_payload["weighted_average"][input_key]),
                float(aggregate_metrics[input_key]),
                rtol=1e-10,
                atol=1e-12,
            ):
                raise RuntimeError(
                    f"weighted aggregate mismatch for "
                    f"{matrix_key}/{input_key}"
                )
        aggregate_rows.append(
            {
                "dataset": dataset,
                "method": method,
                "abs_rel": float(aggregate_metrics["Abs Rel"]),
                "delta_1": float(aggregate_metrics["δ < 1.25"]),
                "total_inference_sec": float(summary["total_inference_sec"]),
                "fps_inference": float(summary["fps_inference"]),
                "max_peak_allocated_mb": float(
                    summary["max_peak_allocated_mb"]
                ),
                "max_peak_reserved_mb": float(summary["max_peak_reserved_mb"]),
                "source": result_dir,
            }
        )

    if seen_matrix != expected_matrix:
        raise RuntimeError(
            "incomplete Stage 4A matrix: "
            f"missing={sorted(expected_matrix-seen_matrix)}, "
            f"unexpected={sorted(seen_matrix-expected_matrix)}"
        )

    for dataset, expected_count in DATASETS.items():
        method_sets = {}
        for method in METHODS:
            rows = [
                row
                for row in sequence_rows
                if row["dataset"] == dataset and row["method"] == method
            ]
            method_sets[method] = {row["sequence"] for row in rows}
            if len(rows) != expected_count:
                raise RuntimeError(
                    f"{dataset}/{method}: expected {expected_count} sequences, "
                    f"found {len(rows)}"
                )
        reference = method_sets["full_cache"]
        for method, sequences in method_sets.items():
            if sequences != reference:
                raise RuntimeError(
                    f"{dataset} sequence coverage differs for {method}"
                )

        for sequence in reference:
            frame_counts = {
                row["num_frames"]
                for row in sequence_rows
                if row["dataset"] == dataset and row["sequence"] == sequence
            }
            if len(frame_counts) != 1:
                raise RuntimeError(
                    f"{dataset}/{sequence}: methods used different frame counts"
                )

    expected_rows = sum(DATASETS.values()) * len(METHODS)
    if len(sequence_rows) != expected_rows:
        raise RuntimeError(
            f"expected {expected_rows} paired sequence rows, "
            f"found {len(sequence_rows)}"
        )
    gpu_names = {row["gpu_name"] for row in sequence_rows}
    torch_versions = {row["torch_version"] for row in sequence_rows}
    cuda_versions = {row["cuda_version"] for row in sequence_rows}
    if (
        len(gpu_names) != 1
        or "6000 ada" not in next(iter(gpu_names)).lower()
        or len(torch_versions) != 1
        or len(cuda_versions) != 1
    ):
        raise RuntimeError(
            "Stage 4B provenance mismatch: "
            f"gpu={sorted(gpu_names)}, torch={sorted(torch_versions)}, "
            f"cuda={sorted(cuda_versions)}"
        )
    sequence_rows.sort(
        key=lambda row: (
            row["dataset"],
            row["sequence"],
            METHOD_ORDER[row["method"]],
        )
    )
    aggregate_rows.sort(
        key=lambda row: (row["dataset"], METHOD_ORDER[row["method"]])
    )
    return sequence_rows, aggregate_rows


def build_statistics(sequence_rows, rng, bootstrap_samples):
    outputs = []
    for dataset in DATASETS:
        for method in METHODS:
            rows = [
                row
                for row in sequence_rows
                if row["dataset"] == dataset and row["method"] == method
            ]
            for metric in STATS_METRICS:
                values = np.asarray(
                    [float(row[metric]) for row in rows], dtype=np.float64
                )
                ci_low, ci_high = bootstrap_mean_ci(
                    values, rng, bootstrap_samples
                )
                outputs.append(
                    {
                        "dataset": dataset,
                        "method": method,
                        "metric": metric,
                        "n_sequences": len(values),
                        "mean": float(np.mean(values)),
                        "median": float(np.median(values)),
                        "std": float(np.std(values, ddof=1)),
                        "ci95_low": float(ci_low),
                        "ci95_high": float(ci_high),
                        "aggregation": "unweighted_sequence_mean",
                    }
                )
    return outputs


def build_paired_comparisons(sequence_rows, rng, bootstrap_samples):
    indexed = {
        (row["dataset"], row["sequence"], row["method"]): row
        for row in sequence_rows
    }
    outputs = []
    for dataset in DATASETS:
        sequences = sorted(
            {
                row["sequence"]
                for row in sequence_rows
                if row["dataset"] == dataset
            }
        )
        for method_a, method_b in itertools.combinations(METHODS, 2):
            for metric, direction in PAIRED_METRICS.items():
                values_a = np.asarray(
                    [
                        float(indexed[(dataset, sequence, method_a)][metric])
                        for sequence in sequences
                    ]
                )
                values_b = np.asarray(
                    [
                        float(indexed[(dataset, sequence, method_b)][metric])
                        for sequence in sequences
                    ]
                )
                advantage = (
                    values_b - values_a
                    if direction == "lower"
                    else values_a - values_b
                )
                ci_low, ci_high = bootstrap_mean_ci(
                    advantage, rng, bootstrap_samples
                )
                wins = sum(
                    not close(a, b)
                    and ((a < b) if direction == "lower" else (a > b))
                    for a, b in zip(values_a, values_b)
                )
                ties = sum(close(a, b) for a, b in zip(values_a, values_b))
                losses = len(sequences) - wins - ties
                if ci_low > 0:
                    significance = "A_BETTER"
                elif ci_high < 0:
                    significance = "B_BETTER"
                else:
                    significance = "NO_CLEAR_DIFFERENCE"
                outputs.append(
                    {
                        "dataset": dataset,
                        "metric": metric,
                        "better_direction": direction,
                        "method_a": method_a,
                        "method_b": method_b,
                        "n_pairs": len(sequences),
                        "mean_a": float(np.mean(values_a)),
                        "mean_b": float(np.mean(values_b)),
                        "mean_advantage_a": float(np.mean(advantage)),
                        "advantage_ci95_low": float(ci_low),
                        "advantage_ci95_high": float(ci_high),
                        "wins_a": wins,
                        "ties": ties,
                        "losses_a": losses,
                        "significance": significance,
                    }
                )
    return outputs


def build_regret(sequence_rows):
    indexed = {
        (row["dataset"], row["sequence"], row["method"]): row
        for row in sequence_rows
    }
    outputs = []
    for dataset_label, datasets in [
        *[(dataset, (dataset,)) for dataset in DATASETS],
        ("all", tuple(DATASETS)),
    ]:
        sequence_keys = sorted(
            {
                (row["dataset"], row["sequence"])
                for row in sequence_rows
                if row["dataset"] in datasets
            }
        )
        for method in METHODS:
            for metric, direction in REGRET_METRICS.items():
                regrets = []
                wins = 0
                for dataset, sequence in sequence_keys:
                    values = {
                        candidate: float(
                            indexed[(dataset, sequence, candidate)][metric]
                        )
                        for candidate in METHODS
                    }
                    oracle = (
                        min(values.values())
                        if direction == "lower"
                        else max(values.values())
                    )
                    value = values[method]
                    if close(value, oracle):
                        wins += 1
                    denominator = max(abs(oracle), 1e-12)
                    regret = (
                        (value - oracle) / denominator
                        if direction == "lower"
                        else (oracle - value) / denominator
                    )
                    regrets.append(regret)
                outputs.append(
                    {
                        "dataset": dataset_label,
                        "method": method,
                        "metric": metric,
                        "n_sequences": len(regrets),
                        "mean_normalized_regret": float(np.mean(regrets)),
                        "median_normalized_regret": float(np.median(regrets)),
                        "max_normalized_regret": float(np.max(regrets)),
                        "oracle_wins": wins,
                    }
                )
    return outputs


def build_pareto(aggregate_rows):
    outputs = []
    for dataset in DATASETS:
        rows = [row for row in aggregate_rows if row["dataset"] == dataset]
        for row in rows:
            dominated_by = []
            for other in rows:
                if other["method"] == row["method"]:
                    continue
                weakly_better = (
                    other["abs_rel"] <= row["abs_rel"]
                    and other["max_peak_allocated_mb"]
                    <= row["max_peak_allocated_mb"]
                    and other["total_inference_sec"]
                    <= row["total_inference_sec"]
                )
                strictly_better = (
                    other["abs_rel"] < row["abs_rel"]
                    or other["max_peak_allocated_mb"]
                    < row["max_peak_allocated_mb"]
                    or other["total_inference_sec"]
                    < row["total_inference_sec"]
                )
                if weakly_better and strictly_better:
                    dominated_by.append(other["method"])
            output = dict(row)
            output["pareto_absrel_allocated_time"] = (
                "yes" if not dominated_by else "no"
            )
            output["dominated_by"] = " ".join(
                sorted(dominated_by, key=METHOD_ORDER.get)
            )
            outputs.append(output)
    return outputs


def main():
    parser = argparse.ArgumentParser("Summarize Stage 4B VideoDepth")
    parser.add_argument("--results-root", default="eval_results/video_depth")
    parser.add_argument(
        "--sequence-output", default="stage4b_video_depth_sequence_results.csv"
    )
    parser.add_argument(
        "--statistics-output", default="stage4b_video_depth_statistics.csv"
    )
    parser.add_argument(
        "--paired-output", default="stage4b_video_depth_paired_comparison.csv"
    )
    parser.add_argument(
        "--regret-output", default="stage4b_video_depth_regret.csv"
    )
    parser.add_argument("--pareto-output", default="stage4b_pareto.csv")
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if args.bootstrap_samples < 1000:
        raise ValueError("--bootstrap-samples must be at least 1000")

    rng = np.random.default_rng(args.seed)
    sequence_rows, aggregate_rows = load_rows(args.results_root)
    statistics = build_statistics(
        sequence_rows, rng, args.bootstrap_samples
    )
    paired = build_paired_comparisons(
        sequence_rows, rng, args.bootstrap_samples
    )
    regret = build_regret(sequence_rows)
    pareto = build_pareto(aggregate_rows)

    write_csv(args.sequence_output, SEQUENCE_FIELDS, sequence_rows)
    write_csv(args.statistics_output, STAT_FIELDS, statistics)
    write_csv(args.paired_output, PAIRED_FIELDS, paired)
    write_csv(args.regret_output, REGRET_FIELDS, regret)
    write_csv(args.pareto_output, PARETO_FIELDS, pareto)


if __name__ == "__main__":
    main()
