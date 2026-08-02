#!/usr/bin/env python3
"""Validate and summarize the public VideoDepth reproduction matrix."""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


DATASET_ORDER = ("bonn", "kitti", "sintel")
FROZEN_SEQUENCE_FRAMES = {
    "bonn": {
        "balloon2": 110,
        "crowd2": 110,
        "crowd3": 110,
        "person_tracking2": 110,
        "synchronous": 110,
    },
    "kitti": {
        "2011_09_26_drive_0002_sync_02": 67,
        "2011_09_26_drive_0005_sync_02": 110,
        "2011_09_26_drive_0013_sync_02": 110,
        "2011_09_26_drive_0020_sync_02": 76,
        "2011_09_26_drive_0023_sync_02": 110,
        "2011_09_26_drive_0036_sync_02": 110,
        "2011_09_26_drive_0079_sync_02": 90,
        "2011_09_26_drive_0095_sync_02": 110,
        "2011_09_26_drive_0113_sync_02": 77,
        "2011_09_28_drive_0037_sync_02": 79,
        "2011_09_29_drive_0026_sync_02": 110,
        "2011_09_30_drive_0016_sync_02": 110,
        "2011_10_03_drive_0047_sync_02": 110,
    },
    "sintel": {
        "alley_1": 50,
        "alley_2": 50,
        "ambush_2": 21,
        "ambush_4": 33,
        "ambush_5": 50,
        "ambush_6": 20,
        "ambush_7": 50,
        "bamboo_1": 50,
        "bamboo_2": 50,
        "bandage_1": 50,
        "bandage_2": 50,
        "cave_2": 50,
        "cave_4": 50,
        "market_2": 50,
        "market_5": 50,
        "market_6": 40,
        "mountain_1": 50,
        "shaman_2": 50,
        "shaman_3": 50,
        "sleeping_1": 50,
        "sleeping_2": 50,
        "temple_2": 50,
        "temple_3": 50,
    },
}
EXPECTED_SEQUENCE_COUNTS = {
    dataset: len(sequences) for dataset, sequences in FROZEN_SEQUENCE_FRAMES.items()
}
METHOD_ORDER = (
    "full_cache",
    "anchor_recent_dino_diverse_k4",
    "anchor_recent_dino_diverse_k6",
    "anchor_recent_dino_diverse_k8",
)
METHOD_CONFIGS = {
    "full_cache": ("full_cache", None),
    "anchor_recent_dino_diverse_k4": ("anchor_recent_dino_diverse_k4", 4),
    "anchor_recent_dino_diverse_k6": ("anchor_recent_dino_diverse_k6", 6),
    "anchor_recent_dino_diverse_k8": ("anchor_recent_dino_diverse_k8", 8),
}
METHOD_INDEX = {method: index for index, method in enumerate(METHOD_ORDER)}
DATASET_INDEX = {dataset: index for index, dataset in enumerate(DATASET_ORDER)}

METRIC_KEYS = {
    "abs_rel": "Abs Rel",
    "sq_rel": "Sq Rel",
    "rmse": "RMSE",
    "log_rmse": "Log RMSE",
    "delta_1": "δ < 1.25",
    "delta_2": "δ < 1.25^2",
    "delta_3": "δ < 1.25^3",
}
ERROR_METRICS = {"Abs Rel", "Sq Rel", "RMSE", "Log RMSE"}
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

AGGREGATE_FIELDS = (
    "run_scope",
    "dataset",
    "method",
    "cache_policy",
    "cache_window_size",
    "gpu_name",
    "torch_version",
    "cuda_version",
    "python_version",
    "slurm_job_id",
    "hostname",
    "input_size",
    "pose_eval_stride",
    "requested_max_frames",
    "num_sequences",
    "num_ok",
    "num_oom",
    "total_frames",
    "total_inference_sec",
    "fps_inference",
    "max_peak_allocated_mb",
    "max_peak_reserved_mb",
    "abs_rel",
    "sq_rel",
    "rmse",
    "log_rmse",
    "delta_1",
    "delta_2",
    "delta_3",
    "result_dir",
)
SEQUENCE_FIELDS = (
    "run_scope",
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
    "python_version",
    "slurm_job_id",
    "hostname",
    "input_size",
    "pose_eval_stride",
    "requested_max_frames",
    "source",
)
STAT_FIELDS = (
    "run_scope",
    "dataset",
    "method",
    "metric",
    "n_sequences",
    "mean",
    "median",
    "std",
    "ci95_low",
    "ci95_high",
    "bootstrap_samples",
    "bootstrap_seed",
    "aggregation",
)
PAIRED_FIELDS = (
    "run_scope",
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
    "bootstrap_samples",
    "bootstrap_seed",
)
REGRET_FIELDS = (
    "run_scope",
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
    "run_scope",
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="aggregate CSV; the five detailed CSVs are written beside it",
    )
    parser.add_argument("--datasets", nargs="+", required=True)
    parser.add_argument("--methods", nargs="+", required=True)
    parser.add_argument(
        "--allow-subset",
        action="store_true",
        help="allow fewer than the frozen sequence count (coverage must still match across methods)",
    )
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def ordered_selection(
    requested: Sequence[str], frozen_order: Sequence[str], label: str
) -> tuple[str, ...]:
    if len(requested) != len(set(requested)):
        raise ValueError(f"duplicate {label}: {requested}")
    unknown = sorted(set(requested) - set(frozen_order))
    if unknown:
        raise ValueError(f"unknown {label}: {unknown}")
    return tuple(item for item in frozen_order if item in requested)


def reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for key, value in pairs:
        if key in output:
            raise ValueError(f"duplicate JSON key: {key!r}")
        output[key] = value
    return output


def load_json(path: Path) -> dict[str, Any]:
    try:
        with path.open(encoding="utf-8") as handle:
            payload = json.load(handle, object_pairs_hook=reject_duplicate_keys)
    except (json.JSONDecodeError, ValueError) as error:
        raise ValueError(f"invalid JSON in {path}: {error}") from error
    if not isinstance(payload, dict):
        raise ValueError(f"expected a JSON object in {path}")
    return payload


def finite_float(value: Any, label: str, *, minimum: float | None = None) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be numeric")
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label} must be numeric") from error
    if not math.isfinite(number):
        raise ValueError(f"{label} must be finite")
    if minimum is not None and number < minimum:
        raise ValueError(f"{label} must be >= {minimum}")
    return number


def exact_int(value: Any, label: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be an integer")
    try:
        number = int(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label} must be an integer") from error
    if isinstance(value, float) and not value.is_integer():
        raise ValueError(f"{label} must be an integer")
    if isinstance(value, str) and value.strip() != str(number):
        raise ValueError(f"{label} must be an integer")
    if number < minimum:
        raise ValueError(f"{label} must be >= {minimum}")
    return number


def close(a: Any, b: Any) -> bool:
    return math.isclose(float(a), float(b), rel_tol=1e-8, abs_tol=1e-10)


def normalize_window(value: Any, label: str) -> int | None:
    if value in (None, ""):
        return None
    return exact_int(value, label, minimum=1)


def unique_index(
    items: Any, key: str, label: str
) -> dict[str, Mapping[str, Any]]:
    if not isinstance(items, list):
        raise ValueError(f"{label} must be a JSON list")
    indexed: dict[str, Mapping[str, Any]] = {}
    for position, item in enumerate(items):
        if not isinstance(item, dict):
            raise ValueError(f"{label}[{position}] must be a JSON object")
        name = item.get(key)
        if not isinstance(name, str) or not name:
            raise ValueError(f"{label}[{position}].{key} must be a non-empty string")
        if name in indexed:
            raise ValueError(f"duplicate {label} entry: {name!r}")
        indexed[name] = item
    return indexed


def validate_metric_map(
    metrics: Any, label: str
) -> tuple[int, dict[str, float]]:
    if not isinstance(metrics, dict):
        raise ValueError(f"{label} must be a JSON object")
    missing = (set(METRIC_KEYS.values()) | {"valid_pixels"}) - set(metrics)
    if missing:
        raise ValueError(f"{label} is missing metrics: {sorted(missing)}")
    valid_pixels = exact_int(metrics["valid_pixels"], f"{label}.valid_pixels", minimum=1)
    values: dict[str, float] = {}
    for key, value in metrics.items():
        if key == "valid_pixels":
            continue
        number = finite_float(value, f"{label}.{key}")
        if key in ERROR_METRICS and number < 0:
            raise ValueError(f"{label}.{key} must be non-negative")
        if key.startswith("δ <") and not 0 <= number <= 1:
            raise ValueError(f"{label}.{key} must be in [0, 1]")
        values[key] = number
    return valid_pixels, values


def validate_aggregate_map(payload: Any, label: str) -> dict[str, float]:
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must be a JSON object")
    missing = set(METRIC_KEYS.values()) - set(payload)
    if missing:
        raise ValueError(f"{label} is missing metrics: {sorted(missing)}")
    values = {key: finite_float(value, f"{label}.{key}") for key, value in payload.items()}
    for key, number in values.items():
        if key in ERROR_METRICS and number < 0:
            raise ValueError(f"{label}.{key} must be non-negative")
        if key.startswith("δ <") and not 0 <= number <= 1:
            raise ValueError(f"{label}.{key} must be in [0, 1]")
    return values


def validate_method_metadata(
    payload: Mapping[str, Any], method: str, label: str
) -> tuple[str, int | None]:
    expected_policy, expected_window = METHOD_CONFIGS[method]
    policy = payload.get("cache_policy")
    window = normalize_window(payload.get("cache_window_size"), f"{label}.cache_window_size")
    if policy != expected_policy or window != expected_window:
        raise ValueError(
            f"method metadata mismatch in {label}: expected "
            f"{expected_policy}/K{expected_window}, got {policy}/K{window}"
        )
    return expected_policy, window


def require_summary_match(observed: Any, expected: Any, label: str) -> None:
    observed_number = finite_float(observed, label)
    if not close(observed_number, expected):
        raise ValueError(f"summary mismatch for {label}: {observed_number} != {expected}")


def relative_result_dir(dataset: str, method: str) -> str:
    return (Path(dataset) / method).as_posix()


def load_rows(
    results_root: Path,
    datasets: Sequence[str],
    methods: Sequence[str],
    *,
    allow_subset: bool,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    aggregate_output: list[dict[str, Any]] = []
    sequence_output: list[dict[str, Any]] = []
    pareto_inputs: list[dict[str, Any]] = []
    coverage: dict[tuple[str, str], set[str]] = {}
    frame_counts: dict[tuple[str, str, str], int] = {}
    provenance: set[tuple[str, str, str, str]] = set()
    run_scope = "debug_subset" if allow_subset else "frozen"

    for dataset in datasets:
        for method in methods:
            result_dir = results_root / dataset / method
            runtime_path = result_dir / "runtime_memory_rank0.json"
            aggregate_path = result_dir / "result_scale.json"
            sequence_path = result_dir / "result_scale_sequences.json"
            missing = [
                str(path)
                for path in (runtime_path, aggregate_path, sequence_path)
                if not path.is_file()
            ]
            if missing:
                raise FileNotFoundError("missing VideoDepth result(s):\n  " + "\n  ".join(missing))

            runtime_payload = load_json(runtime_path)
            aggregate_payload = validate_aggregate_map(
                load_json(aggregate_path), str(aggregate_path)
            )
            sequence_payload = load_json(sequence_path)
            summary = runtime_payload.get("summary")
            if not isinstance(summary, dict):
                raise ValueError(f"missing runtime summary in {runtime_path}")

            if sequence_payload.get("dataset") != dataset:
                raise ValueError(
                    f"dataset metadata mismatch in {sequence_path}: "
                    f"{sequence_payload.get('dataset')!r} != {dataset!r}"
                )
            if sequence_payload.get("align") != "scale":
                raise ValueError(
                    f"alignment metadata mismatch in {sequence_path}: "
                    f"expected 'scale', got {sequence_payload.get('align')!r}"
                )
            if summary.get("dataset") != dataset:
                raise ValueError(
                    f"runtime dataset mismatch in {runtime_path}: "
                    f"{summary.get('dataset')!r} != {dataset!r}"
                )
            if exact_int(summary.get("input_size"), f"{runtime_path}:summary.input_size") != 518:
                raise ValueError(f"VideoDepth input size must be 518 in {runtime_path}")
            if exact_int(
                summary.get("pose_eval_stride"),
                f"{runtime_path}:summary.pose_eval_stride",
                minimum=1,
            ) != 1:
                raise ValueError(f"VideoDepth pose_eval_stride must be 1 in {runtime_path}")
            requested_max_frames = summary.get("requested_max_frames")
            if requested_max_frames is not None:
                requested_max_frames = exact_int(
                    requested_max_frames,
                    f"{runtime_path}:summary.requested_max_frames",
                    minimum=1,
                )
            if not allow_subset and requested_max_frames is not None:
                raise ValueError(
                    f"formal VideoDepth result unexpectedly has requested_max_frames="
                    f"{requested_max_frames} in {runtime_path}"
                )

            expected_policy, expected_window = validate_method_metadata(
                summary, method, str(runtime_path)
            )
            runtime_sequences = unique_index(
                runtime_payload.get("sequences"), "seq", f"{runtime_path}:sequences"
            )
            metric_sequences = unique_index(
                sequence_payload.get("sequences"),
                "sequence",
                f"{sequence_path}:sequences",
            )
            non_ok = [
                name
                for name, item in runtime_sequences.items()
                if item.get("status") != "ok"
            ]
            if non_ok:
                raise ValueError(f"non-ok VideoDepth sequence(s) in {result_dir}: {non_ok}")
            if set(runtime_sequences) != set(metric_sequences):
                raise ValueError(
                    f"runtime/metric sequence mismatch in {result_dir}: "
                    f"runtime-only={sorted(set(runtime_sequences) - set(metric_sequences))}, "
                    f"metrics-only={sorted(set(metric_sequences) - set(runtime_sequences))}"
                )

            observed_count = len(metric_sequences)
            if observed_count == 0:
                raise ValueError(f"no VideoDepth sequences in {sequence_path}")
            if exact_int(
                sequence_payload.get("num_sequences"),
                f"{sequence_path}:num_sequences",
            ) != observed_count:
                raise ValueError(f"invalid sequence count in {sequence_path}")
            if not allow_subset and observed_count != EXPECTED_SEQUENCE_COUNTS[dataset]:
                raise ValueError(
                    f"{dataset}/{method}: expected {EXPECTED_SEQUENCE_COUNTS[dataset]} "
                    f"frozen sequences, found {observed_count}"
                )

            coverage[(dataset, method)] = set(metric_sequences)
            valid_pixel_weights: list[int] = []
            metric_values: list[dict[str, float]] = []
            result_source = relative_result_dir(dataset, method)
            for sequence in sorted(metric_sequences):
                metric_item = metric_sequences[sequence]
                runtime_item = runtime_sequences[sequence]
                validate_method_metadata(
                    runtime_item, method, f"{runtime_path}:{sequence}"
                )
                metric_frames = exact_int(
                    metric_item.get("num_frames"),
                    f"{sequence_path}:{sequence}.num_frames",
                    minimum=1,
                )
                runtime_frames = exact_int(
                    runtime_item.get("num_frames"),
                    f"{runtime_path}:{sequence}.num_frames",
                    minimum=1,
                )
                if metric_frames != runtime_frames:
                    raise ValueError(
                        f"frame-count mismatch in {result_dir}/{sequence}: "
                        f"metrics={metric_frames}, runtime={runtime_frames}"
                    )
                frame_counts[(dataset, method, sequence)] = metric_frames

                valid_pixels, raw_metrics = validate_metric_map(
                    metric_item.get("metrics"), f"{sequence_path}:{sequence}.metrics"
                )
                if metric_values and set(raw_metrics) != set(metric_values[0]):
                    raise ValueError(f"metric-key mismatch in {sequence_path}:{sequence}")
                valid_pixel_weights.append(valid_pixels)
                metric_values.append(raw_metrics)

                inference_sec = finite_float(
                    runtime_item.get("inference_sec"),
                    f"{runtime_path}:{sequence}.inference_sec",
                    minimum=0.0,
                )
                if inference_sec <= 0:
                    raise ValueError(f"{runtime_path}:{sequence}.inference_sec must be positive")
                fps = finite_float(
                    runtime_item.get("fps_inference"),
                    f"{runtime_path}:{sequence}.fps_inference",
                    minimum=0.0,
                )
                require_summary_match(
                    fps,
                    runtime_frames / inference_sec,
                    f"{runtime_path}:{sequence}.fps_inference",
                )
                allocated = finite_float(
                    runtime_item.get("peak_allocated_mb"),
                    f"{runtime_path}:{sequence}.peak_allocated_mb",
                    minimum=0.0,
                )
                reserved = finite_float(
                    runtime_item.get("peak_reserved_mb"),
                    f"{runtime_path}:{sequence}.peak_reserved_mb",
                    minimum=0.0,
                )
                row: dict[str, Any] = {
                    "run_scope": run_scope,
                    "dataset": dataset,
                    "sequence": sequence,
                    "method": method,
                    "num_frames": metric_frames,
                    "valid_pixels": valid_pixels,
                    "inference_sec": inference_sec,
                    "fps_inference": fps,
                    "peak_allocated_mb": allocated,
                    "peak_reserved_mb": reserved,
                    "gpu_name": summary.get("gpu_name"),
                    "torch_version": summary.get("torch_version"),
                    "cuda_version": summary.get("cuda_version"),
                    "python_version": summary.get("python_version"),
                    "slurm_job_id": summary.get("slurm_job_id"),
                    "hostname": summary.get("hostname"),
                    "input_size": summary.get("input_size"),
                    "pose_eval_stride": summary.get("pose_eval_stride"),
                    "requested_max_frames": requested_max_frames,
                    "source": result_source,
                }
                row.update(
                    {output_key: raw_metrics[input_key] for output_key, input_key in METRIC_KEYS.items()}
                )
                sequence_output.append(row)

            weighted_payload = validate_aggregate_map(
                sequence_payload.get("weighted_average"),
                f"{sequence_path}:weighted_average",
            )
            if set(weighted_payload) != set(aggregate_payload):
                raise ValueError(
                    f"aggregate metric-key mismatch in {result_dir}: "
                    f"sequence={sorted(weighted_payload)}, aggregate={sorted(aggregate_payload)}"
                )
            if set(weighted_payload) != set(metric_values[0]):
                raise ValueError(f"sequence/aggregate metric-key mismatch in {result_dir}")
            total_valid_pixels = sum(valid_pixel_weights)
            for key in weighted_payload:
                recomputed = sum(
                    metrics[key] * weight
                    for metrics, weight in zip(metric_values, valid_pixel_weights)
                ) / total_valid_pixels
                if not close(weighted_payload[key], recomputed):
                    raise ValueError(f"weighted aggregate mismatch in {result_dir}: {key}")
                if not close(aggregate_payload[key], recomputed):
                    raise ValueError(f"official aggregate mismatch in {result_dir}: {key}")

            summary_count = exact_int(
                summary.get("num_sequences"), f"{runtime_path}:summary.num_sequences"
            )
            summary_ok = exact_int(summary.get("num_ok"), f"{runtime_path}:summary.num_ok")
            summary_oom = exact_int(summary.get("num_oom"), f"{runtime_path}:summary.num_oom")
            if (summary_count, summary_ok, summary_oom) != (
                len(runtime_sequences),
                len(runtime_sequences),
                0,
            ):
                raise ValueError(
                    f"runtime summary counts mismatch in {runtime_path}: "
                    f"got {(summary_count, summary_ok, summary_oom)}"
                )
            total_frames = sum(
                exact_int(item["num_frames"], f"{runtime_path}:{name}.num_frames", minimum=1)
                for name, item in runtime_sequences.items()
            )
            if exact_int(
                summary.get("total_frames"), f"{runtime_path}:summary.total_frames"
            ) != total_frames:
                raise ValueError(f"runtime total_frames mismatch in {runtime_path}")
            total_inference_sec = sum(
                finite_float(item["inference_sec"], f"{runtime_path}:{name}.inference_sec")
                for name, item in runtime_sequences.items()
            )
            require_summary_match(
                summary.get("total_inference_sec"),
                total_inference_sec,
                f"{runtime_path}:summary.total_inference_sec",
            )
            fps_inference = total_frames / total_inference_sec
            require_summary_match(
                summary.get("fps_inference"),
                fps_inference,
                f"{runtime_path}:summary.fps_inference",
            )
            max_allocated = max(float(item["peak_allocated_mb"]) for item in runtime_sequences.values())
            max_reserved = max(float(item["peak_reserved_mb"]) for item in runtime_sequences.values())
            require_summary_match(
                summary.get("max_peak_allocated_mb"),
                max_allocated,
                f"{runtime_path}:summary.max_peak_allocated_mb",
            )
            require_summary_match(
                summary.get("max_peak_reserved_mb"),
                max_reserved,
                f"{runtime_path}:summary.max_peak_reserved_mb",
            )

            provenance_values: list[str] = []
            for key in ("gpu_name", "torch_version", "cuda_version", "python_version"):
                value = summary.get(key)
                if not isinstance(value, str) or not value.strip():
                    raise ValueError(f"missing provenance field {key!r} in {runtime_path}")
                provenance_values.append(value)
            provenance.add(tuple(provenance_values))

            aggregate_row: dict[str, Any] = {
                "run_scope": run_scope,
                "dataset": dataset,
                "method": method,
                "cache_policy": expected_policy,
                "cache_window_size": expected_window,
                "gpu_name": provenance_values[0],
                "torch_version": provenance_values[1],
                "cuda_version": provenance_values[2],
                "python_version": provenance_values[3],
                "slurm_job_id": summary.get("slurm_job_id"),
                "hostname": summary.get("hostname"),
                "input_size": summary.get("input_size"),
                "pose_eval_stride": summary.get("pose_eval_stride"),
                "requested_max_frames": requested_max_frames,
                "num_sequences": summary_count,
                "num_ok": summary_ok,
                "num_oom": summary_oom,
                "total_frames": total_frames,
                "total_inference_sec": total_inference_sec,
                "fps_inference": fps_inference,
                "max_peak_allocated_mb": max_allocated,
                "max_peak_reserved_mb": max_reserved,
                "result_dir": result_source,
            }
            aggregate_row.update(
                {output_key: aggregate_payload[input_key] for output_key, input_key in METRIC_KEYS.items()}
            )
            aggregate_output.append(aggregate_row)
            pareto_inputs.append(
                {
                    "run_scope": run_scope,
                    "dataset": dataset,
                    "method": method,
                    "abs_rel": aggregate_row["abs_rel"],
                    "delta_1": aggregate_row["delta_1"],
                    "total_inference_sec": total_inference_sec,
                    "fps_inference": fps_inference,
                    "max_peak_allocated_mb": max_allocated,
                    "max_peak_reserved_mb": max_reserved,
                    "source": result_source,
                }
            )

    if len(provenance) != 1:
        raise ValueError(f"VideoDepth provenance differs across cells: {sorted(provenance)}")

    for dataset in datasets:
        reference_method = methods[0]
        reference_sequences = coverage[(dataset, reference_method)]
        for method in methods[1:]:
            if coverage[(dataset, method)] != reference_sequences:
                raise ValueError(
                    f"{dataset}: sequence coverage differs between {reference_method} and {method}"
                )
        for sequence in reference_sequences:
            counts = {frame_counts[(dataset, method, sequence)] for method in methods}
            if len(counts) != 1:
                raise ValueError(
                    f"{dataset}/{sequence}: methods used different frame counts: {sorted(counts)}"
                )

        frozen = FROZEN_SEQUENCE_FRAMES[dataset]
        unknown = reference_sequences - set(frozen)
        if unknown:
            raise ValueError(f"{dataset}: unknown sequence(s): {sorted(unknown)}")
        if not allow_subset and reference_sequences != set(frozen):
            raise ValueError(
                f"{dataset}: frozen sequence coverage mismatch: "
                f"missing={sorted(set(frozen) - reference_sequences)}, "
                f"extra={sorted(reference_sequences - set(frozen))}"
            )
        for sequence in reference_sequences:
            observed_frames = frame_counts[(dataset, reference_method, sequence)]
            expected_frames = frozen[sequence]
            if allow_subset:
                if observed_frames > expected_frames:
                    raise ValueError(
                        f"{dataset}/{sequence}: debug/subset run has {observed_frames} frames, "
                        f"above the frozen maximum {expected_frames}"
                    )
            elif observed_frames != expected_frames:
                raise ValueError(
                    f"{dataset}/{sequence}: expected {expected_frames} frozen frames, "
                    f"found {observed_frames}"
                )

    sequence_output.sort(
        key=lambda row: (
            DATASET_INDEX[row["dataset"]],
            row["sequence"],
            METHOD_INDEX[row["method"]],
        )
    )
    return aggregate_output, sequence_output, pareto_inputs


def bootstrap_mean_ci(
    values: Iterable[float], rng: np.random.Generator, samples: int
) -> tuple[float, float]:
    array = np.asarray(tuple(values), dtype=np.float64)
    if len(array) == 1:
        return float(array[0]), float(array[0])
    indices = rng.integers(0, len(array), size=(samples, len(array)))
    means = array[indices].mean(axis=1)
    low, high = np.percentile(means, (2.5, 97.5))
    return float(low), float(high)


def build_statistics(
    sequence_rows: Sequence[Mapping[str, Any]],
    datasets: Sequence[str],
    methods: Sequence[str],
    rng: np.random.Generator,
    bootstrap_samples: int,
) -> list[dict[str, Any]]:
    outputs = []
    for dataset in datasets:
        for method in methods:
            rows = [
                row
                for row in sequence_rows
                if row["dataset"] == dataset and row["method"] == method
            ]
            for metric in STATS_METRICS:
                values = np.asarray([float(row[metric]) for row in rows], dtype=np.float64)
                ci_low, ci_high = bootstrap_mean_ci(values, rng, bootstrap_samples)
                outputs.append(
                    {
                        "run_scope": rows[0]["run_scope"],
                        "dataset": dataset,
                        "method": method,
                        "metric": metric,
                        "n_sequences": len(values),
                        "mean": float(np.mean(values)),
                        "median": float(np.median(values)),
                        "std": float(np.std(values, ddof=1)) if len(values) >= 2 else 0.0,
                        "ci95_low": ci_low,
                        "ci95_high": ci_high,
                        "aggregation": "unweighted_sequence_mean",
                    }
                )
    return outputs


def build_paired_comparisons(
    sequence_rows: Sequence[Mapping[str, Any]],
    datasets: Sequence[str],
    methods: Sequence[str],
    rng: np.random.Generator,
    bootstrap_samples: int,
) -> list[dict[str, Any]]:
    indexed = {
        (row["dataset"], row["sequence"], row["method"]): row for row in sequence_rows
    }
    outputs = []
    for dataset in datasets:
        sequences = sorted(
            {row["sequence"] for row in sequence_rows if row["dataset"] == dataset}
        )
        for method_a, method_b in itertools.combinations(methods, 2):
            for metric, direction in PAIRED_METRICS.items():
                values_a = np.asarray(
                    [float(indexed[(dataset, sequence, method_a)][metric]) for sequence in sequences]
                )
                values_b = np.asarray(
                    [float(indexed[(dataset, sequence, method_b)][metric]) for sequence in sequences]
                )
                advantage = values_b - values_a if direction == "lower" else values_a - values_b
                ci_low, ci_high = bootstrap_mean_ci(advantage, rng, bootstrap_samples)
                wins = sum(
                    not close(a, b)
                    and ((a < b) if direction == "lower" else (a > b))
                    for a, b in zip(values_a, values_b)
                )
                ties = sum(close(a, b) for a, b in zip(values_a, values_b))
                losses = len(sequences) - wins - ties
                significance = (
                    "INSUFFICIENT_PAIRS"
                    if len(sequences) < 2
                    else "A_BETTER"
                    if ci_low > 0
                    else "B_BETTER"
                    if ci_high < 0
                    else "NO_CLEAR_DIFFERENCE"
                )
                outputs.append(
                    {
                        "run_scope": indexed[(dataset, sequences[0], method_a)]["run_scope"],
                        "dataset": dataset,
                        "metric": metric,
                        "better_direction": direction,
                        "method_a": method_a,
                        "method_b": method_b,
                        "n_pairs": len(sequences),
                        "mean_a": float(np.mean(values_a)),
                        "mean_b": float(np.mean(values_b)),
                        "mean_advantage_a": float(np.mean(advantage)),
                        "advantage_ci95_low": ci_low,
                        "advantage_ci95_high": ci_high,
                        "wins_a": wins,
                        "ties": ties,
                        "losses_a": losses,
                        "significance": significance,
                    }
                )
    return outputs


def build_regret(
    sequence_rows: Sequence[Mapping[str, Any]],
    datasets: Sequence[str],
    methods: Sequence[str],
) -> list[dict[str, Any]]:
    indexed = {
        (row["dataset"], row["sequence"], row["method"]): row for row in sequence_rows
    }
    outputs = []
    dataset_groups = [*((dataset, (dataset,)) for dataset in datasets), ("all", tuple(datasets))]
    for dataset_label, group in dataset_groups:
        sequence_keys = sorted(
            {
                (row["dataset"], row["sequence"])
                for row in sequence_rows
                if row["dataset"] in group
            }
        )
        for method in methods:
            for metric, direction in REGRET_METRICS.items():
                regrets = []
                wins = 0
                for dataset, sequence in sequence_keys:
                    values = {
                        candidate: float(indexed[(dataset, sequence, candidate)][metric])
                        for candidate in methods
                    }
                    oracle = min(values.values()) if direction == "lower" else max(values.values())
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
                        "run_scope": indexed[(sequence_keys[0][0], sequence_keys[0][1], method)]["run_scope"],
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


def build_pareto(
    aggregate_rows: Sequence[Mapping[str, Any]],
    datasets: Sequence[str],
) -> list[dict[str, Any]]:
    outputs = []
    for dataset in datasets:
        rows = [row for row in aggregate_rows if row["dataset"] == dataset]
        for row in rows:
            dominated_by = []
            for other in rows:
                if other["method"] == row["method"]:
                    continue
                weakly_better = (
                    other["abs_rel"] <= row["abs_rel"]
                    and other["max_peak_allocated_mb"] <= row["max_peak_allocated_mb"]
                    and other["total_inference_sec"] <= row["total_inference_sec"]
                )
                strictly_better = (
                    other["abs_rel"] < row["abs_rel"]
                    or other["max_peak_allocated_mb"] < row["max_peak_allocated_mb"]
                    or other["total_inference_sec"] < row["total_inference_sec"]
                )
                if weakly_better and strictly_better:
                    dominated_by.append(other["method"])
            output = dict(row)
            output["pareto_absrel_allocated_time"] = "yes" if not dominated_by else "no"
            output["dominated_by"] = " ".join(sorted(dominated_by, key=METHOD_INDEX.get))
            outputs.append(output)
    return outputs


def write_csv(path: Path, fields: Sequence[str], rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} rows to {path}")


def main() -> None:
    args = parse_args()
    if args.bootstrap_samples < 1000:
        raise ValueError("--bootstrap-samples must be at least 1000")
    datasets = ordered_selection(args.datasets, DATASET_ORDER, "datasets")
    methods = ordered_selection(args.methods, METHOD_ORDER, "methods")
    aggregate_rows, sequence_rows, pareto_inputs = load_rows(
        args.results_root,
        datasets,
        methods,
        allow_subset=args.allow_subset,
    )
    rng = np.random.default_rng(args.seed)
    statistics = build_statistics(
        sequence_rows, datasets, methods, rng, args.bootstrap_samples
    )
    paired = build_paired_comparisons(
        sequence_rows, datasets, methods, rng, args.bootstrap_samples
    )
    for row in statistics:
        row["bootstrap_samples"] = args.bootstrap_samples
        row["bootstrap_seed"] = args.seed
    for row in paired:
        row["bootstrap_samples"] = args.bootstrap_samples
        row["bootstrap_seed"] = args.seed
    regret = build_regret(sequence_rows, datasets, methods)
    pareto = build_pareto(pareto_inputs, datasets)

    output_dir = args.output.parent
    write_csv(args.output, AGGREGATE_FIELDS, aggregate_rows)
    write_csv(output_dir / "video_depth_sequence_results.csv", SEQUENCE_FIELDS, sequence_rows)
    write_csv(output_dir / "video_depth_paired_bootstrap.csv", PAIRED_FIELDS, paired)
    write_csv(output_dir / "video_depth_sequence_statistics.csv", STAT_FIELDS, statistics)
    write_csv(output_dir / "video_depth_regret.csv", REGRET_FIELDS, regret)
    write_csv(output_dir / "video_depth_pareto.csv", PARETO_FIELDS, pareto)


if __name__ == "__main__":
    main()
