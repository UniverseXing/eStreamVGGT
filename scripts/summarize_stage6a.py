#!/usr/bin/env python3
"""Validate and summarize the frozen Stage 6A journal experiment matrix."""

from __future__ import annotations

import argparse
import csv
import glob
import json
from pathlib import Path

import numpy as np


SAME_METHODS = (
    "full_cache",
    "recent4",
    "anchor_recent4",
    "anchor_uniform4",
    "random4_seed0",
    "random4_seed1",
    "random4_seed2",
    "dino_only4",
    "proposed_k4",
)
COMPONENT_METHODS = ("proposed_k6", "no_recent_k6")
CONFIGS = {
    "full_cache": ("full_cache", None, 0),
    "recent4": ("fifo", 4, 0),
    "anchor_recent4": ("anchor_recent", 4, 0),
    "anchor_uniform4": ("anchor_uniform_k4", 4, 0),
    "random4_seed0": ("random_reservoir_k4", 4, 0),
    "random4_seed1": ("random_reservoir_k4", 4, 1),
    "random4_seed2": ("random_reservoir_k4", 4, 2),
    "dino_only4": ("dino_diverse_no_anchor_k4", 4, 0),
    "proposed_k4": ("anchor_recent_dino_diverse_k4", 4, 0),
    "proposed_k6": ("anchor_recent_dino_diverse_k6", 6, 0),
    "no_recent_k6": ("anchor_dino_diverse_no_recent_k6", 6, 0),
}
VD_DATASETS = ("bonn", "sintel", "kitti")
TASKS = ("video_depth", "pose", "reconstruction")
METRIC_FIELDS = (
    "abs_rel", "rmse", "delta_1", "ate", "rpe_trans", "rpe_rot_deg",
    "overall", "acc", "comp", "nc",
)
COMMON_FIELDS = (
    "task", "dataset", "method", "random_seed", "cache_policy",
    "cache_window_size", "num_sequences", "num_successful", "num_failed",
    "total_frames", *METRIC_FIELDS, "fps_inference", "peak_allocated_mb",
    "peak_reserved_mb", "gpu_name", "torch_version", "cuda_version",
    "slurm_job_id", "source",
)
SEQUENCE_FIELDS = (
    "task", "dataset", "sequence", "method", "random_seed", "num_frames",
    *METRIC_FIELDS, "inference_sec", "fps_inference", "peak_allocated_mb",
    "peak_reserved_mb", "source",
)
PAIRED_FIELDS = (
    "task", "dataset", "metric", "proposed", "control", "n_pairs",
    "mean_proposed", "mean_control", "mean_advantage_proposed",
    "ci95_low", "ci95_high", "wins_proposed", "ties", "losses_proposed",
    "bootstrap_samples",
)


def read_json(path: Path):
    with path.open() as handle:
        return json.load(handle)


def write_csv(path: Path, fields, rows):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} rows to {path}")


def one_path(pattern: str, label: str, allow_incomplete: bool):
    paths = [Path(path) for path in sorted(glob.glob(pattern))]
    if len(paths) > 1:
        raise RuntimeError(f"duplicate {label}: {[str(path) for path in paths]}")
    if not paths:
        if allow_incomplete:
            print(f"Missing (allowed): {label}")
            return None
        raise FileNotFoundError(f"missing {label}: {pattern}")
    return paths[0]


def normalized_window(value):
    return None if value in (None, "") else int(value)


def validate_config(metadata, method, path):
    expected_policy, expected_window, expected_seed = CONFIGS[method]
    observed = (
        metadata.get("cache_policy"),
        normalized_window(metadata.get("cache_window_size")),
        int(metadata.get("cache_random_seed", 0)),
    )
    expected = (expected_policy, expected_window, expected_seed)
    if observed != expected:
        raise RuntimeError(f"configuration mismatch in {path}: {observed} != {expected}")


def blank_metrics():
    return {field: "" for field in METRIC_FIELDS}


def load_video_depth(
    root: Path,
    dataset: str,
    method: str,
    allow_incomplete: bool,
    stage_tag: str = "stage6a",
):
    pattern = str(root / "eval_results/video_depth" / f"{dataset}_streamvggt*{stage_tag}_{method}" / "runtime_memory_rank0.json")
    runtime_path = one_path(pattern, f"VideoDepth {dataset}/{method}", allow_incomplete)
    if runtime_path is None:
        return None, []
    result_dir = runtime_path.parent
    runtime = read_json(runtime_path)
    summary = runtime["summary"]
    validate_config(summary, method, runtime_path)
    aggregate = read_json(result_dir / "result_scale.json")
    sequence_metrics = read_json(result_dir / "result_scale_sequences.json")
    runtime_by_seq = {item["seq"]: item for item in runtime["sequences"]}
    metrics_by_seq = {item["sequence"]: item for item in sequence_metrics["sequences"]}
    if set(runtime_by_seq) != set(metrics_by_seq):
        raise RuntimeError(f"runtime/metric coverage mismatch in {result_dir}")
    failed = [item for item in runtime["sequences"] if item.get("status") != "ok"]
    if failed:
        raise RuntimeError(f"non-ok VideoDepth sequence(s) in {result_dir}")
    _, _, seed = CONFIGS[method]
    row = {
        **{field: "" for field in COMMON_FIELDS}, **blank_metrics(),
        "task": "video_depth", "dataset": dataset, "method": method,
        "random_seed": seed, "cache_policy": summary["cache_policy"],
        "cache_window_size": normalized_window(summary["cache_window_size"]),
        "num_sequences": summary["num_sequences"], "num_successful": summary["num_ok"],
        "num_failed": summary["num_oom"], "total_frames": summary["total_frames"],
        "abs_rel": aggregate["Abs Rel"], "rmse": aggregate["RMSE"],
        "delta_1": aggregate["δ < 1.25"], "fps_inference": summary["fps_inference"],
        "peak_allocated_mb": summary["max_peak_allocated_mb"],
        "peak_reserved_mb": summary["max_peak_reserved_mb"],
        "gpu_name": summary.get("gpu_name", ""), "torch_version": summary.get("torch_version", ""),
        "cuda_version": summary.get("cuda_version", ""), "slurm_job_id": summary.get("slurm_job_id", ""),
        "source": str(result_dir),
    }
    sequence_rows = []
    for sequence in sorted(runtime_by_seq):
        run = runtime_by_seq[sequence]
        metric = metrics_by_seq[sequence]
        values = metric["metrics"]
        if int(run["num_frames"]) != int(metric["num_frames"]):
            raise RuntimeError(f"frame mismatch in {result_dir}/{sequence}")
        sequence_rows.append({
            **{field: "" for field in SEQUENCE_FIELDS}, **blank_metrics(),
            "task": "video_depth", "dataset": dataset, "sequence": sequence,
            "method": method, "random_seed": seed, "num_frames": run["num_frames"],
            "abs_rel": values["Abs Rel"], "rmse": values["RMSE"],
            "delta_1": values["δ < 1.25"], "inference_sec": run["inference_sec"],
            "fps_inference": run["fps_inference"], "peak_allocated_mb": run["peak_allocated_mb"],
            "peak_reserved_mb": run["peak_reserved_mb"], "source": str(result_dir),
        })
    return row, sequence_rows


def load_pose(root: Path, method: str, allow_incomplete: bool):
    pattern = str(root / "eval_results/pose" / f"tum_streamvggt_*stage6a_{method}" / "pose_metrics.json")
    path = one_path(pattern, f"TUM pose/{method}", allow_incomplete)
    if path is None:
        return None, []
    payload = read_json(path)
    summary = payload["summary"]
    validate_config(summary, method, path)
    failed = [item for item in payload["sequences"] if item.get("status") != "ok"]
    if failed:
        raise RuntimeError(f"non-ok pose sequence(s) in {path}")
    _, _, seed = CONFIGS[method]
    row = {
        **{field: "" for field in COMMON_FIELDS}, **blank_metrics(),
        "task": "pose", "dataset": "tum", "method": method, "random_seed": seed,
        "cache_policy": summary["cache_policy"], "cache_window_size": summary["cache_window_size"],
        "num_sequences": summary["num_sequences"], "num_successful": summary["num_successful"],
        "num_failed": summary["num_failed"], "total_frames": summary["total_frames"],
        "ate": summary["mean_ate"], "rpe_trans": summary["mean_rpe_trans"],
        "rpe_rot_deg": summary["mean_rpe_rot_deg"], "fps_inference": summary["fps_inference"],
        "peak_allocated_mb": summary["max_peak_allocated_mb"],
        "peak_reserved_mb": summary["max_peak_reserved_mb"],
        "gpu_name": summary.get("gpu_name", ""), "torch_version": summary.get("torch_version", ""),
        "cuda_version": summary.get("cuda_version", ""), "slurm_job_id": summary.get("slurm_job_id", ""),
        "source": str(path.parent),
    }
    rows = []
    for item in payload["sequences"]:
        rows.append({
            **{field: "" for field in SEQUENCE_FIELDS}, **blank_metrics(),
            "task": "pose", "dataset": "tum", "sequence": item["sequence"],
            "method": method, "random_seed": seed, "num_frames": item["num_frames"],
            "ate": item["ate"], "rpe_trans": item["rpe_trans"],
            "rpe_rot_deg": item["rpe_rot_deg"], "inference_sec": item["inference_sec"],
            "fps_inference": item["fps_inference"], "peak_allocated_mb": item["peak_allocated_mb"],
            "peak_reserved_mb": item["peak_reserved_mb"], "source": str(path.parent),
        })
    return row, rows


def load_reconstruction(root: Path, method: str, allow_incomplete: bool):
    pattern = str(root / "eval_results/mv_recon" / f"streamvggt_stage6a_{method}" / "reconstruction_metrics.json")
    path = one_path(pattern, f"TUM reconstruction/{method}", allow_incomplete)
    if path is None:
        return None, []
    payload = read_json(path)
    validate_config(payload, method, path)
    dataset = payload["datasets"]["tum"]
    summary = dataset["summary"]
    failed = [item for item in dataset["sequences"] if item.get("status") != "ok"]
    if failed:
        raise RuntimeError(f"non-ok reconstruction sequence(s) in {path}")
    _, _, seed = CONFIGS[method]
    row = {
        **{field: "" for field in COMMON_FIELDS}, **blank_metrics(),
        "task": "reconstruction", "dataset": "tum_dynamics", "method": method,
        "random_seed": seed, "cache_policy": payload["cache_policy"],
        "cache_window_size": payload["cache_window_size"], "num_sequences": summary["num_sequences"],
        "num_successful": summary["num_successful"], "num_failed": summary["num_failed"],
        "total_frames": summary["total_frames"], "ate": summary["mean_ate"],
        "rpe_trans": summary["mean_rpe_trans"], "rpe_rot_deg": summary["mean_rpe_rot_deg"],
        "overall": summary["mean_overall"], "acc": summary["mean_acc"],
        "comp": summary["mean_comp"], "nc": summary["mean_nc"],
        "fps_inference": summary["fps_inference"], "peak_allocated_mb": summary["max_peak_allocated_mb"],
        "peak_reserved_mb": summary["max_peak_reserved_mb"], "gpu_name": payload.get("gpu_name", ""),
        "torch_version": payload.get("torch_version", ""), "cuda_version": payload.get("cuda_version", ""),
        "slurm_job_id": payload.get("slurm_job_id", ""), "source": str(path.parent),
    }
    rows = []
    for item in dataset["sequences"]:
        rows.append({
            **{field: "" for field in SEQUENCE_FIELDS}, **blank_metrics(),
            "task": "reconstruction", "dataset": "tum_dynamics", "sequence": item["sequence"],
            "method": method, "random_seed": seed, "num_frames": item["num_frames"],
            "ate": item.get("ate", ""), "rpe_trans": item.get("rpe_trans", ""),
            "rpe_rot_deg": item.get("rpe_rot_deg", ""), "overall": item["overall"],
            "acc": item["acc"], "comp": item["comp"], "nc": item["nc"],
            "inference_sec": item["inference_sec"], "fps_inference": item["fps_inference"],
            "peak_allocated_mb": item["peak_allocated_mb"],
            "peak_reserved_mb": item["peak_reserved_mb"], "source": str(path.parent),
        })
    return row, rows


def verify_coverage(rows, methods, allow_incomplete):
    groups = {}
    for row in rows:
        groups.setdefault((row["task"], row["dataset"], row["method"]), set()).add(
            (row["sequence"], int(row["num_frames"]))
        )
    for task, dataset in sorted({(row["task"], row["dataset"]) for row in rows}):
        available = [method for method in methods if (task, dataset, method) in groups]
        if not allow_incomplete and set(available) != set(methods):
            raise RuntimeError(f"incomplete method coverage for {task}/{dataset}: {available}")
        if available:
            reference = groups[(task, dataset, available[0])]
            for method in available[1:]:
                if groups[(task, dataset, method)] != reference:
                    raise RuntimeError(f"sequence/frame coverage differs for {task}/{dataset}/{method}")


def averaged_method_values(rows, task, dataset, method, metric):
    selected = [row for row in rows if row["task"] == task and row["dataset"] == dataset]
    if method == "random4_mean":
        seeds = [name for name in SAME_METHODS if name.startswith("random4_seed")]
        by_sequence = {}
        for row in selected:
            if row["method"] in seeds:
                by_sequence.setdefault(row["sequence"], []).append(float(row[metric]))
        if not by_sequence or any(len(values) != 3 for values in by_sequence.values()):
            return {}
        return {sequence: float(np.mean(values)) for sequence, values in by_sequence.items()}
    return {row["sequence"]: float(row[metric]) for row in selected if row["method"] == method}


def paired_rows(sequence_rows, samples):
    rng = np.random.default_rng(20260819)
    outputs = []
    comparisons = [
        ("proposed_k4", control)
        for control in ("recent4", "anchor_recent4", "anchor_uniform4", "random4_mean", "dino_only4")
    ] + [("proposed_k6", "no_recent_k6")]
    primary = {"video_depth": "abs_rel", "pose": "ate", "reconstruction": "overall"}
    for task, dataset in sorted({(row["task"], row["dataset"]) for row in sequence_rows}):
        metric = primary[task]
        for proposed, control in comparisons:
            a = averaged_method_values(sequence_rows, task, dataset, proposed, metric)
            b = averaged_method_values(sequence_rows, task, dataset, control, metric)
            if not a or set(a) != set(b):
                continue
            names = sorted(a)
            values_a = np.asarray([a[name] for name in names])
            values_b = np.asarray([b[name] for name in names])
            advantage = values_b - values_a
            indices = rng.integers(0, len(names), size=(samples, len(names)))
            boot = advantage[indices].mean(axis=1)
            outputs.append({
                "task": task, "dataset": dataset, "metric": metric,
                "proposed": proposed, "control": control, "n_pairs": len(names),
                "mean_proposed": float(values_a.mean()), "mean_control": float(values_b.mean()),
                "mean_advantage_proposed": float(advantage.mean()),
                "ci95_low": float(np.percentile(boot, 2.5)),
                "ci95_high": float(np.percentile(boot, 97.5)),
                "wins_proposed": int(np.sum(advantage > 1e-12)),
                "ties": int(np.sum(np.abs(advantage) <= 1e-12)),
                "losses_proposed": int(np.sum(advantage < -1e-12)),
                "bootstrap_samples": samples,
            })
    return outputs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--allow-incomplete", action="store_true")
    args = parser.parse_args()
    root = args.repo_root.resolve()
    aggregate_rows, sequence_rows = [], []
    for method in (*SAME_METHODS, *COMPONENT_METHODS):
        datasets = VD_DATASETS if method in SAME_METHODS else ("kitti",)
        for dataset in datasets:
            aggregate, sequences = load_video_depth(root, dataset, method, args.allow_incomplete)
            if aggregate:
                aggregate_rows.append(aggregate); sequence_rows.extend(sequences)
        for loader in (load_pose, load_reconstruction):
            aggregate, sequences = loader(root, method, args.allow_incomplete)
            if aggregate:
                aggregate_rows.append(aggregate); sequence_rows.extend(sequences)
    same_sequence = [row for row in sequence_rows if row["method"] in SAME_METHODS]
    component_sequence = [row for row in sequence_rows if row["method"] in COMPONENT_METHODS]
    verify_coverage(same_sequence, SAME_METHODS, args.allow_incomplete)
    verify_coverage(component_sequence, COMPONENT_METHODS, args.allow_incomplete)
    if not args.allow_incomplete:
        expected_counts = {
            ("video_depth", "bonn"): 5,
            ("video_depth", "sintel"): 23,
            ("video_depth", "kitti"): 13,
            ("pose", "tum"): 8,
            ("reconstruction", "tum_dynamics"): 8,
        }
        for row in aggregate_rows:
            expected = expected_counts[(row["task"], row["dataset"])]
            if (
                int(row["num_sequences"]) != expected
                or int(row["num_successful"]) != expected
                or int(row["num_failed"]) != 0
            ):
                raise RuntimeError(
                    f"incomplete formal coverage for {row['task']}/{row['dataset']}/"
                    f"{row['method']}: expected {expected} successful sequences"
                )
    gpu_names = {row["gpu_name"] for row in aggregate_rows if row["gpu_name"]}
    if len(gpu_names) > 1:
        raise RuntimeError(f"Stage 6A mixes GPU models: {sorted(gpu_names)}")
    if not args.allow_incomplete:
        if len(gpu_names) != 1 or "6000 ada" not in next(iter(gpu_names)).lower():
            raise RuntimeError(f"formal Stage 6A requires one RTX 6000 Ada GPU model: {gpu_names}")
        for field in ("torch_version", "cuda_version"):
            values = {str(row[field]) for row in aggregate_rows if row[field] != ""}
            if len(values) != 1:
                raise RuntimeError(f"Stage 6A mixes {field}: {sorted(values)}")
    same_aggregate = [row for row in aggregate_rows if row["method"] in SAME_METHODS]
    component_aggregate = [row for row in aggregate_rows if row["method"] in COMPONENT_METHODS]
    order = {method: index for index, method in enumerate((*SAME_METHODS, *COMPONENT_METHODS))}
    aggregate_key = lambda row: (row["task"], row["dataset"], order[row["method"]])
    sequence_key = lambda row: (row["task"], row["dataset"], row["sequence"], order[row["method"]])
    write_csv(root / "stage6a_same_budget_results.csv", COMMON_FIELDS, sorted(same_aggregate, key=aggregate_key))
    write_csv(root / "stage6a_same_budget_sequence_results.csv", SEQUENCE_FIELDS, sorted(same_sequence, key=sequence_key))
    write_csv(root / "stage6a_component_results.csv", COMMON_FIELDS, sorted(component_aggregate, key=aggregate_key))
    write_csv(root / "stage6a_paired_statistics.csv", PAIRED_FIELDS, paired_rows(sequence_rows, args.bootstrap_samples))


if __name__ == "__main__":
    main()
