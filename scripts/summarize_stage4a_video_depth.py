#!/usr/bin/env python3
"""Summarize the same-GPU Stage 4A VideoDepth matrix."""

import argparse
import csv
import glob
import json
import os


FIELDS = (
    "dataset",
    "method",
    "stage",
    "cache_policy",
    "cache_window_size",
    "gpu_name",
    "torch_version",
    "cuda_version",
    "slurm_job_id",
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
    "source",
)
METHOD_ORDER = {
    "full_cache": 0,
    "stage3_2_k4": 1,
    "old_dino_k6": 2,
    "temporal_binned_dino_k8": 3,
}

DATASETS = ("bonn", "kitti", "sintel")
METHODS = (
    "full_cache",
    "stage3_2_k4",
    "old_dino_k6",
    "temporal_binned_dino_k8",
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


def main():
    parser = argparse.ArgumentParser("Summarize Stage 4A VideoDepth")
    parser.add_argument("--results-root", default="eval_results/video_depth")
    parser.add_argument("--output", default="stage4a_video_depth_results.csv")
    args = parser.parse_args()

    rows = []
    pattern = os.path.join(args.results_root, "*stage4a_*", "runtime_memory_rank0.json")
    for runtime_path in sorted(glob.glob(pattern)):
        result_dir = os.path.dirname(runtime_path)
        dirname = os.path.basename(result_dir)
        dataset = dataset_from_directory(dirname)
        method = method_from_directory(dirname)
        with open(runtime_path) as handle:
            runtime = json.load(handle)["summary"]
        metrics_path = os.path.join(result_dir, "result_scale.json")
        if not os.path.isfile(metrics_path):
            raise FileNotFoundError(f"missing VideoDepth metrics: {metrics_path}")
        with open(metrics_path) as handle:
            metrics = json.load(handle)
        row = {field: "" for field in FIELDS}
        row.update(
            dataset=dataset,
            method=method,
            stage="stage4a",
            cache_policy=runtime.get("cache_policy"),
            cache_window_size=runtime.get("cache_window_size"),
            gpu_name=runtime.get("gpu_name"),
            torch_version=runtime.get("torch_version"),
            cuda_version=runtime.get("cuda_version"),
            slurm_job_id=runtime.get("slurm_job_id"),
            num_sequences=runtime.get("num_sequences"),
            num_ok=runtime.get("num_ok"),
            num_oom=runtime.get("num_oom"),
            total_frames=runtime.get("total_frames"),
            total_inference_sec=runtime.get("total_inference_sec"),
            fps_inference=runtime.get("fps_inference"),
            max_peak_allocated_mb=runtime.get("max_peak_allocated_mb"),
            max_peak_reserved_mb=runtime.get("max_peak_reserved_mb"),
            abs_rel=metrics.get("Abs Rel"),
            sq_rel=metrics.get("Sq Rel"),
            rmse=metrics.get("RMSE"),
            log_rmse=metrics.get("Log RMSE"),
            delta_1=metrics.get("δ < 1.25"),
            delta_2=metrics.get("δ < 1.25^2"),
            delta_3=metrics.get("δ < 1.25^3"),
            source=result_dir,
        )
        rows.append(row)

    seen = set()
    for row in rows:
        key = (row["dataset"], row["method"])
        if key in seen:
            raise RuntimeError(f"duplicate Stage 4A row: {key}")
        seen.add(key)
    expected = {(dataset, method) for dataset in DATASETS for method in METHODS}
    if seen != expected:
        missing = sorted(expected - seen)
        unexpected = sorted(seen - expected)
        raise RuntimeError(
            f"incomplete Stage 4A matrix: missing={missing}, unexpected={unexpected}"
        )
    rows.sort(key=lambda row: (row["dataset"], METHOD_ORDER[row["method"]]))

    with open(args.output, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} VideoDepth rows to {args.output}")


if __name__ == "__main__":
    main()
