#!/usr/bin/env python3
"""Merge reused Stage 3.3 baselines with the incremental Stage 3.7 K8 runs."""

import argparse
import csv


METHOD_ORDER = {
    "full_cache": 0,
    "stage3_2_k4": 1,
    "old_dino_k6": 2,
    "temporal_binned_dino_k8": 3,
}
TASK_ORDER = {"pose": 0, "static_recon": 1, "dynamic_recon": 2}
FIELDS = (
    "task",
    "dataset",
    "method",
    "cache_policy",
    "cache_window_size",
    "num_sequences",
    "num_successful",
    "num_failed",
    "total_frames",
    "mean_ate",
    "mean_rpe_trans",
    "mean_rpe_rot_deg",
    "mean_overall",
    "mean_nc",
    "fps_inference",
    "max_peak_allocated_mb",
    "max_peak_reserved_mb",
    "source",
)


def read_csv(path):
    with open(path, newline="") as handle:
        return list(csv.DictReader(handle))


def normalized_method(row):
    policy = row.get("cache_policy", "")
    window = row.get("cache_window_size", "")
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
    if method in METHOD_ORDER:
        return method
    return None


def selected_rows(
    path,
    task,
    allowed_methods,
    allowed_datasets,
    final_only=False,
    required_protocol=None,
):
    rows = []
    for source_row in read_csv(path):
        if final_only and source_row.get("prefix_frames"):
            continue
        if required_protocol and source_row.get("protocol") != required_protocol:
            continue
        if source_row.get("dataset") not in allowed_datasets:
            continue
        method = normalized_method(source_row)
        if method not in allowed_methods:
            continue
        row = {field: source_row.get(field, "") for field in FIELDS}
        row.update(
            task=task,
            method=method,
            source=path,
        )
        rows.append(row)
    return rows


def assert_unique(rows):
    seen = set()
    for row in rows:
        key = (row["task"], row["dataset"], row["method"])
        if key in seen:
            raise RuntimeError(f"duplicate Stage 3.7 comparison row: {key}")
        seen.add(key)


def main():
    parser = argparse.ArgumentParser("Build the Stage 3.7 four-method comparison")
    parser.add_argument("--pose-baseline", default="stage3_3_pose_results.csv")
    parser.add_argument("--pose-k8", default="stage3_7_pose_results.csv")
    parser.add_argument(
        "--static-baseline", default="refine_stage3_3b_recon_results.csv"
    )
    parser.add_argument("--static-k8", default="stage3_7b_recon_results.csv")
    parser.add_argument("--dynamic-baseline", default="stage3_3c_recon_results.csv")
    parser.add_argument("--dynamic-k8", default="stage3_7c_recon_results.csv")
    parser.add_argument("--output", default="stage3_7_comparison.csv")
    args = parser.parse_args()

    baselines = {"full_cache", "stage3_2_k4", "old_dino_k6"}
    candidate = {"temporal_binned_dino_k8"}
    pose_datasets = {"sintel", "scannet", "tum"}
    static_datasets = {"7scenes", "nrgbd", "eth3d"}

    rows = []
    rows += selected_rows(args.pose_baseline, "pose", baselines, pose_datasets)
    rows += selected_rows(args.pose_k8, "pose", candidate, pose_datasets)
    rows += selected_rows(
        args.static_baseline,
        "static_recon",
        baselines,
        static_datasets,
        final_only=True,
        required_protocol="dense",
    )
    rows += selected_rows(
        args.static_k8,
        "static_recon",
        candidate,
        static_datasets,
        final_only=True,
        required_protocol="dense",
    )
    rows += selected_rows(
        args.dynamic_baseline,
        "dynamic_recon",
        baselines,
        {"tum"},
        final_only=True,
        required_protocol="paper",
    )
    rows += selected_rows(
        args.dynamic_k8,
        "dynamic_recon",
        candidate,
        {"tum"},
        final_only=True,
        required_protocol="paper",
    )
    assert_unique(rows)
    rows.sort(
        key=lambda row: (
            TASK_ORDER[row["task"]],
            row["dataset"],
            METHOD_ORDER[row["method"]],
        )
    )

    with open(args.output, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} Stage 3.7 comparison rows to {args.output}")


if __name__ == "__main__":
    main()
