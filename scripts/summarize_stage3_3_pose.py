#!/usr/bin/env python3
import argparse
import csv
import glob
import json
import os


PUBLIC_METHOD_CONFIGS = {
    "full_cache": ("full_cache", None),
    "anchor_recent_dino_diverse_k4": ("anchor_recent_dino_diverse_k4", 4),
    "anchor_recent_dino_diverse_k6": ("anchor_recent_dino_diverse_k6", 6),
    "anchor_recent_dino_diverse_k8": ("anchor_recent_dino_diverse_k8", 8),
}

FROZEN_SIGNATURES = {
    "sintel": {
        "alley_2": 50,
        "ambush_4": 33,
        "ambush_5": 50,
        "ambush_6": 20,
        "cave_2": 50,
        "cave_4": 50,
        "market_2": 50,
        "market_5": 50,
        "market_6": 40,
        "shaman_3": 50,
        "sleeping_1": 50,
        "sleeping_2": 50,
        "temple_2": 50,
        "temple_3": 50,
    },
    "scannet": {
        "scene0707_00": 90,
        "scene0708_00": 90,
        "scene0709_00": 90,
        "scene0710_00": 90,
        "scene0711_00": 87,
        "scene0712_00": 90,
    },
    "tum": {
        "rgbd_dataset_freiburg3_sitting_halfsphere": 90,
        "rgbd_dataset_freiburg3_sitting_rpy": 90,
        "rgbd_dataset_freiburg3_sitting_static": 90,
        "rgbd_dataset_freiburg3_sitting_xyz": 90,
        "rgbd_dataset_freiburg3_walking_halfsphere": 90,
        "rgbd_dataset_freiburg3_walking_rpy": 90,
        "rgbd_dataset_freiburg3_walking_static": 90,
        "rgbd_dataset_freiburg3_walking_xyz": 90,
    },
}

FIELDS = (
    "run_scope",
    "dataset",
    "cache_policy",
    "cache_window_size",
    "gpu_name",
    "torch_version",
    "cuda_version",
    "python_version",
    "slurm_job_id",
    "hostname",
    "input_size",
    "stride",
    "requested_max_frames",
    "num_sequences",
    "num_successful",
    "num_failed",
    "total_frames",
    "mean_ate",
    "mean_rpe_trans",
    "mean_rpe_rot_deg",
    "total_inference_sec",
    "fps_inference",
    "max_peak_allocated_mb",
    "max_peak_reserved_mb",
    "result_dir",
)


def public_run_identity(path, name_filter):
    directory = os.path.basename(os.path.dirname(path))
    for dataset in FROZEN_SIGNATURES:
        prefix = f"{dataset}_{name_filter}_"
        if directory.startswith(prefix):
            method = directory[len(prefix) :]
            if method in PUBLIC_METHOD_CONFIGS:
                return dataset, method
    expected = f"<dataset>_{name_filter}_<public_method>"
    raise RuntimeError(
        f"invalid pose result directory {directory!r}; expected {expected}"
    )


def validate_method_metadata(summary, dataset, method, path):
    if summary.get("dataset") != dataset:
        raise RuntimeError(
            f"pose dataset metadata mismatch in {path}: directory={dataset!r}, "
            f"summary={summary.get('dataset')!r}"
        )
    expected_policy, expected_window = PUBLIC_METHOD_CONFIGS[method]
    actual_policy = summary.get("cache_policy")
    actual_window = summary.get("cache_window_size")
    if (
        "cache_policy" not in summary
        or "cache_window_size" not in summary
        or actual_policy != expected_policy
        or actual_window != expected_window
    ):
        raise RuntimeError(
            f"pose method metadata mismatch in {path}: directory={method!r}, "
            f"expected policy/window={expected_policy!r}/{expected_window!r}, "
            f"summary={actual_policy!r}/{actual_window!r}"
        )


def validate_frozen_protocol(summary, dataset, signature, path):
    expected_signature = set(FROZEN_SIGNATURES[dataset].items())
    if signature != expected_signature:
        missing = sorted(expected_signature - signature)
        unexpected = sorted(signature - expected_signature)
        raise RuntimeError(
            f"pose frozen sequence/frame signature mismatch for {dataset} in {path}; "
            f"missing={missing}, unexpected={unexpected}"
        )
    expected_config = {
        "input_size": 518,
        "stride": 1,
        "requested_max_frames": None,
    }
    mismatches = {
        field: (summary.get(field), expected)
        for field, expected in expected_config.items()
        if field not in summary or summary.get(field) != expected
    }
    if mismatches:
        raise RuntimeError(
            f"pose frozen configuration mismatch in {path}: {mismatches}"
        )


def main():
    parser = argparse.ArgumentParser("Summarize Stage 3.3 pose runs")
    parser.add_argument("--results-root", default="eval_results/pose")
    parser.add_argument("--name-filter", default="stage3_3")
    parser.add_argument("--output", default="stage3_3_pose_results.csv")
    parser.add_argument("--expected-runs", type=int)
    parser.add_argument(
        "--require-all-success",
        action="store_true",
        help="fail if a run contains a failed sequence or methods used different coverage",
    )
    parser.add_argument(
        "--allow-subset",
        action="store_true",
        help="label custom sequence/frame coverage as debug_subset instead of enforcing the frozen protocol",
    )
    args = parser.parse_args()
    if args.allow_subset and not args.require_all_success:
        raise RuntimeError("--allow-subset requires --require-all-success")

    pattern = os.path.join(args.results_root, f"*{args.name_filter}*", "pose_metrics.json")
    paths = sorted(glob.glob(pattern))
    if args.expected_runs is not None and len(paths) != args.expected_runs:
        raise RuntimeError(
            f"expected {args.expected_runs} pose result files, found {len(paths)}; "
            "use a fresh task output root when running a subset"
        )
    rows = []
    coverage_by_dataset = {}
    provenance_values = {
        field: set()
        for field in ("gpu_name", "torch_version", "cuda_version", "python_version")
    }
    seen_runs = set()
    for path in paths:
        with open(path) as handle:
            payload = json.load(handle)
        summary = payload["summary"]
        if args.require_all_success:
            dataset, method = public_run_identity(path, args.name_filter)
            run_key = (dataset, method)
            if run_key in seen_runs:
                raise RuntimeError(f"duplicate pose dataset/method result: {run_key}")
            seen_runs.add(run_key)
            validate_method_metadata(summary, dataset, method, path)
            for field, values in provenance_values.items():
                value = summary.get(field)
                if value is None or not str(value).strip():
                    raise RuntimeError(f"missing pose {field} in {path}")
                values.add(str(value).strip())
            sequence_rows = payload.get("sequences", [])
            if len(sequence_rows) != int(summary.get("num_sequences", -1)):
                raise RuntimeError(f"invalid pose sequence count in {path}")
            names = [item.get("sequence") for item in sequence_rows]
            if len(names) != len(set(names)):
                raise RuntimeError(f"duplicate pose sequence in {path}")
            failed = [
                item.get("sequence")
                for item in sequence_rows
                if item.get("status") != "ok"
            ]
            if failed or int(summary.get("num_failed", 0)) != 0:
                raise RuntimeError(f"failed pose sequence(s) in {path}: {failed}")
            if int(summary.get("num_successful", -1)) != len(sequence_rows):
                raise RuntimeError(f"invalid pose successful-sequence count in {path}")
            signature = {
                (item["sequence"], int(item["num_frames"]))
                for item in sequence_rows
            }
            reference = coverage_by_dataset.setdefault(dataset, signature)
            if signature != reference:
                raise RuntimeError(
                    f"pose sequence/frame coverage differs across methods for {dataset}"
                )
            if not args.allow_subset:
                validate_frozen_protocol(summary, dataset, signature, path)
        row = {field: summary.get(field) for field in FIELDS}
        row["run_scope"] = "debug_subset" if args.allow_subset else "frozen"
        row["result_dir"] = os.path.dirname(path)
        rows.append(row)
    if not rows:
        raise RuntimeError(f"no pose result files matched {pattern}")
    if args.require_all_success:
        for field, values in provenance_values.items():
            if len(values) != 1:
                raise RuntimeError(
                    f"inconsistent pose {field}: {sorted(values)}"
                )

    with open(args.output, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} runs to {args.output}")


if __name__ == "__main__":
    main()
