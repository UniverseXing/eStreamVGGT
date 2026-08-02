#!/usr/bin/env python3
import argparse
import csv
import glob
import json
import os
from collections import defaultdict


FIELDS = (
    "dataset",
    "method",
    "run_scope",
    "gpu_name",
    "torch_version",
    "cuda_version",
    "python_version",
    "slurm_job_id",
    "hostname",
    "input_size",
    "use_proj",
    "max_scenes",
    "max_frames",
    "icp_threshold",
    "protocol",
    "sampling_stride",
    "prefix_frames",
    "cache_policy",
    "cache_window_size",
    "num_sequences",
    "num_successful",
    "num_failed",
    "total_frames",
    "mean_acc",
    "mean_acc_med",
    "mean_comp",
    "mean_comp_med",
    "mean_nc",
    "mean_nc_med",
    "mean_overall",
    "mean_ate",
    "mean_rpe_trans",
    "mean_rpe_rot_deg",
    "total_inference_sec",
    "fps_inference",
    "mean_final_frame_ms",
    "max_peak_allocated_mb",
    "max_peak_reserved_mb",
    "result_dir",
)

RUN_METADATA_FIELDS = (
    "gpu_name",
    "torch_version",
    "cuda_version",
    "python_version",
    "slurm_job_id",
    "hostname",
    "input_size",
    "use_proj",
    "max_scenes",
    "max_frames",
    "icp_threshold",
)

CORE_PROVENANCE_FIELDS = (
    "gpu_name",
    "torch_version",
    "cuda_version",
    "python_version",
)

METHOD_CONFIGS = {
    "full_cache": ("full_cache", None),
    "anchor_recent_dino_diverse_k4": ("anchor_recent_dino_diverse_k4", 4),
    "anchor_recent_dino_diverse_k6": ("anchor_recent_dino_diverse_k6", 6),
    "anchor_recent_dino_diverse_k8": ("anchor_recent_dino_diverse_k8", 8),
}

FROZEN_SEQUENCE_SIGNATURES = {
    "7scenes": {
        "chess/seq-03": 20,
        "chess/seq-05": 8,
        "fire/seq-03": 20,
        "fire/seq-04": 20,
        "heads/seq-01": 20,
        "office/seq-02": 20,
        "pumpkin/seq-01": 20,
        "pumpkin/seq-07": 20,
        "redkitchen/seq-03": 20,
        "redkitchen/seq-04": 20,
        "stairs/seq-01": 10,
        "stairs/seq-04": 10,
    },
    "nrgbd": {
        "breakfast_room": 12,
        "complete_kitchen": 13,
        "green_room": 15,
        "grey_white_room": 15,
        "kitchen": 16,
        "morning_apartment": 10,
        "staircase": 12,
        "thin_geometry": 4,
        "whiteroom": 17,
    },
    "eth3d": {
        "courtyard": 10,
        "delivery_area": 10,
        "electro": 10,
        "facade": 10,
        "kicker": 10,
        "meadow": 10,
        "office": 10,
        "pipes": 10,
        "playground": 10,
        "relief": 10,
        "relief_2": 10,
        "terrace": 10,
        "terrains": 10,
    },
    "tum": {
        "rgbd_dataset_freiburg3_sitting_halfsphere": 50,
        "rgbd_dataset_freiburg3_sitting_rpy": 50,
        "rgbd_dataset_freiburg3_sitting_static": 50,
        "rgbd_dataset_freiburg3_sitting_xyz": 50,
        "rgbd_dataset_freiburg3_walking_halfsphere": 50,
        "rgbd_dataset_freiburg3_walking_rpy": 50,
        "rgbd_dataset_freiburg3_walking_static": 50,
        "rgbd_dataset_freiburg3_walking_xyz": 50,
    },
}

FROZEN_DATASET_CONFIGS = {
    "7scenes": ("dense", 50, (4, 6, 8, 10)),
    "nrgbd": ("dense", 100, (4, 6, 8, 10)),
    "eth3d": ("dense", "random_10", (4, 6, 8, 10)),
    "tum": ("paper", "first_50", (10, 20, 30, 40, 50)),
}

QUALITY_FIELDS = (
    "mean_acc",
    "mean_acc_med",
    "mean_comp",
    "mean_comp_med",
    "mean_nc",
    "mean_nc_med",
    "mean_overall",
    "mean_ate",
    "mean_rpe_trans",
    "mean_rpe_rot_deg",
    "mean_final_frame_ms",
)


def mean(values):
    values = [value for value in values if value is not None]
    return sum(values) / len(values) if values else None


def method_name(result_dir, name_filter):
    name = os.path.basename(result_dir)
    prefix = f"streamvggt_{name_filter}_"
    return name[len(prefix) :] if name.startswith(prefix) else name


def normalize_window(value, label):
    if value in (None, ""):
        return None
    if isinstance(value, bool):
        raise RuntimeError(f"{label} must be a positive integer or null")
    try:
        window = int(value)
    except (TypeError, ValueError) as error:
        raise RuntimeError(
            f"{label} must be a positive integer or null"
        ) from error
    if window < 1 or str(value).strip() != str(window):
        raise RuntimeError(f"{label} must be a positive integer or null")
    return window


def validate_method_metadata(payload, summary, method, path, dataset_name):
    if method not in METHOD_CONFIGS:
        raise RuntimeError(
            f"unknown public reconstruction method in directory {path}: {method!r}"
        )
    expected_policy, expected_window = METHOD_CONFIGS[method]
    payload_window = normalize_window(
        payload.get("cache_window_size"),
        f"{path}:cache_window_size",
    )
    if (
        payload.get("cache_policy") != expected_policy
        or payload_window != expected_window
    ):
        raise RuntimeError(
            f"reconstruction method metadata mismatch in {path}: directory "
            f"method {method!r} requires {expected_policy!r}/K{expected_window}, "
            f"got {payload.get('cache_policy')!r}/K{payload_window}"
        )
    summary_window = normalize_window(
        summary.get("cache_window_size"),
        f"{path}/{dataset_name}:summary.cache_window_size",
    )
    if (
        summary.get("cache_policy") != expected_policy
        or summary_window != expected_window
    ):
        raise RuntimeError(
            f"reconstruction summary method metadata mismatch in "
            f"{path}/{dataset_name}: expected {expected_policy!r}/K{expected_window}, "
            f"got {summary.get('cache_policy')!r}/K{summary_window}"
        )
    if summary.get("dataset") != dataset_name:
        raise RuntimeError(
            f"reconstruction dataset metadata mismatch in {path}/{dataset_name}: "
            f"{summary.get('dataset')!r}"
        )
    if summary.get("protocol") != payload.get("protocol"):
        raise RuntimeError(
            f"reconstruction protocol metadata mismatch in {path}/{dataset_name}"
        )
    sampling_strides = payload.get("sampling_strides")
    if not isinstance(sampling_strides, dict) or dataset_name not in sampling_strides:
        raise RuntimeError(
            f"missing reconstruction sampling stride in {path}/{dataset_name}"
        )
    if summary.get("sampling_stride") != sampling_strides[dataset_name]:
        raise RuntimeError(
            f"reconstruction sampling-stride metadata mismatch in "
            f"{path}/{dataset_name}"
        )


def validate_frozen_dataset(payload, dataset_payload, signature, path, dataset_name):
    if dataset_name not in FROZEN_DATASET_CONFIGS:
        raise RuntimeError(
            f"unknown dataset in frozen reconstruction run {path}: {dataset_name!r}"
        )
    expected_protocol, expected_stride, expected_prefixes = (
        FROZEN_DATASET_CONFIGS[dataset_name]
    )
    required_top_level = (
        "model_name",
        "input_size",
        "use_proj",
        "max_scenes",
        "max_frames",
        "icp_threshold",
        "seed",
        "prefix_frames",
        "protocol",
        "sampling_strides",
    )
    missing = [field for field in required_top_level if field not in payload]
    if missing:
        raise RuntimeError(
            f"missing frozen reconstruction config field(s) in {path}: {missing}"
        )
    expected_scalars = {
        "model_name": "StreamVGGT",
        "input_size": 518,
        "use_proj": False,
        "max_scenes": None,
        "max_frames": None,
        "seed": 0,
        "protocol": expected_protocol,
    }
    for field, expected in expected_scalars.items():
        if payload[field] != expected or type(payload[field]) is not type(expected):
            raise RuntimeError(
                f"non-frozen reconstruction {field} in {path}: "
                f"expected {expected!r}, got {payload[field]!r}"
            )
    try:
        icp_threshold = float(payload["icp_threshold"])
    except (TypeError, ValueError) as error:
        raise RuntimeError(
            f"non-frozen reconstruction icp_threshold in {path}: "
            f"{payload['icp_threshold']!r}"
        ) from error
    if icp_threshold != 0.1:
        raise RuntimeError(
            f"non-frozen reconstruction icp_threshold in {path}: "
            f"expected 0.1, got {payload['icp_threshold']!r}"
        )
    if payload["prefix_frames"] != list(expected_prefixes):
        raise RuntimeError(
            f"non-frozen reconstruction prefix_frames in {path}: expected "
            f"{list(expected_prefixes)}, got {payload['prefix_frames']!r}"
        )
    if payload["sampling_strides"].get(dataset_name) != expected_stride:
        raise RuntimeError(
            f"non-frozen reconstruction sampling stride in {path}/{dataset_name}: "
            f"expected {expected_stride!r}, got "
            f"{payload['sampling_strides'].get(dataset_name)!r}"
        )
    expected_signature = set(FROZEN_SEQUENCE_SIGNATURES[dataset_name].items())
    if signature != expected_signature:
        missing_sequences = sorted(expected_signature - signature)
        extra_sequences = sorted(signature - expected_signature)
        raise RuntimeError(
            f"frozen reconstruction sequence/frame signature mismatch in "
            f"{path}/{dataset_name}; missing={missing_sequences}, "
            f"extra={extra_sequences}"
        )
    prefix_summaries = dataset_payload.get("prefix_summaries", [])
    observed_prefixes = [row.get("prefix_frames") for row in prefix_summaries]
    if observed_prefixes != list(expected_prefixes):
        raise RuntimeError(
            f"frozen reconstruction prefix summary mismatch in "
            f"{path}/{dataset_name}: expected {list(expected_prefixes)}, "
            f"got {observed_prefixes}"
        )


def macro_row(method, rows):
    total_frames = sum(row.get("total_frames") or 0 for row in rows)
    total_time = sum(row.get("total_inference_sec") or 0.0 for row in rows)
    row = {
        "dataset": "macro_average",
        "method": method,
        "run_scope": rows[0]["run_scope"],
        **{field: rows[0].get(field) for field in RUN_METADATA_FIELDS},
        "protocol": rows[0]["protocol"],
        "sampling_stride": None,
        "prefix_frames": None,
        "cache_policy": rows[0]["cache_policy"],
        "cache_window_size": rows[0]["cache_window_size"],
        "num_sequences": sum(row.get("num_sequences") or 0 for row in rows),
        "num_successful": sum(row.get("num_successful") or 0 for row in rows),
        "num_failed": sum(row.get("num_failed") or 0 for row in rows),
        "total_frames": total_frames,
        "total_inference_sec": total_time,
        "fps_inference": total_frames / total_time if total_time else None,
        "max_peak_allocated_mb": max(
            (row["max_peak_allocated_mb"] for row in rows if row.get("max_peak_allocated_mb") is not None),
            default=None,
        ),
        "max_peak_reserved_mb": max(
            (row["max_peak_reserved_mb"] for row in rows if row.get("max_peak_reserved_mb") is not None),
            default=None,
        ),
        "result_dir": rows[0]["result_dir"],
    }
    for field in QUALITY_FIELDS:
        row[field] = mean([dataset_row.get(field) for dataset_row in rows])
    return row


def main():
    parser = argparse.ArgumentParser("Summarize Stage 3.3B reconstruction runs")
    parser.add_argument("--results-root", default="eval_results/mv_recon")
    parser.add_argument("--name-filter", default="stage3_3b")
    parser.add_argument("--output", default="stage3_3b_recon_results.csv")
    parser.add_argument("--expected-runs", type=int)
    parser.add_argument(
        "--require-all-success",
        action="store_true",
        help="fail if a run contains failures or methods used different coverage",
    )
    parser.add_argument(
        "--allow-subset",
        action="store_true",
        help=(
            "label output as debug_subset and skip frozen config/signature checks; "
            "success, paired coverage, provenance, and method metadata remain required"
        ),
    )
    args = parser.parse_args()
    if args.allow_subset and not args.require_all_success:
        raise RuntimeError("--allow-subset requires --require-all-success")

    pattern = os.path.join(
        args.results_root, f"*{args.name_filter}*", "reconstruction_metrics.json"
    )
    paths = sorted(glob.glob(pattern))
    if args.expected_runs is not None and len(paths) != args.expected_runs:
        raise RuntimeError(
            f"expected {args.expected_runs} reconstruction result files, found "
            f"{len(paths)}; use a fresh task output root when running a subset"
        )
    rows = []
    by_method = defaultdict(list)
    coverage_by_dataset = {}
    dataset_set_reference = None
    seen_methods = set()
    provenance_values = {
        field: set() for field in CORE_PROVENANCE_FIELDS
    }
    run_scope = "debug_subset" if args.allow_subset else "frozen"
    for path in paths:
        result_dir = os.path.dirname(path)
        method = method_name(result_dir, args.name_filter)
        with open(path) as handle:
            payload = json.load(handle)
        run_metadata = {
            field: payload.get(field) for field in RUN_METADATA_FIELDS
        }
        if args.require_all_success:
            for field, values in provenance_values.items():
                value = payload.get(field)
                if value is None or not str(value).strip():
                    raise RuntimeError(
                        f"missing reconstruction {field} in {path}"
                    )
                values.add(str(value).strip())
            if method in seen_methods:
                raise RuntimeError(
                    f"duplicate reconstruction method result: {method!r}"
                )
            seen_methods.add(method)
            dataset_names = frozenset(payload.get("datasets", {}))
            if not dataset_names:
                raise RuntimeError(f"no reconstruction datasets in {path}")
            if dataset_set_reference is None:
                dataset_set_reference = dataset_names
            elif dataset_names != dataset_set_reference:
                raise RuntimeError(
                    "reconstruction dataset coverage differs across methods: "
                    f"{sorted(dataset_set_reference)} != {sorted(dataset_names)}"
                )
        for dataset_name, dataset_payload in payload["datasets"].items():
            summary = dataset_payload["summary"]
            if args.require_all_success:
                validate_method_metadata(
                    payload,
                    summary,
                    method,
                    path,
                    dataset_name,
                )
                sequence_rows = dataset_payload.get("sequences", [])
                if len(sequence_rows) != int(summary.get("num_sequences", -1)):
                    raise RuntimeError(
                        f"invalid reconstruction sequence count in "
                        f"{path}/{dataset_name}"
                    )
                names = [item.get("sequence") for item in sequence_rows]
                if len(names) != len(set(names)):
                    raise RuntimeError(
                        f"duplicate reconstruction sequence in {path}/{dataset_name}"
                    )
                failed = [
                    item.get("sequence")
                    for item in sequence_rows
                    if item.get("status") != "ok"
                ]
                if failed or int(summary.get("num_failed", 0)) != 0:
                    raise RuntimeError(
                        f"failed reconstruction sequence(s) in "
                        f"{path}/{dataset_name}: {failed}"
                    )
                if int(summary.get("num_successful", -1)) != len(sequence_rows):
                    raise RuntimeError(
                        f"invalid reconstruction successful-sequence count in "
                        f"{path}/{dataset_name}"
                    )
                pose_failed = [
                    item.get("sequence")
                    for item in sequence_rows
                    if item.get("status") == "ok"
                    and item.get("pose_status") != "ok"
                ]
                if pose_failed:
                    raise RuntimeError(
                        f"failed reconstruction pose metric(s) in "
                        f"{path}/{dataset_name}: {pose_failed}"
                    )
                signature = {
                    (item["sequence"], int(item["num_frames"]))
                    for item in sequence_rows
                }
                reference = coverage_by_dataset.setdefault(
                    dataset_name, signature
                )
                if signature != reference:
                    raise RuntimeError(
                        "reconstruction sequence/frame coverage differs "
                        f"across methods for {dataset_name}"
                    )
                if not args.allow_subset:
                    validate_frozen_dataset(
                        payload,
                        dataset_payload,
                        signature,
                        path,
                        dataset_name,
                    )
            row = {field: summary.get(field) for field in FIELDS}
            row.update(run_metadata)
            row["method"] = method
            row["run_scope"] = run_scope
            row["result_dir"] = result_dir
            rows.append(row)
            by_method[method].append(row)
            for prefix_summary in dataset_payload.get("prefix_summaries", []):
                prefix_row = {field: prefix_summary.get(field) for field in FIELDS}
                prefix_row.update(run_metadata)
                prefix_row["method"] = method
                prefix_row["run_scope"] = run_scope
                prefix_row["sampling_stride"] = summary.get("sampling_stride")
                prefix_row["num_successful"] = prefix_summary.get("num_sequences")
                prefix_row["num_failed"] = 0
                prefix_row["result_dir"] = result_dir
                rows.append(prefix_row)
    if not rows:
        raise RuntimeError(f"no reconstruction result files matched {pattern}")
    if args.require_all_success:
        for field, values in provenance_values.items():
            if len(values) != 1:
                raise RuntimeError(
                    f"inconsistent reconstruction {field}: {sorted(values)}"
                )

    for method, method_rows in by_method.items():
        rows.append(macro_row(method, method_rows))

    with open(args.output, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} rows to {args.output}")


if __name__ == "__main__":
    main()
