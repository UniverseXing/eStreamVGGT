#!/usr/bin/env python3
"""Build the paper's supplementary result tables and calculation note.

This script performs no model inference, selector tuning, or Stage 4E-A
processing. It validates and reorganises the frozen Stage 3--5 results used by
the paper. The public supplementary package is deliberately limited to the
complete result tables and the definitions of normalised regret/oracle wins.
"""

from __future__ import annotations

import argparse
import csv
import io
import math
import shutil
import tarfile
from pathlib import Path, PurePosixPath


FINAL_METHODS = ("full_cache", "k4", "k6", "k8")
METHOD_ORDER = {method: index for index, method in enumerate(FINAL_METHODS)}
METHOD_LABELS = {
    "full_cache": "Full cache",
    "k4": "K4",
    "k6": "K6",
    "k8": "K8",
}
PAPER_POLICIES = {
    "full_cache": "full_cache",
    "k4": "anchor_recent_dino_diverse_k4",
    "k6": "anchor_recent_dino_diverse_k6",
    "k8": "anchor_recent_dino_diverse_k8",
}
WINDOW_SIZES = {"full_cache": "", "k4": 4, "k6": 6, "k8": 8}

METHOD_ALIASES = {
    "full_cache": "full_cache",
    "dense_full_cache": "full_cache",
    "paper_full_cache": "full_cache",
    "stage3_2_k4": "k4",
    "dense_stage3_2_k4": "k4",
    "anchor_recent_dino_diverse_2old_1recent": "k4",
    "anchor_recent_dino_diverse_k4": "k4",
    "old_dino_k6": "k6",
    "dense_old_dino_k6": "k6",
    "anchor_recent_dino_diverse": "k6",
    "anchor_recent_dino_diverse_k6": "k6",
    "temporal_binned_dino_k8": "k8",
    "anchor_recent_dino_diverse_k8": "k8",
}

POSE_DATASETS = ("scannet", "sintel", "tum")
STATIC_RECON_DATASETS = ("7scenes", "nrgbd", "eth3d")
LONG_SEQUENCES = (
    "rgbd_dataset_freiburg1_room",
    "rgbd_dataset_freiburg2_desk",
    "rgbd_dataset_freiburg3_long_office_household",
)
LONG_SEQUENCE_LABELS = {
    "rgbd_dataset_freiburg1_room": "F1-Room",
    "rgbd_dataset_freiburg2_desk": "F2-Desk",
    "rgbd_dataset_freiburg3_long_office_household": "F3-LongOffice",
}

STAGE5_METHODS = (
    "full_cache",
    "recent4",
    "anchor_recent4",
    "proposed_k4",
    "anchor_uniform4",
    "random4_seed0",
    "random4_seed1",
    "random4_seed2",
)
STAGE5_METHOD_LABELS = {
    "full_cache": "Full cache",
    "recent4": "Recent-4",
    "anchor_recent4": "Anchor+Recent-4",
    "proposed_k4": "K4",
    "anchor_uniform4": "Uniform-4",
    "random4_seed0": "Random-4 (seed 0)",
    "random4_seed1": "Random-4 (seed 1)",
    "random4_seed2": "Random-4 (seed 2)",
    "random4_mean": "Random-4 (three-seed mean)",
}

POSE_SUMMARY_FIELDS = (
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
)
RECON_SUMMARY_FIELDS = (
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
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build frozen supplementary CSV tables and calculation note."
    )
    parser.add_argument(
        "--video-depth-summary", default="stage4a_video_depth_results(1).csv"
    )
    parser.add_argument(
        "--video-depth-sequences", default="stage4b_video_depth_sequence_results.csv"
    )
    parser.add_argument(
        "--video-depth-paired", default="stage4b_video_depth_paired_comparison.csv"
    )
    parser.add_argument(
        "--video-depth-statistics", default="stage4b_video_depth_statistics.csv"
    )
    parser.add_argument("--video-depth-pareto", default="stage4b_pareto.csv")
    parser.add_argument("--pose-summary", default="stage3_3_pose_results.csv")
    parser.add_argument("--pose-k8-summary", default="stage3_7_pose_results.csv")
    parser.add_argument(
        "--pose-archive", default="stage4_supp_pose_metrics.tar.gz"
    )
    parser.add_argument(
        "--static-recon-summary", default="refine_stage3_3b_recon_results.csv"
    )
    parser.add_argument(
        "--static-recon-k8-summary", default="stage3_7b_recon_results.csv"
    )
    parser.add_argument(
        "--dynamic-recon-summary", default="stage3_3c_recon_results.csv"
    )
    parser.add_argument(
        "--dynamic-recon-k8-summary", default="stage3_7c_recon_results.csv"
    )
    parser.add_argument(
        "--recon-archive", default="stage3_7_sequence_metrics.tar.gz"
    )
    parser.add_argument(
        "--cross-task-summary", default="stage4b_cross_task_summary.csv"
    )
    parser.add_argument(
        "--cross-task-regret", default="stage4b_cross_task_regret.csv"
    )
    parser.add_argument("--long-results", default="stage4c_results.csv")
    parser.add_argument(
        "--stage5a-summary", default="stage5a_same_budget_results.csv"
    )
    parser.add_argument(
        "--stage5a-sequences", default="stage5a_same_budget_sequence_results.csv"
    )
    parser.add_argument(
        "--stage5a-paired", default="stage5a_paired_statistics.csv"
    )
    parser.add_argument(
        "--stage5b-decomposition", default="stage5b_memory_decomposition.csv"
    )
    parser.add_argument("--stage5b-trace", default="stage5b_memory_trace.csv")
    parser.add_argument(
        "--stage5b-contributions", default="stage5b_memory_contributions.csv"
    )
    parser.add_argument("--output-root", default="supplementary")
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, fields: list[str] | tuple[str, ...], rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fields,
            extrasaction="ignore",
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows):4d} rows: {path}")


def normalise_method(value: str) -> str:
    try:
        return METHOD_ALIASES[value]
    except KeyError as error:
        raise ValueError(f"unknown final method alias: {value!r}") from error


def method_columns(method_id: str) -> dict[str, object]:
    return {
        "method": METHOD_LABELS[method_id],
        "method_id": method_id,
        "cache_policy": PAPER_POLICIES[method_id],
        "cache_window_size": WINDOW_SIZES[method_id],
    }


def clean_number(value):
    return "" if value is None else value


def mib_to_gib(value) -> str:
    if value in (None, ""):
        return ""
    return f"{float(value) / 1024.0:.12g}"


def close(a, b, *, rel_tol=1e-9, abs_tol=1e-10) -> bool:
    if a in (None, "") and b in (None, ""):
        return True
    if a in (None, "") or b in (None, ""):
        return False
    return math.isclose(float(a), float(b), rel_tol=rel_tol, abs_tol=abs_tol)


def required_paths(args: argparse.Namespace) -> list[Path]:
    names = (
        "video_depth_summary",
        "video_depth_sequences",
        "video_depth_paired",
        "video_depth_statistics",
        "video_depth_pareto",
        "pose_summary",
        "pose_k8_summary",
        "pose_archive",
        "static_recon_summary",
        "static_recon_k8_summary",
        "dynamic_recon_summary",
        "dynamic_recon_k8_summary",
        "recon_archive",
        "cross_task_summary",
        "cross_task_regret",
        "long_results",
        "stage5a_summary",
        "stage5a_sequences",
        "stage5a_paired",
        "stage5b_decomposition",
        "stage5b_trace",
        "stage5b_contributions",
    )
    paths = [Path(getattr(args, name)) for name in names]
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError("missing required inputs:\n  " + "\n  ".join(missing))
    if any("stage4e" in str(path).lower() for path in paths):
        raise ValueError("Stage 4E-A inputs are intentionally forbidden")
    return paths


def json_members(path: Path) -> list[tuple[str, dict]]:
    records = []
    with tarfile.open(path, "r:gz") as archive:
        for member in sorted(archive.getmembers(), key=lambda item: item.name):
            if not member.isfile():
                continue
            pure = PurePosixPath(member.name)
            if pure.is_absolute() or ".." in pure.parts:
                raise ValueError(f"unsafe archive member: {member.name}")
            if pure.name != "pose_metrics.json" and pure.name != "reconstruction_metrics.json":
                raise ValueError(f"unexpected JSON archive member: {member.name}")
            handle = archive.extractfile(member)
            if handle is None:
                raise ValueError(f"cannot read archive member: {member.name}")
            import json

            records.append((member.name, json.load(io.TextIOWrapper(handle, "utf-8"))))
    return records


def build_video_depth(args, tables_dir: Path) -> None:
    summary_rows = read_csv(Path(args.video_depth_summary))
    if len(summary_rows) != 12:
        raise ValueError(f"expected 12 VideoDepth summary rows, got {len(summary_rows)}")
    seen = set()
    output = []
    for row in summary_rows:
        method_id = normalise_method(row["method"])
        key = (row["dataset"], method_id)
        if key in seen:
            raise ValueError(f"duplicate VideoDepth summary row: {key}")
        seen.add(key)
        item = {
            "dataset": row["dataset"],
            **method_columns(method_id),
            "num_sequences": row["num_sequences"],
            "num_successful": row["num_ok"],
            "num_oom": row["num_oom"],
            "total_frames": row["total_frames"],
            "abs_rel": row["abs_rel"],
            "sq_rel": row["sq_rel"],
            "rmse": row["rmse"],
            "log_rmse": row["log_rmse"],
            "delta_1": row["delta_1"],
            "delta_2": row["delta_2"],
            "delta_3": row["delta_3"],
            "total_inference_sec": row["total_inference_sec"],
            "fps_inference": row["fps_inference"],
            "peak_allocated_mib": row["max_peak_allocated_mb"],
            "peak_allocated_gib": mib_to_gib(row["max_peak_allocated_mb"]),
            "peak_reserved_mib": row["max_peak_reserved_mb"],
            "peak_reserved_gib": mib_to_gib(row["max_peak_reserved_mb"]),
            "gpu_name": row["gpu_name"],
            "torch_version": row["torch_version"],
            "cuda_version": row["cuda_version"],
        }
        output.append(item)
    expected = {(d, m) for d in ("bonn", "kitti", "sintel") for m in FINAL_METHODS}
    if seen != expected:
        raise ValueError(f"VideoDepth coverage mismatch: {sorted(expected - seen)}")
    output.sort(key=lambda r: (("bonn", "kitti", "sintel").index(r["dataset"]), METHOD_ORDER[r["method_id"]]))
    fields = list(output[0])
    write_csv(tables_dir / "table_s01_video_depth_summary.csv", fields, output)

    sequence_rows = []
    sequence_seen = {}
    for row in read_csv(Path(args.video_depth_sequences)):
        method_id = normalise_method(row["method"])
        sequence_seen.setdefault((row["dataset"], method_id), set()).add(row["sequence"])
        sequence_rows.append(
            {
                "dataset": row["dataset"],
                "sequence": row["sequence"],
                **method_columns(method_id),
                "num_frames": row["num_frames"],
                "valid_pixels": row["valid_pixels"],
                "abs_rel": row["abs_rel"],
                "sq_rel": row["sq_rel"],
                "rmse": row["rmse"],
                "log_rmse": row["log_rmse"],
                "delta_1": row["delta_1"],
                "delta_2": row["delta_2"],
                "delta_3": row["delta_3"],
                "inference_sec": row["inference_sec"],
                "fps_inference": row["fps_inference"],
                "peak_allocated_mib": row["peak_allocated_mb"],
                "peak_reserved_mib": row["peak_reserved_mb"],
            }
        )
    if len(sequence_rows) != 164:
        raise ValueError(f"expected 164 VideoDepth sequence rows, got {len(sequence_rows)}")
    for dataset in ("bonn", "kitti", "sintel"):
        references = [sequence_seen[(dataset, method)] for method in FINAL_METHODS]
        if any(items != references[0] for items in references[1:]):
            raise ValueError(f"unpaired VideoDepth sequences for {dataset}")
    sequence_rows.sort(key=lambda r: (r["dataset"], r["sequence"], METHOD_ORDER[r["method_id"]]))
    write_csv(
        tables_dir / "table_s02_video_depth_sequences.csv",
        list(sequence_rows[0]),
        sequence_rows,
    )

    paired_rows = []
    for row in read_csv(Path(args.video_depth_paired)):
        item = dict(row)
        item["method_a"] = METHOD_LABELS[normalise_method(row["method_a"])]
        item["method_b"] = METHOD_LABELS[normalise_method(row["method_b"])]
        paired_rows.append(item)
    write_csv(
        tables_dir / "table_s03_video_depth_paired_bootstrap.csv",
        list(paired_rows[0]),
        paired_rows,
    )

    statistics_rows = []
    for row in read_csv(Path(args.video_depth_statistics)):
        item = dict(row)
        item["method"] = METHOD_LABELS[normalise_method(row["method"])]
        statistics_rows.append(item)
    write_csv(
        tables_dir / "table_s04_video_depth_sequence_statistics.csv",
        list(statistics_rows[0]),
        statistics_rows,
    )

    pareto_rows = []
    for row in read_csv(Path(args.video_depth_pareto)):
        method_id = normalise_method(row["method"])
        dominated = " ".join(
            METHOD_LABELS[normalise_method(value)]
            for value in row["dominated_by"].split()
        )
        pareto_rows.append(
            {
                "dataset": row["dataset"],
                **method_columns(method_id),
                "abs_rel": row["abs_rel"],
                "delta_1": row["delta_1"],
                "total_inference_sec": row["total_inference_sec"],
                "fps_inference": row["fps_inference"],
                "peak_allocated_mib": row["max_peak_allocated_mb"],
                "peak_reserved_mib": row["max_peak_reserved_mb"],
                "on_joint_pareto_front": row["pareto_absrel_allocated_time"],
                "dominated_by": dominated,
            }
        )
    write_csv(
        tables_dir / "table_s05_video_depth_pareto.csv",
        list(pareto_rows[0]),
        pareto_rows,
    )


def final_pose_summary_csv(args) -> dict[tuple[str, str], dict[str, str]]:
    rows = read_csv(Path(args.pose_summary)) + read_csv(Path(args.pose_k8_summary))
    selected = {}
    for row in rows:
        try:
            method_id = normalise_method(row["cache_policy"])
        except ValueError:
            continue
        key = (row["dataset"], method_id)
        if key in selected:
            raise ValueError(f"duplicate final pose summary: {key}")
        selected[key] = row
    expected = {(dataset, method) for dataset in POSE_DATASETS for method in FINAL_METHODS}
    if set(selected) != expected:
        raise ValueError(f"pose CSV coverage mismatch; missing={sorted(expected-set(selected))}")
    return selected


def build_pose(args, tables_dir: Path) -> None:
    csv_summary = final_pose_summary_csv(args)
    members = json_members(Path(args.pose_archive))
    if len(members) != 12:
        raise ValueError(f"expected 12 pose JSON files, got {len(members)}")
    summaries = []
    sequences = []
    seen = set()
    per_dataset_sequences = {}
    for member_name, payload in members:
        if set(payload) != {"summary", "sequences"}:
            raise ValueError(f"unexpected pose JSON schema in {member_name}")
        summary = payload["summary"]
        method_id = normalise_method(summary["cache_policy"])
        dataset = summary["dataset"]
        key = (dataset, method_id)
        if key in seen:
            raise ValueError(f"duplicate pose JSON: {key}")
        seen.add(key)
        expected = csv_summary[key]
        for field in POSE_SUMMARY_FIELDS:
            if not close(summary.get(field), expected.get(field)):
                raise ValueError(
                    f"pose JSON/CSV mismatch for {key} {field}: "
                    f"{summary.get(field)} != {expected.get(field)}"
                )
        seq_rows = payload["sequences"]
        if len(seq_rows) != int(summary["num_sequences"]):
            raise ValueError(f"pose sequence count mismatch for {key}")
        if any(row.get("status") != "ok" for row in seq_rows):
            raise ValueError(f"non-success pose sequence in {key}")
        names_and_frames = {(row["sequence"], int(row["num_frames"])) for row in seq_rows}
        per_dataset_sequences.setdefault(dataset, []).append(names_and_frames)
        summaries.append(
            {
                "dataset": dataset,
                **method_columns(method_id),
                **{field: clean_number(summary.get(field)) for field in POSE_SUMMARY_FIELDS},
                "peak_allocated_gib": mib_to_gib(summary.get("max_peak_allocated_mb")),
                "peak_reserved_gib": mib_to_gib(summary.get("max_peak_reserved_mb")),
                "aggregation": "sequence_equal_mean",
            }
        )
        for row in seq_rows:
            sequences.append(
                {
                    "dataset": dataset,
                    "sequence": row["sequence"],
                    **method_columns(method_id),
                    "status": row["status"],
                    "num_frames": row["num_frames"],
                    "ate": row["ate"],
                    "rpe_trans": row["rpe_trans"],
                    "rpe_rot_deg": row["rpe_rot_deg"],
                    "align_scale": row["align_scale"],
                    "inference_sec": row["inference_sec"],
                    "fps_inference": row["fps_inference"],
                    "peak_allocated_mib": row["peak_allocated_mb"],
                    "peak_reserved_mib": row["peak_reserved_mb"],
                }
            )
    expected_keys = {(d, m) for d in POSE_DATASETS for m in FINAL_METHODS}
    if seen != expected_keys:
        raise ValueError(f"pose JSON coverage mismatch; missing={sorted(expected_keys-seen)}")
    for dataset, variants in per_dataset_sequences.items():
        if len(variants) != 4 or any(item != variants[0] for item in variants[1:]):
            raise ValueError(f"pose sequence/frame pairing mismatch for {dataset}")
    if len(sequences) != 112:
        raise ValueError(f"expected 112 pose sequence rows, got {len(sequences)}")
    summaries.sort(key=lambda r: (POSE_DATASETS.index(r["dataset"]), METHOD_ORDER[r["method_id"]]))
    sequences.sort(key=lambda r: (POSE_DATASETS.index(r["dataset"]), r["sequence"], METHOD_ORDER[r["method_id"]]))
    write_csv(tables_dir / "table_s06_pose_summary.csv", list(summaries[0]), summaries)
    write_csv(tables_dir / "table_s07_pose_sequences.csv", list(sequences[0]), sequences)

    # Compact policy ablation retained for supplementary review only.
    ablation_methods = {
        "full_cache": ("Full cache", "full_cache"),
        "anchor_recent_dino_diverse_2old_1recent": (
            "K4",
            "anchor_recent_dino_diverse_k4",
        ),
        "anchor_recent_dino_diverse": (
            "K6",
            "anchor_recent_dino_diverse_k6",
        ),
        "anchor_recent_uniform": ("Uniform K6", "anchor_recent_uniform"),
        "fifo": ("FIFO K6", "fifo"),
    }
    ablation = []
    for row in read_csv(Path(args.pose_summary)):
        policy = row["cache_policy"]
        if policy not in ablation_methods:
            continue
        method_label, public_policy = ablation_methods[policy]
        ablation.append(
            {
                "dataset": row["dataset"],
                "method": method_label,
                "cache_policy": public_policy,
                "cache_window_size": row["cache_window_size"],
                **{field: row.get(field, "") for field in POSE_SUMMARY_FIELDS},
            }
        )
    method_order = [label for label, _ in ablation_methods.values()]
    ablation.sort(key=lambda r: (r["dataset"], method_order.index(r["method"])))
    write_csv(tables_dir / "table_s08_pose_policy_ablation.csv", list(ablation[0]), ablation)


def expected_recon_csv(args, task: str) -> dict[tuple[str, str], dict[str, str]]:
    if task == "static":
        rows = read_csv(Path(args.static_recon_summary)) + read_csv(Path(args.static_recon_k8_summary))
        datasets = STATIC_RECON_DATASETS
        protocol = "dense"
    else:
        rows = read_csv(Path(args.dynamic_recon_summary)) + read_csv(Path(args.dynamic_recon_k8_summary))
        datasets = ("tum",)
        protocol = "paper"
    selected = {}
    for row in rows:
        if row["dataset"] not in datasets or row["protocol"] != protocol or row["prefix_frames"]:
            continue
        try:
            method_id = normalise_method(row["method"])
        except ValueError:
            continue
        key = (row["dataset"], method_id)
        if key in selected:
            raise ValueError(f"duplicate {task} reconstruction summary: {key}")
        selected[key] = row
    expected = {(dataset, method) for dataset in datasets for method in FINAL_METHODS}
    if set(selected) != expected:
        raise ValueError(f"{task} reconstruction CSV coverage mismatch")
    return selected


def build_reconstruction(args, tables_dir: Path) -> None:
    expected_static = expected_recon_csv(args, "static")
    expected_dynamic = expected_recon_csv(args, "dynamic")
    members = json_members(Path(args.recon_archive))
    if len(members) != 8:
        raise ValueError(f"expected 8 reconstruction JSON files, got {len(members)}")
    static_summaries = []
    dynamic_summaries = []
    prefix_rows = []
    sequence_rows = []
    seen_method_tasks = set()
    sequence_pairing = {}
    timing_fields = {"total_inference_sec", "fps_inference", "mean_final_frame_ms"}
    for member_name, payload in members:
        policy = payload.get("cache_policy")
        method_id = normalise_method(policy)
        protocol = payload.get("protocol")
        task = "static" if protocol == "dense" else "dynamic" if protocol == "paper" else None
        if task is None:
            raise ValueError(f"unexpected reconstruction protocol in {member_name}: {protocol}")
        method_task = (task, method_id)
        if method_task in seen_method_tasks:
            raise ValueError(f"duplicate reconstruction JSON for {method_task}")
        seen_method_tasks.add(method_task)
        datasets = STATIC_RECON_DATASETS if task == "static" else ("tum",)
        if set(payload.get("datasets", {})) != set(datasets):
            raise ValueError(f"reconstruction dataset coverage mismatch in {member_name}")
        expected_map = expected_static if task == "static" else expected_dynamic
        for dataset in datasets:
            block = payload["datasets"][dataset]
            summary = block["summary"]
            expected = expected_map[(dataset, method_id)]
            for field in RECON_SUMMARY_FIELDS:
                if not close(summary.get(field), expected.get(field)):
                    if field not in timing_fields:
                        raise ValueError(
                            f"reconstruction JSON/CSV mismatch for "
                            f"{(dataset, method_id)} {field}"
                        )
                    # The sequence JSON is the source for the complete table.
                    # A historical aggregate CSV may differ slightly in timing
                    # while retaining identical quality, pose and coverage.
            item = {
                "task": task,
                "dataset": dataset,
                **method_columns(method_id),
                "protocol": protocol,
                "sampling_stride": payload["sampling_strides"][dataset],
                **{field: clean_number(summary.get(field)) for field in RECON_SUMMARY_FIELDS},
                "peak_allocated_gib": mib_to_gib(summary.get("max_peak_allocated_mb")),
                "peak_reserved_gib": mib_to_gib(summary.get("max_peak_reserved_mb")),
                "aggregation": "sequence_equal_mean",
            }
            (static_summaries if task == "static" else dynamic_summaries).append(item)
            for prefix in block.get("prefix_summaries", []):
                prefix_rows.append(
                    {
                        "task": task,
                        "dataset": dataset,
                        **method_columns(method_id),
                        "protocol": protocol,
                        "sampling_stride": payload["sampling_strides"][dataset],
                        "prefix_frames": prefix["prefix_frames"],
                        "num_sequences": prefix["num_sequences"],
                        "mean_acc": prefix["mean_acc"],
                        "mean_acc_med": prefix["mean_acc_med"],
                        "mean_comp": prefix["mean_comp"],
                        "mean_comp_med": prefix["mean_comp_med"],
                        "mean_nc": prefix["mean_nc"],
                        "mean_nc_med": prefix["mean_nc_med"],
                        "mean_overall": prefix["mean_overall"],
                        "mean_final_frame_ms": prefix["mean_final_frame_ms"],
                    }
                )
            # Compare the successfully evaluated sequence/frame pairs.  The
            # legacy 7-Scenes Full/K4/K6 JSONs retain six ineligible (<2-frame)
            # sequences as explicit failures, while the later K8 evaluator
            # omits those six rows.  The 12 successful sequences are paired.
            compact_success = set()
            for row in block.get("sequences", []):
                if row["status"] == "ok":
                    compact_success.add((row["sequence"], int(row["num_frames"])))
                sequence_rows.append(
                    {
                        "task": task,
                        "dataset": dataset,
                        "sequence": row["sequence"],
                        **method_columns(method_id),
                        "status": row["status"],
                        "error": row.get("error", ""),
                        "num_frames": row.get("num_frames", ""),
                        "accuracy": row.get("acc", ""),
                        "accuracy_median": row.get("acc_med", ""),
                        "completeness": row.get("comp", ""),
                        "completeness_median": row.get("comp_med", ""),
                        "normal_consistency": row.get("nc", ""),
                        "normal_consistency_median": row.get("nc_med", ""),
                        "overall": row.get("overall", ""),
                        "pose_status": row.get("pose_status", ""),
                        "pose_error": row.get("pose_error", ""),
                        "ate": row.get("ate", ""),
                        "rpe_trans": row.get("rpe_trans", ""),
                        "rpe_rot_deg": row.get("rpe_rot_deg", ""),
                        "icp_fitness": row.get("icp_fitness", ""),
                        "icp_rmse": row.get("icp_rmse", ""),
                        "inference_sec": row.get("inference_sec", ""),
                        "fps_inference": row.get("fps_inference", ""),
                        "final_frame_ms": row.get("final_frame_ms", ""),
                        "peak_allocated_mib": row.get("peak_allocated_mb", ""),
                        "peak_reserved_mib": row.get("peak_reserved_mb", ""),
                    }
                )
            sequence_pairing.setdefault((task, dataset), []).append(compact_success)
    expected_method_tasks = {(task, method) for task in ("static", "dynamic") for method in FINAL_METHODS}
    if seen_method_tasks != expected_method_tasks:
        raise ValueError("reconstruction JSON method/task coverage mismatch")
    for key, variants in sequence_pairing.items():
        if len(variants) != 4 or any(item != variants[0] for item in variants[1:]):
            raise ValueError(f"reconstruction sequence/frame pairing mismatch for {key}")
    if len(static_summaries) != 12 or len(dynamic_summaries) != 4:
        raise ValueError("unexpected reconstruction summary row count")
    if len(sequence_rows) != 186:
        raise ValueError(f"expected 186 reconstruction sequence rows, got {len(sequence_rows)}")
    static_summaries.sort(key=lambda r: (STATIC_RECON_DATASETS.index(r["dataset"]), METHOD_ORDER[r["method_id"]]))
    dynamic_summaries.sort(key=lambda r: METHOD_ORDER[r["method_id"]])
    prefix_rows.sort(key=lambda r: (r["task"], r["dataset"], int(r["prefix_frames"]), METHOD_ORDER[r["method_id"]]))
    sequence_rows.sort(key=lambda r: (r["task"], r["dataset"], r["sequence"], METHOD_ORDER[r["method_id"]]))
    write_csv(tables_dir / "table_s09_static_reconstruction_summary.csv", list(static_summaries[0]), static_summaries)
    write_csv(tables_dir / "table_s10_dynamic_reconstruction_summary.csv", list(dynamic_summaries[0]), dynamic_summaries)
    write_csv(tables_dir / "table_s11_reconstruction_prefixes.csv", list(prefix_rows[0]), prefix_rows)
    write_csv(tables_dir / "table_s12_reconstruction_sequences.csv", list(sequence_rows[0]), sequence_rows)


def normalise_method_field(row: dict[str, str], field: str = "method") -> dict[str, str]:
    item = dict(row)
    method_id = normalise_method(row[field])
    if field == "method":
        item.pop("method")
        item = {
            "method": METHOD_LABELS[method_id],
            "method_id": method_id,
            **item,
        }
    else:
        item[field] = METHOD_LABELS[method_id]
    return item


def strip_fields(
    row: dict[str, str],
    fields=(
        "source",
        "result_dir",
        "slurm_job_id",
        "hostname",
        "coverage_ok",
        "camera_pose_sha256",
    ),
):
    return {key: value for key, value in row.items() if key not in fields}


def build_cross_task_and_long(args, tables_dir: Path) -> None:
    cross_rows = []
    for row in read_csv(Path(args.cross_task_summary)):
        # The historical Stage 3.6B platform row is a provenance record, not
        # one of the ten frozen task-dataset benchmark cells used for 7/1/2
        # oracle wins. Stage 4C long-sequence results are reported separately.
        if row["task"] == "long_sequence_platform":
            continue
        item = strip_fields(row)
        item = normalise_method_field(item)
        cross_rows.append(item)
    if len(cross_rows) != 40:
        raise ValueError(f"expected 40 final cross-task cells, got {len(cross_rows)}")
    write_csv(tables_dir / "table_s13_cross_task_summary.csv", list(cross_rows[0]), cross_rows)

    regret_rows = []
    for row in read_csv(Path(args.cross_task_regret)):
        item = normalise_method_field(row)
        regret_rows.append(item)
    write_csv(tables_dir / "table_s14_cross_task_regret.csv", list(regret_rows[0]), regret_rows)

    long_rows = []
    source = read_csv(Path(args.long_results))
    if len(source) != 42:
        raise ValueError(f"expected 42 Stage 4C rows, got {len(source)}")
    bounded_ok = 0
    full_seen = set()
    for row in source:
        method_id = normalise_method(row["method"])
        sequence = row["sequence"]
        frames = int(row["num_frames"])
        if sequence not in LONG_SEQUENCES:
            raise ValueError(f"unexpected long-sequence name: {sequence}")
        if method_id == "full_cache":
            full_seen.add((sequence, frames, row["status"]))
        elif row["status"] == "ok":
            bounded_ok += 1
        item = strip_fields(row)
        item.update(method_columns(method_id))
        item["sequence_label"] = LONG_SEQUENCE_LABELS[sequence]
        long_rows.append(item)
    if bounded_ok != 36:
        raise ValueError(f"expected 36 successful bounded long runs, got {bounded_ok}")
    for sequence in LONG_SEQUENCES:
        if (sequence, 100, "ok") not in full_seen or not any(
            item[0] == sequence and item[1] == 250 and item[2] == "failed"
            for item in full_seen
        ):
            raise ValueError(f"unexpected full-cache ceiling for {sequence}")
    long_rows.sort(key=lambda r: (LONG_SEQUENCES.index(r["sequence"]), int(r["num_frames"]), METHOD_ORDER[r["method_id"]]))
    write_csv(tables_dir / "table_s15_long_sequence_results.csv", list(long_rows[0]), long_rows)


def stage5_method_columns(method_id: str) -> dict[str, str]:
    if method_id not in STAGE5_METHOD_LABELS:
        raise ValueError(f"unknown Stage 5 method: {method_id!r}")
    return {
        "method": STAGE5_METHOD_LABELS[method_id],
        "method_id": "k4" if method_id == "proposed_k4" else method_id,
    }


def build_stage5(args, tables_dir: Path) -> None:
    summary_rows = []
    summary_seen = set()
    for row in read_csv(Path(args.stage5a_summary)):
        method_id = row["method"]
        key = (row["dataset"], method_id)
        if key in summary_seen:
            raise ValueError(f"duplicate Stage 5A summary row: {key}")
        summary_seen.add(key)
        summary_rows.append(
            {
                "dataset": row["dataset"],
                **stage5_method_columns(method_id),
                "random_seed": row["random_seed"],
                "cache_policy": row["cache_policy"],
                "cache_window_size": row["cache_window_size"],
                "num_sequences": row["num_sequences"],
                "num_successful": row["num_successful"],
                "num_failed": row["num_failed"],
                "total_frames": row["total_frames"],
                "abs_rel": row["abs_rel"],
                "rmse": row["rmse"],
                "delta_1": row["delta_1"],
                "fps_inference": row["fps_inference"],
                "peak_allocated_mib": row["peak_allocated_mb"],
                "peak_reserved_mib": row["peak_reserved_mb"],
                "gpu_name": row["gpu_name"],
                "torch_version": row["torch_version"],
                "cuda_version": row["cuda_version"],
            }
        )
    expected = {
        (dataset, method)
        for dataset in ("bonn", "kitti", "sintel")
        for method in STAGE5_METHODS
    }
    if summary_seen != expected:
        raise ValueError(
            "Stage 5A summary coverage mismatch; "
            f"missing={sorted(expected-summary_seen)}, "
            f"extra={sorted(summary_seen-expected)}"
        )
    stage5_order = {
        method: index for index, method in enumerate(STAGE5_METHODS)
    }
    summary_rows.sort(
        key=lambda row: (
            ("bonn", "kitti", "sintel").index(row["dataset"]),
            stage5_order[
                "proposed_k4" if row["method_id"] == "k4" else row["method_id"]
            ],
        )
    )
    write_csv(
        tables_dir / "table_s16_same_budget_video_depth_summary.csv",
        list(summary_rows[0]),
        summary_rows,
    )

    sequence_rows = []
    sequence_seen = {}
    for row in read_csv(Path(args.stage5a_sequences)):
        method_id = row["method"]
        sequence_seen.setdefault((row["dataset"], method_id), set()).add(
            (row["sequence"], int(row["num_frames"]))
        )
        sequence_rows.append(
            {
                "dataset": row["dataset"],
                "sequence": row["sequence"],
                **stage5_method_columns(method_id),
                "random_seed": row["random_seed"],
                "num_frames": row["num_frames"],
                "abs_rel": row["abs_rel"],
                "rmse": row["rmse"],
                "delta_1": row["delta_1"],
                "inference_sec": row["inference_sec"],
                "fps_inference": row["fps_inference"],
                "peak_allocated_mib": row["peak_allocated_mb"],
                "peak_reserved_mib": row["peak_reserved_mb"],
            }
        )
    if len(sequence_rows) != 328:
        raise ValueError(
            f"expected 328 Stage 5A sequence rows, got {len(sequence_rows)}"
        )
    for dataset in ("bonn", "kitti", "sintel"):
        variants = [sequence_seen[(dataset, method)] for method in STAGE5_METHODS]
        if any(items != variants[0] for items in variants[1:]):
            raise ValueError(f"unpaired Stage 5A sequences for {dataset}")
    sequence_rows.sort(
        key=lambda row: (
            row["dataset"],
            row["sequence"],
            stage5_order[
                "proposed_k4" if row["method_id"] == "k4" else row["method_id"]
            ],
        )
    )
    write_csv(
        tables_dir / "table_s17_same_budget_video_depth_sequences.csv",
        list(sequence_rows[0]),
        sequence_rows,
    )

    paired_rows = []
    for row in read_csv(Path(args.stage5a_paired)):
        item = dict(row)
        item["proposed"] = STAGE5_METHOD_LABELS[item["proposed"]]
        item["control"] = STAGE5_METHOD_LABELS[item["control"]]
        paired_rows.append(item)
    if len(paired_rows) != 12:
        raise ValueError(f"expected 12 Stage 5A paired rows, got {len(paired_rows)}")
    write_csv(
        tables_dir / "table_s18_same_budget_paired_bootstrap.csv",
        list(paired_rows[0]),
        paired_rows,
    )

    decomposition_rows = [
        strip_fields(row, fields=("source", "slurm_job_id", "hostname"))
        for row in read_csv(Path(args.stage5b_decomposition))
    ]
    if len(decomposition_rows) != 4 or any(
        row["status"] != "ok" or row["processed_frames"] != "110"
        for row in decomposition_rows
    ):
        raise ValueError("Stage 5B decomposition must contain four complete cells")
    write_csv(
        tables_dir / "table_s19_memory_decomposition.csv",
        list(decomposition_rows[0]),
        decomposition_rows,
    )

    trace_rows = read_csv(Path(args.stage5b_trace))
    if len(trace_rows) != 440:
        raise ValueError(f"expected 440 Stage 5B trace rows, got {len(trace_rows)}")
    write_csv(
        tables_dir / "table_s20_memory_trace.csv",
        list(trace_rows[0]),
        trace_rows,
    )

    contribution_rows = read_csv(Path(args.stage5b_contributions))
    if len(contribution_rows) != 4:
        raise ValueError(
            f"expected four Stage 5B contribution rows, got {len(contribution_rows)}"
        )
    write_csv(
        tables_dir / "table_s21_memory_contributions.csv",
        list(contribution_rows[0]),
        contribution_rows,
    )


def write_calculation_methods(output_root: Path) -> None:
    content = r"""# Normalised Regret and Oracle Wins

This note defines exactly how the values in
`tables/table_s14_cross_task_regret.csv` were calculated.

## Comparison scope

The oracle is restricted to the three bounded methods $\mathcal{M}=\{\mathrm{K4},
\mathrm{K6},\mathrm{K8}\}$. Full cache is an unbounded reference and is not
eligible for a bounded oracle win. All methods compared within one row use the
same evaluation units and metric definition.

## Unit-level normalised regret

Let $x_{m,u}$ be the value obtained by method $m$ on evaluation unit $u$. For a
metric where lower values are better, the bounded oracle and regret are

\begin{equation}
o_u=\min_{m\in\mathcal{M}}x_{m,u},\qquad
r_{m,u}=\frac{x_{m,u}-o_u}{\max(|o_u|,10^{-12})}.
\end{equation}

For a metric where higher values are better, they are

\begin{equation}
o_u=\max_{m\in\mathcal{M}}x_{m,u},\qquad
r_{m,u}=\frac{o_u-x_{m,u}}{\max(|o_u|,10^{-12})}.
\end{equation}

The denominator makes regret dimensionless and comparable across metrics. The
$10^{-12}$ floor only prevents division by zero. For each reported group,
`mean_normalized_regret`, `median_normalized_regret`, and
`max_normalized_regret` are the corresponding statistics over its evaluation
units.

## Oracle wins

A method receives one oracle win on unit $u$ when its value equals $o_u$ under
`math.isclose` with relative tolerance $10^{-8}$ and absolute tolerance
$10^{-10}$. Tied methods each receive a win, so the total number of wins can
exceed the number of units. The `oracle_wins` column is the sum of these
unit-level indicators.

## Evaluation units and metrics

| Task | Dataset coverage | Unit used in dataset rows | Primary metric | Secondary metric |
|---|---|---|---|---|
| VideoDepth | Bonn, KITTI, Sintel | sequence | AbsRel (lower) | $\delta_1$ (higher) |
| Camera pose | ScanNet, Sintel, TUM | dataset aggregate | ATE (lower) | rotation RPE in degrees (lower) |
| Static reconstruction | 7-Scenes, NRGBD, ETH3D | successful sequence | overall error (lower) | normal consistency (higher) |
| Dynamic reconstruction | TUM Dynamics | sequence | overall error (lower) | normal consistency (higher) |

Rows with `dataset=all` pool the comparable units across datasets within the
same task. The `cross_task_macro` rows are different: they use only the primary
metric from each of the ten task--dataset benchmark cells (three VideoDepth,
three pose, three static reconstruction, and one dynamic reconstruction cell).
The oracle is recomputed independently in each cell, after which the ten
normalised regrets are summarised. This produces the paper's bounded primary
oracle-win counts of 7 for K4, 1 for K6, and 2 for K8.

The aggregate values supplied to this calculation are listed in
`tables/table_s13_cross_task_summary.csv`; the complete regret output is in
`tables/table_s14_cross_task_regret.csv`.
"""
    (output_root / "CALCULATION_METHODS.md").write_text(content, encoding="utf-8")


def write_readme(output_root: Path) -> None:
    content = """# Supplementary Information

This directory is generated by `scripts/build_supplementary_assets.py`.
Its scope follows the statement in the paper: complete experimental results,
plus the calculation of normalised regret and oracle wins.

## Complete experimental results

- S01--S05: complete VideoDepth aggregate, sequence, bootstrap, statistics, and Pareto data.
- S06--S08: final pose aggregate/sequence results and the compact policy ablation.
- S09--S12: static/dynamic reconstruction aggregate, prefix, and sequence results.
- S13--S14: cross-task summary and normalised regret/oracle-win results.
- S15: all held-out long-sequence resource and pose runs.
- S16--S18: the complete same-budget VideoDepth controls, sequence results, and paired bootstrap tests.
- S19--S21: the four-cell memory decomposition, per-frame trace, and isolated contributions.

All table assets are CSV files with full stored precision. Pose and
reconstruction aggregate means are sequence-equal means.  VideoDepth S01 uses
the official valid-pixel-weighted aggregate; S02--S04 contain sequence-level
values and sequence-equal statistics.

For 7-Scenes, Full/K4/K6 retain six ineligible one-frame sequences as failed
records, whereas K8 records only the same 12 valid sequences. Quality comparison
therefore uses the common 12 successful sequences; K8's zero failed count is not
interpreted as greater robustness.

## Normalised regret and oracle wins

`CALCULATION_METHODS.md` gives the exact oracle scope, equations, tie tolerance,
evaluation units, metrics, aggregation rules, and cross-task macro procedure.
"""
    (output_root / "README.md").write_text(content, encoding="utf-8")


def main() -> None:
    args = parse_args()
    required_paths(args)
    output_root = Path(args.output_root)
    tables_dir = output_root / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    # Remove the previous, broader supplementary package before rebuilding the
    # deliberately narrow public package requested by the paper statement.
    for directory in (output_root / "figures", output_root / "audit"):
        if directory.exists():
            shutil.rmtree(directory)
    for name in ("asset_manifest.csv", "source_manifest.csv", "figure_inventory.csv"):
        path = output_root / name
        if path.is_file():
            path.unlink()

    generated_tables = {f"table_s{index:02d}" for index in range(1, 22)}
    if tables_dir.exists():
        for path in tables_dir.glob("*.csv"):
            if any(path.stem.startswith(prefix) for prefix in generated_tables):
                path.unlink()

    build_video_depth(args, tables_dir)
    build_pose(args, tables_dir)
    build_reconstruction(args, tables_dir)
    build_cross_task_and_long(args, tables_dir)
    build_stage5(args, tables_dir)
    write_calculation_methods(output_root)
    write_readme(output_root)

    if any(path.suffix == ".tex" for path in output_root.rglob("*")):
        raise RuntimeError("unexpected TeX table generated")
    if any("stage4e" in path.name.lower() or "fusion" in path.name.lower() for path in output_root.rglob("*")):
        raise RuntimeError("unexpected Stage 4E-A/fusion artifact generated")
    expected_files = {
        Path("README.md"),
        Path("CALCULATION_METHODS.md"),
        *(Path("tables") / f"table_s{index:02d}_{name}.csv" for index, name in (
            (1, "video_depth_summary"),
            (2, "video_depth_sequences"),
            (3, "video_depth_paired_bootstrap"),
            (4, "video_depth_sequence_statistics"),
            (5, "video_depth_pareto"),
            (6, "pose_summary"),
            (7, "pose_sequences"),
            (8, "pose_policy_ablation"),
            (9, "static_reconstruction_summary"),
            (10, "dynamic_reconstruction_summary"),
            (11, "reconstruction_prefixes"),
            (12, "reconstruction_sequences"),
            (13, "cross_task_summary"),
            (14, "cross_task_regret"),
            (15, "long_sequence_results"),
            (16, "same_budget_video_depth_summary"),
            (17, "same_budget_video_depth_sequences"),
            (18, "same_budget_paired_bootstrap"),
            (19, "memory_decomposition"),
            (20, "memory_trace"),
            (21, "memory_contributions"),
        )),
    }
    actual_files = {
        path.relative_to(output_root)
        for path in output_root.rglob("*")
        if path.is_file()
    }
    if actual_files != expected_files:
        raise RuntimeError(
            "supplementary package scope mismatch; "
            f"missing={sorted(expected_files-actual_files)}, "
            f"extra={sorted(actual_files-expected_files)}"
        )
    print(f"Supplementary package complete: {output_root} ({len(actual_files)} files)")


if __name__ == "__main__":
    main()
