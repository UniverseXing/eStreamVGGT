#!/usr/bin/env python3
"""Build CSV-only supplementary tables and collect frozen PDF figures.

This script performs no model inference, metric recomputation, selector tuning,
or Stage 4E-A processing.  It validates and reorganises the frozen Stage 3/4
artifacts used by the paper into a compact, auditable supplementary package.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import math
import os
import shutil
import tarfile
import tempfile
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

CSV_FIGURES = {
    "fig_video_depth_pareto.pdf": "main-candidate: VideoDepth quality-memory-time Pareto",
    "fig_cross_task_regret.pdf": "supplementary: cross-task normalised regret",
    "fig_stage4c_scaling.pdf": "main-candidate: long-sequence resource scaling",
    "fig_stage4c_pose_scaling.pdf": "supplementary: pose metrics versus prefix length",
}
SERVER_SOURCE_FIGURES = {
    "fig_stage4c_trajectories.pdf": "main-candidate: frozen trajectory cases",
    "fig_stage4c_cache_timeline.pdf": "supplementary: retained-frame timeline",
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
        description="Build frozen supplementary CSV tables and collect PDFs."
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
    parser.add_argument("--method-roles", default="stage4b_method_roles.csv")
    parser.add_argument("--claim-audit", default="stage4b_claim_audit.csv")
    parser.add_argument("--stage4a-gate", default="stage4a_gate.csv")
    parser.add_argument("--long-results", default="stage4c_results.csv")
    parser.add_argument("--long-gate", default="stage4c_gate.csv")
    parser.add_argument("--case-audit", default="stage4d_case_audit.csv")
    parser.add_argument("--stage4d-gate", default="stage4d_gate.csv")
    parser.add_argument(
        "--figure-source-archive",
        default="stage4_supp_figure_sources.tar.gz",
        help="Raw Stage 4C trajectories and memory traces for the final figures.",
    )
    parser.add_argument(
        "--skip-figures",
        action="store_true",
        help="Build tables only when matplotlib/evo are unavailable.",
    )
    parser.add_argument("--output-root", default="supplementary")
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, fields: list[str] | tuple[str, ...], rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows):4d} rows: {path}")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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
        "method_roles",
        "claim_audit",
        "stage4a_gate",
        "long_results",
        "long_gate",
        "case_audit",
        "stage4d_gate",
    )
    paths = [Path(getattr(args, name)) for name in names]
    if not args.skip_figures:
        paths.append(Path(args.figure_source_archive))
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
    ablation_labels = {
        "full_cache": "Full cache",
        "anchor_recent_dino_diverse_2old_1recent": "K4",
        "anchor_recent_dino_diverse": "K6",
        "anchor_recent_uniform": "Uniform K6",
        "fifo": "FIFO K6",
    }
    ablation = []
    for row in read_csv(Path(args.pose_summary)):
        policy = row["cache_policy"]
        if policy not in ablation_labels:
            continue
        ablation.append(
            {
                "dataset": row["dataset"],
                "method": ablation_labels[policy],
                "cache_policy": policy,
                "cache_window_size": row["cache_window_size"],
                **{field: row.get(field, "") for field in POSE_SUMMARY_FIELDS},
            }
        )
    ablation.sort(key=lambda r: (r["dataset"], list(ablation_labels.values()).index(r["method"])))
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


def build_reconstruction(args, tables_dir: Path, audit_dir: Path) -> None:
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
    consistency_notes = []
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
                    consistency_notes.append(
                        {
                            "task": task,
                            "dataset": dataset,
                            "method": METHOD_LABELS[method_id],
                            "field": field,
                            "sequence_json_value": clean_number(summary.get(field)),
                            "final_summary_csv_value": clean_number(expected.get(field)),
                            "resolution": "aggregate_and_sequence_use_json; old_summary_csv_timing_not_used",
                        }
                    )
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
    if consistency_notes:
        write_csv(
            audit_dir / "reconstruction_source_consistency.csv",
            list(consistency_notes[0]),
            consistency_notes,
        )


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


def strip_fields(row: dict[str, str], fields=("source", "result_dir", "slurm_job_id", "hostname")):
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

    role_rows = []
    for row in read_csv(Path(args.method_roles)):
        role_rows.append(normalise_method_field(row))
    if {row["method_id"] for row in role_rows} != set(FINAL_METHODS):
        raise ValueError("method role table does not cover exactly the four final methods")
    write_csv(tables_dir / "table_s15_method_roles.csv", list(role_rows[0]), role_rows)

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
    write_csv(tables_dir / "table_s16_long_sequence_results.csv", list(long_rows[0]), long_rows)


def copy_audits(args, audit_dir: Path) -> None:
    sources = {
        "stage4a_gate.csv": Path(args.stage4a_gate),
        "stage4c_gate.csv": Path(args.long_gate),
        "claim_audit.csv": Path(args.claim_audit),
        "case_audit.csv": Path(args.case_audit),
        "stage4d_gate.csv": Path(args.stage4d_gate),
    }
    audit_dir.mkdir(parents=True, exist_ok=True)
    for name, source in sources.items():
        shutil.copyfile(source, audit_dir / name)


def expected_figure_source_members() -> set[str]:
    prefix = "eval_results/stage4c_tum_long"
    methods = ("stage3_2_k4", "old_dino_k6", "temporal_binned_dino_k8")
    trajectory_cases = (
        ("rgbd_dataset_freiburg1_room", 250),
        ("rgbd_dataset_freiburg2_desk", 500),
        ("rgbd_dataset_freiburg3_long_office_household", 500),
    )
    members = {
        f"{prefix}/{method}/{sequence}/{frames}/trajectory.npz"
        for method in methods
        for sequence, frames in trajectory_cases
    }
    members.update(
        f"{prefix}/{method}/rgbd_dataset_freiburg3_long_office_household/1000/"
        "memory_trace.json"
        for method in methods
    )
    return members


def extract_figure_sources(archive_path: Path, destination: Path) -> Path:
    """Safely extract the exact 12 frozen figure sources into a temporary tree."""
    expected = expected_figure_source_members()
    with tarfile.open(archive_path, "r:gz") as archive:
        files = [member for member in archive.getmembers() if member.isfile()]
        names = {member.name for member in files}
        if len(files) != len(names):
            raise ValueError("duplicate members in figure-source archive")
        if names != expected:
            raise ValueError(
                "figure-source archive coverage mismatch; "
                f"missing={sorted(expected-names)}, extra={sorted(names-expected)}"
            )
        for member in files:
            pure = PurePosixPath(member.name)
            if pure.is_absolute() or ".." in pure.parts:
                raise ValueError(f"unsafe figure-source member: {member.name}")
            handle = archive.extractfile(member)
            if handle is None:
                raise ValueError(f"cannot read figure-source member: {member.name}")
            target = destination.joinpath(*pure.parts)
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open("wb") as output:
                shutil.copyfileobj(handle, output)
    return destination / "eval_results" / "stage4c_tum_long"


def generate_csv_figures(args, figures_dir: Path) -> list[dict[str, str]]:
    """Regenerate all available PDFs with final paper-facing K4/K6/K8 labels."""
    figures_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    if not args.skip_figures:
        try:
            import build_stage4d_paper_assets as plots
        except (ImportError, ModuleNotFoundError) as error:
            raise RuntimeError(
                "figure generation requires the StreamVGGT environment with "
                "numpy, matplotlib and evo; rerun with --skip-figures to build "
                "CSV tables only"
            ) from error
        plots.METHOD_LABELS.update(
            {
                "full_cache": "Full cache",
                "stage3_2_k4": "K4",
                "old_dino_k6": "K6",
                "temporal_binned_dino_k8": "K8",
            }
        )
        plots.plot_video_depth_pareto(
            read_csv(Path(args.video_depth_pareto)), str(figures_dir)
        )
        plots.plot_cross_task_regret(
            read_csv(Path(args.method_roles)), str(figures_dir)
        )
        long_rows = read_csv(Path(args.long_results))
        plots.plot_stage4c_scaling(long_rows, str(figures_dir))
        plots.plot_stage4c_pose(long_rows, str(figures_dir))
        with tempfile.TemporaryDirectory(prefix="streamvggt-supp-fig-") as temp:
            results_root = extract_figure_sources(
                Path(args.figure_source_archive), Path(temp)
            )
            plots.plot_trajectory_cases(
                long_rows, str(results_root), str(figures_dir)
            )
            plots.plot_cache_timeline(
                long_rows, str(results_root), str(figures_dir)
            )
        for png in figures_dir.glob("*.png"):
            png.unlink()

    for name, purpose in {**CSV_FIGURES, **SERVER_SOURCE_FIGURES}.items():
        path = figures_dir / name
        ready = path.is_file() and not args.skip_figures
        source = (
            "frozen CSV files"
            if name in CSV_FIGURES
            else "stage4_supp_figure_sources.tar.gz"
        )
        rows.append(
            {
                "figure": name,
                "recommended_use": purpose,
                "status": "ready" if ready else "not_generated",
                "source": source,
                "required_action": "none" if ready else "rerun without --skip-figures",
                "size_bytes": path.stat().st_size if ready else "",
                "sha256": sha256(path) if ready else "",
            }
        )
    return rows


def write_readme(output_root: Path) -> None:
    content = """# Supplementary asset package

This directory is generated by `scripts/build_supplementary_assets.py`.
It contains no new inference, selector tuning, or Stage 4E-A fusion results.

## Tables

- S01--S05: complete VideoDepth aggregate, sequence, bootstrap, statistics, and Pareto data.
- S06--S08: final pose aggregate/sequence results and the compact policy ablation.
- S09--S12: static/dynamic reconstruction aggregate, prefix, and sequence results.
- S13--S15: cross-task summary, normalized regret, and frozen method roles.
- S16: all held-out long-sequence resource and pose runs.

All table assets are CSV files with full stored precision.  Pose and
reconstruction aggregate means are sequence-equal means.  VideoDepth S01 uses
the official valid-pixel-weighted aggregate; S02--S04 contain sequence-level
values and sequence-equal statistics.

For 7-Scenes, Full/K4/K6 retain six ineligible one-frame sequences as failed
records, whereas K8 records only the same 12 valid sequences. Quality comparison
therefore uses the common 12 successful sequences; K8's zero failed count is not
interpreted as greater robustness.

## Figures

Four figures backed by frozen CSV files and two figures backed by the archived
Stage 4C NPZ/JSON sources are regenerated with the final paper-facing labels
`K4`, `K6`, and `K8`. Their sources and hashes are recorded in
`figure_inventory.csv`. No Stage 4E-A fusion figure is included.

## Audits and provenance

`audit/` preserves the frozen gate/claim/case records. `source_manifest.csv`
hashes every input, and `asset_manifest.csv` hashes every generated asset.
`audit/reconstruction_source_consistency.csv` records the sole provenance
difference: K8/TUM Dynamics has identical quality, pose, coverage and memory in
the JSON and old summary CSV, but slightly different timing. The JSON timing is
used because it is also the source frozen by the cross-task summary.
"""
    (output_root / "README.md").write_text(content, encoding="utf-8")


def build_manifest(paths: list[Path], output_root: Path) -> None:
    # The manifest identifies frozen inputs by their logical bundle names.
    # Recording an absolute caller-side path would leak a machine-specific
    # workspace and make otherwise identical builds produce different CSVs.
    source_names = [path.name for path in paths]
    if len(source_names) != len(set(source_names)):
        raise ValueError(
            "supplementary inputs must have unique logical file names: "
            f"{source_names}"
        )
    source_rows = [
        {
            "source_file": source_name,
            "size_bytes": path.stat().st_size,
            "sha256": sha256(path),
        }
        for path, source_name in zip(paths, source_names)
    ]
    write_csv(output_root / "source_manifest.csv", ("source_file", "size_bytes", "sha256"), source_rows)

    asset_rows = []
    for path in sorted(output_root.rglob("*")):
        if not path.is_file() or path.name == "asset_manifest.csv":
            continue
        relative = path.relative_to(output_root)
        asset_rows.append(
            {
                "relative_path": relative.as_posix(),
                "category": relative.parts[0] if len(relative.parts) > 1 else "root",
                "size_bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
        )
    write_csv(
        output_root / "asset_manifest.csv",
        ("relative_path", "category", "size_bytes", "sha256"),
        asset_rows,
    )


def main() -> None:
    args = parse_args()
    paths = required_paths(args)
    output_root = Path(args.output_root)
    tables_dir = output_root / "tables"
    figures_dir = output_root / "figures"
    audit_dir = output_root / "audit"
    tables_dir.mkdir(parents=True, exist_ok=True)

    # Remove only known generator-owned files. User-authored files in the
    # supplementary tree are left untouched.
    generated_tables = {f"table_s{index:02d}" for index in range(1, 17)}
    if tables_dir.exists():
        for path in tables_dir.glob("*.csv"):
            if any(path.stem.startswith(prefix) for prefix in generated_tables):
                path.unlink()
    if figures_dir.exists():
        for name in (
            *CSV_FIGURES,
            *SERVER_SOURCE_FIGURES,
            "fig_stage4e_fusion_failure.pdf",
        ):
            path = figures_dir / name
            if path.is_file():
                path.unlink()
    if audit_dir.exists():
        for name in (
            "stage4a_gate.csv",
            "stage4c_gate.csv",
            "claim_audit.csv",
            "case_audit.csv",
            "stage4d_gate.csv",
            "reconstruction_source_consistency.csv",
        ):
            path = audit_dir / name
            if path.is_file():
                path.unlink()

    build_video_depth(args, tables_dir)
    build_pose(args, tables_dir)
    build_reconstruction(args, tables_dir, audit_dir)
    build_cross_task_and_long(args, tables_dir)
    copy_audits(args, audit_dir)
    figure_rows = generate_csv_figures(args, figures_dir)
    write_csv(
        output_root / "figure_inventory.csv",
        (
            "figure",
            "recommended_use",
            "status",
            "source",
            "required_action",
            "size_bytes",
            "sha256",
        ),
        figure_rows,
    )
    write_readme(output_root)
    build_manifest(paths, output_root)

    if any(path.suffix == ".tex" for path in output_root.rglob("*")):
        raise RuntimeError("unexpected TeX table generated")
    if any("stage4e" in path.name.lower() or "fusion" in path.name.lower() for path in output_root.rglob("*")):
        raise RuntimeError("unexpected Stage 4E-A/fusion artifact generated")
    print(f"Supplementary package complete: {output_root}")


if __name__ == "__main__":
    main()
