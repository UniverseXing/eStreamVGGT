#!/usr/bin/env python3
"""Curate the P0/P1 supplementary package requested in the review worksheet.

This builder performs no model inference.  It copies existing frozen assets,
derives the protocol/memory/paired tables, and writes an auditable manifest.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import shutil
from collections import defaultdict
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/estreamvggt-supp-builder-matplotlib")
METHOD_LABELS = {
    "full_cache": "Full cache",
    "recent4": "Recent-4",
    "anchor_recent4": "Anchor+Recent-4",
    "anchor_uniform4": "Uniform-4",
    "proposed_k4": "K4",
}
CONTROL_ORDER = (
    "full_cache",
    "recent4",
    "anchor_recent4",
    "anchor_uniform4",
    "random4_mean",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument(
        "--output-root", type=Path, default=Path("supplementary material")
    )
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict], fields=None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"refusing to write empty CSV: {path}")
    fields = fields or list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows):4d} rows: {path}")


def copy_file(source: Path, destination: Path) -> None:
    if not source.is_file():
        raise FileNotFoundError(f"missing frozen supplementary source: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    print(f"Copied: {source} -> {destination}")


def copy_existing_assets(root: Path, output: Path) -> None:
    source = root / "supplementary"
    table_map = {
        "table_s01_video_depth_summary.csv": "p0_04_depth_secondary_metrics_summary.csv",
        "table_s02_video_depth_sequences.csv": "p0_04_depth_secondary_metrics_sequences.csv",
        "table_s06_pose_summary.csv": "p0_05_pose_summary.csv",
        "table_s07_pose_sequences.csv": "p0_05_pose_sequences.csv",
        "table_s09_static_reconstruction_summary.csv": "p0_05_static_reconstruction_summary.csv",
        "table_s10_dynamic_reconstruction_summary.csv": "p0_05_dynamic_reconstruction_summary.csv",
        "table_s12_reconstruction_sequences.csv": "p0_05_reconstruction_sequences.csv",
        "table_s13_cross_task_summary.csv": "p0_06_cross_task_summary.csv",
        "table_s14_cross_task_regret.csv": "p0_06_cross_task_regret.csv",
        "table_s15_long_sequence_results.csv": "p0_07_long_sequence_complete.csv",
        "table_s19_memory_decomposition.csv": "p0_08_memory_factorial_raw.csv",
        "table_s20_memory_trace.csv": "p0_08_memory_trace.csv",
        "table_s21_memory_contributions.csv": "p0_08_memory_contributions.csv",
        "table_s08_pose_policy_ablation.csv": "p1_02_existing_policy_ablation.csv",
    }
    for old, new in table_map.items():
        copy_file(source / "tables" / old, output / "tables" / new)
    for old, new in (
        ("stage5a_same_budget_results.csv", "p0_02_same_budget_summary_current.csv"),
        ("stage5a_same_budget_sequence_results.csv", "p0_03_same_budget_sequences.csv"),
        ("stage5a_paired_statistics.csv", "p0_03_same_budget_bootstrap_current.csv"),
    ):
        copy_file(root / old, output / "tables" / new)

    figure_map = {
        "fig_cross_task_regret.pdf": "p0_06_cross_task_regret.pdf",
        "fig_long_sequence_scaling.pdf": "p0_07_long_sequence_scaling.pdf",
        "fig_pose_case_comparison.pdf": "p0_09_pose_case_comparison.pdf",
        "fig_stage4c_pose_scaling.pdf": "p0_09_pose_scaling.pdf",
        "fig_stage4c_trajectories.pdf": "p0_09_trajectories.pdf",
        "fig_stage5b_memory_decomposition.png": "p0_08_memory_decomposition.png",
        "fig_stage4c_cache_timeline.pdf": "p1_01_existing_cache_timeline_partial.pdf",
    }
    for old, new in figure_map.items():
        copy_file(source / "figures" / old, output / "figures" / new)
    copy_file(
        source / "CALCULATION_METHODS.md",
        output / "methods" / "p0_06_normalized_regret_and_oracle_wins.md",
    )

    selector_dir = root / "eval_results" / "supplementary_selector_trace"
    if (selector_dir / "p1_selector_overhead.csv").is_file():
        copy_file(
            selector_dir / "p1_selector_overhead.csv",
            output / "tables" / "p1_04_selector_overhead.csv",
        )
        copy_file(
            selector_dir / "selector_trace_metadata.json",
            output / "raw" / "p1_01_selector_trace_metadata.json",
        )
        for path in sorted(selector_dir.glob("figure_p1_01_*.png")):
            copy_file(path, output / "figures" / path.name)
        for path in sorted(selector_dir.glob("k*_selector_trace.json")):
            copy_file(path, output / "raw" / path.name)
    k8_dir = root / "eval_results" / "supplementary_k8_controls"
    for name in (
        "k8_controls_summary.csv",
        "k8_controls_sequences.csv",
        "k8_controls_paired.csv",
    ):
        if (k8_dir / name).is_file():
            copy_file(k8_dir / name, output / "tables" / f"p1_02_{name}")
    coverage_dir = root / "eval_results" / "supplementary_k8_coverage"
    for name in (
        "k8_temporal_coverage_steps.csv",
        "k8_temporal_coverage_summary.csv",
        "k8_temporal_coverage_gate.csv",
    ):
        if (coverage_dir / name).is_file():
            copy_file(coverage_dir / name, output / "tables" / f"p1_05_{name}")
    for name in (
        "figure_k8_temporal_coverage.pdf",
        "figure_k8_temporal_coverage.png",
    ):
        if (coverage_dir / name).is_file():
            copy_file(coverage_dir / name, output / "figures" / f"p1_05_{name}")
    for name in (
        "hierarchical_k8_selector_trace.json",
        "nonhierarchical_dino8_selector_trace.json",
        "k8_temporal_coverage_metadata.json",
    ):
        if (coverage_dir / name).is_file():
            copy_file(coverage_dir / name, output / "raw" / f"p1_05_{name}")


def protocol_spec(task: str, dataset: str) -> dict[str, str]:
    common = {
        "input_resize_crop": "long side 518; evaluator-native aspect ratio; no crop",
        "pose_alignment": "n/a",
        "reconstruction_alignment_threshold": "n/a",
        "split_role": "evaluation used before final policy freeze",
        "policy_frozen_before_evaluation": "no",
        "notes": "all compared methods use identical ordered inputs",
    }
    if task == "video_depth":
        if dataset == "bonn":
            common.update(
                sampling="five fixed 110-frame sampled sequences; stride 1 in evaluator",
                valid_mask="GT depth > 0 and < 70 m",
                depth_range="0 < depth < 70 m",
                pose_alignment="per-sequence robust scale-only depth alignment",
            )
        elif dataset == "sintel":
            common.update(
                sampling="23 complete selected sequences; native frame order; stride 1",
                valid_mask="GT depth > 0 and < 70 m",
                depth_range="0 < depth < 70 m; aligned prediction clipped at 70 m",
                pose_alignment="per-sequence robust scale-only depth alignment",
            )
        elif dataset == "kitti":
            common.update(
                sampling="13 depth-validation drives; first up to 110 paired RGB/GT frames",
                valid_mask="GT depth > 0 (sparse annotated validation pixels)",
                depth_range="positive GT depth; no additional maximum-depth cutoff",
                pose_alignment="per-drive robust scale-only depth alignment",
                split_role="held-out outdoor VideoDepth domain",
                policy_frozen_before_evaluation="yes",
                notes="KITTI was added after K4/K6/K8 and bank boundaries were frozen",
            )
        return common
    if task == "pose":
        common.update(
            sampling="prepared frozen sequence list; evaluator stride 1",
            valid_mask="finite matched RGB/ground-truth poses",
            depth_range="n/a",
            pose_alignment="Sim(3) Umeyama alignment; nearest SO(3) projection before ATE/RPE",
        )
        return common
    if task in {"static_reconstruction", "dynamic_reconstruction"}:
        common.update(
            input_resize_crop="518 x 392 evaluator resolution; common centre crop for scoring",
            valid_mask="dataset depth validity mask and finite predicted/GT 3D points",
            depth_range="dataset loader validity; 7-Scenes uses 0.001--10 m projected depth",
            pose_alignment="Sim(3) for reported camera trajectory metrics",
            reconstruction_alignment_threshold="point-to-point ICP, threshold 0.1; direct point-head output",
        )
        if dataset == "7scenes":
            common["sampling"] = "dense protocol; every 50th valid RGB/projected-depth frame"
            common["depth_range"] = "0.001--10 m projected depth"
            common["notes"] = "12 common eligible sequences; six one-frame records are retained as ineligible audit rows"
        elif dataset == "nrgbd":
            common["sampling"] = "dense protocol; every 100th frame"
            common["depth_range"] = "0.001--10 m depth"
        elif dataset == "eth3d":
            common["sampling"] = "10 views per scene, random seed 0"
            common["depth_range"] = "finite positive raw ETH3D ground-truth depth"
        else:
            common["sampling"] = "first 50 frames; prefix metrics at 10/20/30/40/50"
            common["depth_range"] = "0.001--10 m depth"
        return common
    if task == "long_sequence":
        common.update(
            sampling="nearest RGB/mocap association (max 0.02 s); prefixes 100/250/500/1000",
            valid_mask="finite associated RGB/ground-truth pose pairs",
            depth_range="n/a (depth images are not used)",
            pose_alignment="Sim(3) Umeyama alignment; ATE and frame-delta-1 RPE",
            split_role="held-out real long-sequence validation",
            policy_frozen_before_evaluation="yes",
            notes="no K, DINO threshold, bank boundary, or sampling changes permitted after evaluation",
        )
        return common
    raise ValueError(f"unknown task/dataset protocol: {task}/{dataset}")


def build_protocol_table(root: Path, output: Path) -> None:
    source = root / "supplementary" / "tables"
    records: list[dict[str, str]] = []

    video = read_csv(source / "table_s02_video_depth_sequences.csv")
    for row in video:
        if row["method_id"] != "full_cache":
            continue
        records.append(
            {
                "task": "video_depth",
                "dataset": row["dataset"],
                "sequence": row["sequence"],
                "evaluated_frames_or_prefix": row["num_frames"],
                "eligible_for_quality": "yes",
                **protocol_spec("video_depth", row["dataset"]),
            }
        )

    pose = read_csv(source / "table_s07_pose_sequences.csv")
    for row in pose:
        if row["method_id"] != "full_cache":
            continue
        records.append(
            {
                "task": "pose",
                "dataset": row["dataset"],
                "sequence": row["sequence"],
                "evaluated_frames_or_prefix": row["num_frames"],
                "eligible_for_quality": "yes" if row["status"] == "ok" else "no",
                **protocol_spec("pose", row["dataset"]),
            }
        )

    recon = read_csv(source / "table_s12_reconstruction_sequences.csv")
    for row in recon:
        if row["method_id"] != "full_cache":
            continue
        task = "static_reconstruction" if row["task"] == "static" else "dynamic_reconstruction"
        eligible = row["status"] == "ok"
        records.append(
            {
                "task": task,
                "dataset": row["dataset"],
                "sequence": row["sequence"],
                "evaluated_frames_or_prefix": row["num_frames"] or "1 (ineligible)",
                "eligible_for_quality": "yes" if eligible else "no",
                **protocol_spec(task, row["dataset"]),
            }
        )

    long_rows = read_csv(source / "table_s15_long_sequence_results.csv")
    for row in long_rows:
        if row["method_id"] != "k4":
            continue
        records.append(
            {
                "task": "long_sequence",
                "dataset": "tum_rgbd_raw",
                "sequence": row["sequence"],
                "evaluated_frames_or_prefix": row["num_frames"],
                "eligible_for_quality": "yes",
                **protocol_spec("long_sequence", "tum_rgbd_raw"),
            }
        )

    counts = defaultdict(set)
    for row in records:
        if row["eligible_for_quality"] == "yes":
            counts[(row["task"], row["dataset"])].add(row["sequence"])
    for row in records:
        row["eligible_sequence_count"] = len(counts[(row["task"], row["dataset"])])
        row["input_resolution"] = "518"
    records.sort(
        key=lambda row: (
            row["task"], row["dataset"], row["sequence"],
            int(str(row["evaluated_frames_or_prefix"]).split()[0]),
        )
    )
    fields = (
        "task", "dataset", "sequence", "eligible_sequence_count",
        "evaluated_frames_or_prefix", "eligible_for_quality", "sampling",
        "input_resolution", "input_resize_crop", "valid_mask", "depth_range",
        "pose_alignment", "reconstruction_alignment_threshold", "split_role",
        "policy_frozen_before_evaluation", "notes",
    )
    write_csv(output / "tables" / "p0_01_dataset_and_protocol.csv", records, fields)


def linear_slope(rows: list[dict[str, str]], field: str) -> float:
    pairs = [
        (float(row["frame_index"]), float(row[field]))
        for row in rows
        if row.get(field) not in (None, "")
    ]
    if len(pairs) < 2:
        return math.nan
    mean_x = sum(x for x, _ in pairs) / len(pairs)
    mean_y = sum(y for _, y in pairs) / len(pairs)
    denominator = sum((x - mean_x) ** 2 for x, _ in pairs)
    return sum((x - mean_x) * (y - mean_y) for x, y in pairs) / denominator


def build_memory_table(root: Path, output: Path) -> None:
    source = root / "supplementary" / "tables"
    cells = read_csv(source / "table_s19_memory_decomposition.csv")
    traces = read_csv(source / "table_s20_memory_trace.csv")
    by_cell: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in traces:
        by_cell[row["cell"]].append(row)
    for rows in by_cell.values():
        rows.sort(key=lambda row: int(row["frame_index"]))

    output_rows = []
    for cell in cells:
        rows = by_cell[cell["cell"]]
        if len(rows) != int(cell["processed_frames"]):
            raise ValueError(f"memory trace coverage mismatch for {cell['cell']}")
        steady = rows[len(rows) // 2 :]
        final_output_mib = float(rows[-1]["retained_outputs_mib"])
        paired_cell = (
            cell["cell"].replace("accumulated", "release")
            if "accumulated" in cell["cell"]
            else cell["cell"].replace("release", "accumulated")
        )
        counterpart = next(item for item in cells if item["cell"] == paired_cell)
        output_rows.append(
            {
                "cell": cell["cell"],
                "cache_policy": cell["cache_policy"],
                "output_mode": cell["output_mode"],
                "processed_frames": cell["processed_frames"],
                "peak_allocated_mib": cell["peak_allocated_mb"],
                "peak_reserved_mib": cell["peak_reserved_mb"],
                "fps_inference": cell["fps_inference"],
                "wall_seconds": cell["wall_sec"],
                "final_output_tensor_mib": final_output_mib,
                "final_output_tensor_bytes": int(round(final_output_mib * 1024 ** 2)),
                "output_growth_mib_per_frame_all": linear_slope(rows, "retained_outputs_mib"),
                "aggregator_kv_growth_mib_per_frame_last_half": linear_slope(steady, "aggregator_kv_mib"),
                "camera_kv_growth_mib_per_frame_last_half": linear_slope(steady, "camera_kv_mib"),
                "allocated_growth_mib_per_frame_last_half": linear_slope(steady, "cuda_allocated_mib"),
                "reserved_growth_mib_per_frame_last_half": linear_slope(steady, "cuda_reserved_mib"),
                "pose_hash": cell["camera_pose_sha256"],
                "depth_hash": cell["depth_sha256"],
                "lifecycle_pair": paired_cell,
                "pose_hash_equal_with_lifecycle_pair": str(
                    cell["camera_pose_sha256"] == counterpart["camera_pose_sha256"]
                ).lower(),
                "depth_hash_equal_with_lifecycle_pair": str(
                    cell["depth_sha256"] == counterpart["depth_sha256"]
                ).lower(),
            }
        )
    write_csv(output / "tables" / "p0_08_memory_factorial_complete.csv", output_rows)


def random_sequence_means(rows: list[dict[str, str]]) -> dict[tuple[str, str], float]:
    grouped: dict[tuple[str, str], list[float]] = defaultdict(list)
    for row in rows:
        if row["method"].startswith("random4_seed"):
            grouped[(row["dataset"], row["sequence"])].append(float(row["abs_rel"]))
    if any(len(values) != 3 for values in grouped.values()):
        raise ValueError("Random-4 requires exactly three seeds per sequence")
    return {key: sum(values) / len(values) for key, values in grouped.items()}


def paired_absrel_rows(root: Path) -> list[dict]:
    rows = read_csv(root / "stage5a_same_budget_sequence_results.csv")
    index = {
        (row["dataset"], row["sequence"], row["method"]): float(row["abs_rel"])
        for row in rows
    }
    random_means = random_sequence_means(rows)
    proposed_keys = sorted(
        (dataset, sequence)
        for dataset, sequence, method in index
        if method == "proposed_k4"
    )
    output = []
    for dataset, sequence in proposed_keys:
        proposed = index[(dataset, sequence, "proposed_k4")]
        for control in CONTROL_ORDER:
            control_value = (
                random_means[(dataset, sequence)]
                if control == "random4_mean"
                else index[(dataset, sequence, control)]
            )
            output.append(
                {
                    "dataset": dataset,
                    "sequence": sequence,
                    "control": control,
                    "control_label": (
                        "Random-4 (3-seed mean)"
                        if control == "random4_mean"
                        else METHOD_LABELS[control]
                    ),
                    "control_abs_rel": control_value,
                    "k4_abs_rel": proposed,
                    "k4_advantage_control_minus_k4": control_value - proposed,
                    "k4_win": int(proposed < control_value - 1e-12),
                    "tie": int(abs(proposed - control_value) <= 1e-12),
                    "k4_loss": int(proposed > control_value + 1e-12),
                }
            )
    return output


def plot_paired_absrel(rows: list[dict], path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = [
        "Full", "Recent-4", "A+R-4", "Uniform-4", "Random-4"
    ]
    datasets = ("bonn", "sintel", "kitti")
    figure, axes = plt.subplots(1, 3, figsize=(11.2, 3.7), dpi=180)
    for axis, dataset in zip(axes, datasets):
        values = [
            [
                float(row["k4_advantage_control_minus_k4"])
                for row in rows
                if row["dataset"] == dataset and row["control"] == control
            ]
            for control in CONTROL_ORDER
        ]
        axis.axhline(0.0, color="black", linewidth=0.8, linestyle="--")
        axis.boxplot(values, tick_labels=labels, showmeans=True, widths=0.62)
        for x, group in enumerate(values, start=1):
            offsets = [((index % 7) - 3) * 0.018 for index in range(len(group))]
            axis.scatter([x + offset for offset in offsets], group, s=10, alpha=0.55)
        axis.set_title(dataset.upper())
        axis.tick_params(axis="x", rotation=34, labelsize=8)
        axis.set_ylabel("AbsRel(control) - AbsRel(K4)")
        axis.grid(axis="y", alpha=0.2)
    figure.suptitle("Paired sequence-level K4 advantage (positive favours K4)")
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, bbox_inches="tight")
    plt.close(figure)
    print(f"Wrote figure: {path}")


def write_manifest(root: Path, output: Path) -> None:
    summary_fields = set(read_csv(root / "stage5a_same_budget_results.csv")[0])
    required_depth = {
        "abs_rel", "sq_rel", "rmse", "log_rmse", "delta_1", "delta_2", "delta_3"
    }
    paired_rows = read_csv(root / "stage5a_paired_statistics.csv")
    same_budget_refreshed = required_depth <= summary_fields and {
        row["dataset"] for row in paired_rows if row["control"] == "full_cache"
    } == {"bonn", "sintel", "kitti"}
    selector_complete = (
        root / "eval_results/supplementary_selector_trace/p1_selector_overhead.csv"
    ).is_file()
    k8_controls_complete = (
        root / "eval_results/supplementary_k8_controls/k8_controls_paired.csv"
    ).is_file()
    k8_coverage_complete = (
        root / "eval_results/supplementary_k8_coverage/k8_temporal_coverage_summary.csv"
    ).is_file()
    rows = [
        {"id": "P0-01", "priority": "P0", "requirement": "dataset and evaluation protocol", "status": "complete_derived", "server_action": "none", "assets": "tables/p0_01_dataset_and_protocol.csv"},
        {"id": "P0-02", "priority": "P0", "requirement": "complete same-budget selection experiment", "status": "complete" if same_budget_refreshed else "awaiting_metric_refresh", "server_action": "none" if same_budget_refreshed else "CPU-only Stage5A finalize; no inference", "assets": "tables/p0_02_same_budget_summary_current.csv"},
        {"id": "P0-03", "priority": "P0", "requirement": "per-sequence results and paired statistics", "status": "complete_from_existing_results", "server_action": "refresh adds K4-vs-Full bootstrap row", "assets": "tables/p0_03_same_budget_sequences.csv; tables/p0_03_paired_absrel_values.csv; figures/p0_03_paired_absrel_boxplots.png"},
        {"id": "P0-04", "priority": "P0", "requirement": "secondary depth metrics", "status": "complete_for_Full_K4_K6_K8", "server_action": "P0-02 refresh extends same-budget controls", "assets": "tables/p0_04_depth_secondary_metrics_summary.csv; tables/p0_04_depth_secondary_metrics_sequences.csv"},
        {"id": "P0-05", "priority": "P0", "requirement": "complete pose and reconstruction metrics", "status": "complete", "server_action": "none", "assets": "tables/p0_05_*"},
        {"id": "P0-06", "priority": "P0", "requirement": "cross-task regret", "status": "complete", "server_action": "none", "assets": "tables/p0_06_*; figures/p0_06_cross_task_regret.pdf; methods/p0_06_normalized_regret_and_oracle_wins.md"},
        {"id": "P0-07", "priority": "P0", "requirement": "complete long-sequence results", "status": "complete", "server_action": "none", "assets": "tables/p0_07_long_sequence_complete.csv; figures/p0_07_long_sequence_scaling.pdf"},
        {"id": "P0-08", "priority": "P0", "requirement": "2x2 memory decomposition", "status": "complete_derived", "server_action": "none", "assets": "tables/p0_08_*; figures/p0_08_memory_decomposition.png"},
        {"id": "P0-09", "priority": "P0", "requirement": "long-sequence pose diagnostics", "status": "complete", "server_action": "none", "assets": "figures/p0_09_*"},
        {"id": "P0-10", "priority": "P0", "requirement": "exact implementation and pseudocode", "status": "complete_from_source", "server_action": "none", "assets": "methods/p0_10_algorithm_s1_cache_update.md"},
        {"id": "P1-01", "priority": "P1", "requirement": "cache selection visualisation", "status": "complete" if selector_complete else "partial_existing_timeline", "server_action": "none" if selector_complete else "run selector diagnostic once", "assets": "figures/p1_01_*"},
        {"id": "P1-02", "priority": "P1", "requirement": "same-budget component ablation", "status": "complete" if k8_controls_complete else "k4_complete_k8_controls_missing", "server_action": "none" if k8_controls_complete else "optional K8 Recent/non-hierarchical run", "assets": "tables/p0_02_*; tables/p1_02_*"},
        {"id": "P1-03", "priority": "P1", "requirement": "budget sweep", "status": "not_run", "server_action": "defer until P0 and P1-01 are frozen", "assets": "none"},
        {"id": "P1-04", "priority": "P1", "requirement": "selector measured overhead", "status": "complete" if selector_complete else "memory_bytes_available_latency_pending", "server_action": "none" if selector_complete else "produced by selector diagnostic", "assets": "tables/p0_08_memory_trace.csv; tables/p1_04_selector_overhead.csv"},
        {"id": "P1-05", "priority": "P1", "requirement": "matched hierarchical/non-hierarchical K8 temporal coverage", "status": "complete" if k8_coverage_complete else "not_run", "server_action": "none" if k8_coverage_complete else "one 110-view matched selector diagnostic", "assets": "tables/p1_05_*; figures/p1_05_*; raw/p1_05_*"},
    ]
    write_csv(output / "MATERIAL_STATUS.csv", rows)


def main() -> None:
    args = parse_args()
    root = args.repo_root.resolve()
    output = args.output_root
    if not output.is_absolute():
        output = root / output
    output.mkdir(parents=True, exist_ok=True)
    copy_existing_assets(root, output)
    build_protocol_table(root, output)
    build_memory_table(root, output)
    paired = paired_absrel_rows(root)
    write_csv(output / "tables" / "p0_03_paired_absrel_values.csv", paired)
    plot_paired_absrel(
        paired, output / "figures" / "p0_03_paired_absrel_boxplots.png"
    )
    write_manifest(root, output)


if __name__ == "__main__":
    main()
