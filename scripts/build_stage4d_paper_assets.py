#!/usr/bin/env python3
"""Build frozen Stage 4D paper tables, figures, and provenance assets."""

import argparse
import csv
import hashlib
import json
import os
import os.path as osp
import platform
import sys
from datetime import datetime, timezone

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/streamvggt-matplotlib")

import matplotlib
import matplotlib.pyplot as plt
import numpy as np


REPO_ROOT = osp.abspath(osp.join(osp.dirname(__file__), ".."))
SRC_ROOT = osp.join(REPO_ROOT, "src")
if SRC_ROOT not in sys.path:
    sys.path.append(SRC_ROOT)

from eval.pose_evaluation.trajectory_metrics import evaluate_trajectory


METHODS = (
    "full_cache",
    "stage3_2_k4",
    "old_dino_k6",
    "temporal_binned_dino_k8",
)
BOUNDED = METHODS[1:]
METHOD_LABELS = {
    "full_cache": "Full cache",
    "stage3_2_k4": "DINO K4",
    "old_dino_k6": "Old DINO K6",
    "temporal_binned_dino_k8": "Temporal DINO K8",
}
METHOD_COLORS = {
    "full_cache": "#4D4D4D",
    "stage3_2_k4": "#0072B2",
    "old_dino_k6": "#009E73",
    "temporal_binned_dino_k8": "#D55E00",
}
SEQUENCES = (
    "rgbd_dataset_freiburg1_room",
    "rgbd_dataset_freiburg2_desk",
    "rgbd_dataset_freiburg3_long_office_household",
)
SEQUENCE_LABELS = {
    "rgbd_dataset_freiburg1_room": "Freiburg1 room",
    "rgbd_dataset_freiburg2_desk": "Freiburg2 desk",
    "rgbd_dataset_freiburg3_long_office_household": "Freiburg3 long office",
}
TRAJECTORY_CASES = (
    (
        "k8_local_pose_success",
        "rgbd_dataset_freiburg1_room",
        250,
        "K8 improves both global ATE and local RPE over K4.",
    ),
    (
        "k4_global_pose_advantage",
        "rgbd_dataset_freiburg2_desk",
        500,
        "K4 retains lower ATE while K8 has lower local RPE.",
    ),
    (
        "k8_global_drift_failure",
        "rgbd_dataset_freiburg3_long_office_household",
        500,
        "K8 local RPE improves but ATE is 2.34x K4.",
    ),
)


def parse_args():
    parser = argparse.ArgumentParser("Build Stage 4D paper assets")
    parser.add_argument(
        "--cross-task-summary", default="stage4b_cross_task_summary.csv"
    )
    parser.add_argument(
        "--cross-task-regret", default="stage4b_cross_task_regret.csv"
    )
    parser.add_argument("--method-roles", default="stage4b_method_roles.csv")
    parser.add_argument("--claim-audit", default="stage4b_claim_audit.csv")
    parser.add_argument("--pareto", default="stage4b_pareto.csv")
    parser.add_argument(
        "--paired-video-depth",
        default="stage4b_video_depth_paired_comparison.csv",
    )
    parser.add_argument("--stage4c-results", default="stage4c_results.csv")
    parser.add_argument("--stage4c-gate", default="stage4c_gate.csv")
    parser.add_argument(
        "--stage4c-results-root",
        default="eval_results/stage4c_tum_long",
    )
    parser.add_argument(
        "--stage4e-results", default="stage4e_a_sequence_results.csv"
    )
    parser.add_argument(
        "--output-root", default="paper_assets/stage4d"
    )
    parser.add_argument(
        "--results-output", default="stage4d_results.csv"
    )
    parser.add_argument(
        "--case-output", default="stage4d_case_audit.csv"
    )
    parser.add_argument(
        "--manifest-output", default="stage4d_asset_manifest.csv"
    )
    parser.add_argument(
        "--allow-missing-server-assets", action="store_true"
    )
    return parser.parse_args()


def read_csv(path):
    with open(path, newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path, fields, rows):
    os.makedirs(osp.dirname(osp.abspath(path)), exist_ok=True)
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} rows to {path}")


def number(row, key):
    value = None if row is None else row.get(key)
    return None if value in (None, "") else float(value)


def mean(values):
    values = [value for value in values if value is not None]
    return sum(values) / len(values) if values else None


def fmt(value, digits=3):
    if value in (None, ""):
        return "--"
    value = float(value)
    if abs(value) >= 1000:
        return f"{value:.0f}"
    return f"{value:.{digits}f}"


def latex_escape(value):
    text = str(value)
    for source, replacement in (
        ("\\", r"\textbackslash{}"),
        ("_", r"\_"),
        ("%", r"\%"),
        ("&", r"\&"),
        ("#", r"\#"),
    ):
        text = text.replace(source, replacement)
    return text


def write_latex(path, columns, headers, rows, caption, label):
    alignment = "l" + "r" * (len(columns) - 1)
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        rf"\begin{{tabular}}{{{alignment}}}",
        r"\toprule",
        " & ".join(latex_escape(item) for item in headers) + r" \\",
        r"\midrule",
    ]
    for row in rows:
        lines.append(
            " & ".join(latex_escape(row[column]) for column in columns)
            + r" \\"
        )
    lines.extend(
        (
            r"\bottomrule",
            r"\end{tabular}",
            rf"\caption{{{latex_escape(caption)}}}",
            rf"\label{{{label}}}",
            r"\end{table}",
        )
    )
    with open(path, "w") as handle:
        handle.write("\n".join(lines) + "\n")


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def save_figure(fig, figures_dir, stem, top=0.97):
    fig.tight_layout(rect=(0.0, 0.0, 1.0, top))
    outputs = []
    for extension in ("png", "pdf"):
        path = osp.join(figures_dir, f"{stem}.{extension}")
        fig.savefig(path, dpi=220, bbox_inches="tight")
        outputs.append(path)
    plt.close(fig)
    return outputs


def run_directory(row, results_root):
    candidates = []
    if row.get("result_dir"):
        candidates.append(row["result_dir"])
    candidates.append(
        osp.join(
            results_root,
            row["method"],
            row["sequence"],
            str(int(float(row["num_frames"]))),
        )
    )
    return next((path for path in candidates if osp.isdir(path)), candidates[-1])


def plot_video_depth_pareto(rows, figures_dir):
    datasets = ("bonn", "kitti", "sintel")
    fig, axes = plt.subplots(1, 3, figsize=(12.0, 3.6))
    for axis, dataset in zip(axes, datasets):
        selected = [row for row in rows if row["dataset"] == dataset]
        for row in selected:
            method = row["method"]
            axis.scatter(
                number(row, "max_peak_allocated_mb") / 1024.0,
                number(row, "abs_rel"),
                s=65,
                color=METHOD_COLORS[method],
                marker="o" if row["pareto_absrel_allocated_time"] == "yes" else "X",
                label=METHOD_LABELS[method],
                zorder=3,
            )
        axis.set_title(dataset.upper())
        axis.set_xlabel("Peak allocated (GiB)")
        axis.grid(alpha=0.25)
    axes[0].set_ylabel("AbsRel ↓")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=4, loc="upper center", frameon=False)
    fig.subplots_adjust(top=0.78)
    return save_figure(
        fig, figures_dir, "fig_video_depth_pareto", top=0.82
    )


def plot_cross_task_regret(role_rows, figures_dir):
    tasks = (
        ("video_depth_mean_regret", "VideoDepth"),
        ("pose_mean_regret", "Pose"),
        ("static_recon_mean_regret", "Static recon"),
        ("dynamic_recon_mean_regret", "Dynamic recon"),
    )
    rows = [row for row in role_rows if row["method"] in BOUNDED]
    x = np.arange(len(tasks))
    width = 0.24
    fig, axis = plt.subplots(figsize=(8.2, 4.1))
    for index, row in enumerate(rows):
        values = [number(row, key) for key, _ in tasks]
        axis.bar(
            x + (index - 1) * width,
            values,
            width,
            color=METHOD_COLORS[row["method"]],
            label=METHOD_LABELS[row["method"]],
        )
    axis.set_xticks(x, [label for _, label in tasks])
    axis.set_ylabel("Mean normalized regret ↓")
    axis.grid(axis="y", alpha=0.25)
    axis.legend(frameon=False, ncol=3)
    return save_figure(fig, figures_dir, "fig_cross_task_regret")


def plot_stage4c_scaling(rows, figures_dir):
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.8))
    for method in BOUNDED:
        selected = [
            row
            for row in rows
            if row["method"] == method and row["status"] == "ok"
        ]
        lengths = sorted({int(row["num_frames"]) for row in selected})
        peaks = [
            max(
                number(row, "peak_allocated_mb")
                for row in selected
                if int(row["num_frames"]) == length
            )
            / 1024.0
            for length in lengths
        ]
        fps = [
            mean(
                number(row, "fps_inference")
                for row in selected
                if int(row["num_frames"]) == length
            )
            for length in lengths
        ]
        axes[0].plot(
            lengths,
            peaks,
            marker="o",
            color=METHOD_COLORS[method],
            label=METHOD_LABELS[method],
        )
        axes[1].plot(
            lengths,
            fps,
            marker="o",
            color=METHOD_COLORS[method],
            label=METHOD_LABELS[method],
        )
    full_ok = [
        row
        for row in rows
        if row["method"] == "full_cache" and row["status"] == "ok"
    ]
    full_failed = [
        row
        for row in rows
        if row["method"] == "full_cache" and row["status"] != "ok"
    ]
    axes[0].scatter(
        [int(row["num_frames"]) for row in full_ok],
        [number(row, "peak_allocated_mb") / 1024.0 for row in full_ok],
        color=METHOD_COLORS["full_cache"],
        label=METHOD_LABELS["full_cache"],
    )
    axes[0].scatter(
        [int(float(row["processed_frames"])) for row in full_failed],
        [number(row, "peak_allocated_mb") / 1024.0 for row in full_failed],
        color="#CC0000",
        marker="x",
        s=65,
        label="Full cache OOM",
    )
    axes[1].scatter(
        [int(row["num_frames"]) for row in full_ok],
        [number(row, "fps_inference") for row in full_ok],
        color=METHOD_COLORS["full_cache"],
        label=METHOD_LABELS["full_cache"],
    )
    axes[0].axhline(12.0, color="#777777", linestyle="--", linewidth=1)
    axes[0].set_ylabel("Peak allocated (GiB)")
    axes[1].set_ylabel("Inference FPS ↑")
    for axis in axes:
        axis.set_xlabel("Requested / processed frames")
        axis.grid(alpha=0.25)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=3, loc="upper center", frameon=False)
    fig.subplots_adjust(top=0.78)
    return save_figure(
        fig, figures_dir, "fig_stage4c_scaling", top=0.82
    )


def plot_stage4c_pose(rows, figures_dir):
    metrics = (
        ("ate", "ATE ↓"),
        ("rpe_trans", "Translation RPE ↓"),
        ("rpe_rot_deg", "Rotation RPE (deg) ↓"),
    )
    fig, axes = plt.subplots(3, 3, figsize=(11.0, 9.0))
    for row_index, sequence in enumerate(SEQUENCES):
        for column_index, (metric, label) in enumerate(metrics):
            axis = axes[row_index, column_index]
            for method in BOUNDED:
                selected = sorted(
                    (
                        row
                        for row in rows
                        if row["method"] == method
                        and row["sequence"] == sequence
                        and row["status"] == "ok"
                    ),
                    key=lambda row: int(row["num_frames"]),
                )
                axis.plot(
                    [int(row["num_frames"]) for row in selected],
                    [number(row, metric) for row in selected],
                    marker="o",
                    color=METHOD_COLORS[method],
                    label=METHOD_LABELS[method],
                )
            axis.set_yscale("log")
            axis.grid(alpha=0.25)
            if row_index == 0:
                axis.set_title(label)
            if column_index == 0:
                axis.set_ylabel(SEQUENCE_LABELS[sequence])
            if row_index == len(SEQUENCES) - 1:
                axis.set_xlabel("Frames")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=3, loc="upper center", frameon=False)
    fig.subplots_adjust(top=0.93)
    return save_figure(
        fig, figures_dir, "fig_stage4c_pose_scaling", top=0.92
    )


def plot_stage4e_failure(rows, figures_dir):
    direct = sorted(
        (
            row
            for row in rows
            if row["variant"] == "direct_k4_geometry_k8_pose"
        ),
        key=lambda row: (row["sequence"], int(row["num_frames"])),
    )
    component = sorted(
        (
            row
            for row in rows
            if row["variant"]
            == "component_k4_translation_k8_rotation"
        ),
        key=lambda row: (row["sequence"], int(row["num_frames"])),
    )
    labels = [
        f"{SEQUENCE_LABELS[row['sequence']].replace('Freiburg', 'F')}\n"
        f"{row['num_frames']}"
        for row in direct
    ]
    fig, axes = plt.subplots(2, 1, figsize=(10.5, 6.5), sharex=True)
    axes[0].bar(
        np.arange(len(direct)),
        [number(row, "ate_ratio_to_k4") for row in direct],
        color=METHOD_COLORS["temporal_binned_dino_k8"],
    )
    axes[0].set_ylabel("Direct: ATE / K4 ATE")
    axes[1].bar(
        np.arange(len(component)),
        [number(row, "rpe_trans_ratio_to_k8") for row in component],
        color="#CC79A7",
    )
    axes[1].set_ylabel("Component: trans RPE / K8")
    for axis in axes:
        axis.axhline(1.10, color="#CC0000", linestyle="--", linewidth=1.3)
        axis.grid(axis="y", alpha=0.25)
    axes[1].set_xticks(np.arange(len(labels)), labels, rotation=35, ha="right")
    return save_figure(fig, figures_dir, "fig_stage4e_fusion_failure")


def load_case_trajectories(stage4c_rows, results_root, sequence, frames):
    indexed = {
        (row["method"], row["sequence"], int(row["num_frames"])): row
        for row in stage4c_rows
    }
    payloads = {}
    for method in BOUNDED:
        row = indexed[(method, sequence, frames)]
        path = osp.join(run_directory(row, results_root), "trajectory.npz")
        if not osp.isfile(path):
            raise FileNotFoundError(path)
        with np.load(path, allow_pickle=False) as payload:
            gt = np.asarray(payload["gt_c2w"], dtype=np.float64)
            pred = np.asarray(payload["pred_c2w"], dtype=np.float64)
        evaluated = evaluate_trajectory(gt, pred)
        payloads[method] = {
            "path": path,
            "gt": gt,
            "aligned": np.asarray(
                evaluated["pred_c2w_aligned"], dtype=np.float64
            ),
        }
    reference = payloads[BOUNDED[0]]["gt"]
    for method in BOUNDED[1:]:
        if not np.allclose(reference, payloads[method]["gt"], atol=1e-12):
            raise ValueError(f"GT mismatch for {sequence}/{frames}/{method}")
    return payloads


def best_projection_axes(gt_positions):
    variances = np.var(gt_positions, axis=0)
    axes = np.argsort(variances)[-2:][::-1]
    return int(axes[0]), int(axes[1])


def plot_trajectory_cases(stage4c_rows, results_root, figures_dir):
    fig, axes = plt.subplots(1, len(TRAJECTORY_CASES), figsize=(13.2, 4.0))
    source_paths = []
    case_rows = []
    for axis, (case_id, sequence, frames, rationale) in zip(
        axes, TRAJECTORY_CASES
    ):
        payloads = load_case_trajectories(
            stage4c_rows, results_root, sequence, frames
        )
        gt = payloads[BOUNDED[0]]["gt"][:, :3, 3]
        first_axis, second_axis = best_projection_axes(gt)
        axis.plot(
            gt[:, first_axis],
            gt[:, second_axis],
            color="black",
            linewidth=2.0,
            label="Ground truth",
        )
        for method in BOUNDED:
            positions = payloads[method]["aligned"][:, :3, 3]
            axis.plot(
                positions[:, first_axis],
                positions[:, second_axis],
                color=METHOD_COLORS[method],
                linewidth=1.2,
                label=METHOD_LABELS[method],
            )
            source_paths.append(payloads[method]["path"])
        axis.set_title(f"{SEQUENCE_LABELS[sequence]}\n{frames} frames")
        axis.set_xlabel(("x", "y", "z")[first_axis] + " (m)")
        axis.set_ylabel(("x", "y", "z")[second_axis] + " (m)")
        axis.axis("equal")
        axis.grid(alpha=0.2)
        case_rows.append(
            {
                "case_id": case_id,
                "asset_type": "trajectory",
                "sequence": sequence,
                "num_frames": frames,
                "rationale": rationale,
                "required_methods": " ".join(BOUNDED),
                "source_available": "yes",
            }
        )
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=4, loc="upper center", frameon=False)
    fig.subplots_adjust(top=0.76)
    outputs = save_figure(
        fig, figures_dir, "fig_stage4c_trajectories", top=0.78
    )
    return outputs, source_paths, case_rows


def plot_cache_timeline(stage4c_rows, results_root, figures_dir):
    sequence = "rgbd_dataset_freiburg3_long_office_household"
    frames = 1000
    indexed = {
        (row["method"], row["sequence"], int(row["num_frames"])): row
        for row in stage4c_rows
    }
    fig, axes = plt.subplots(3, 1, figsize=(10.5, 8.2), sharex=True)
    source_paths = []
    for axis, method in zip(axes, BOUNDED):
        row = indexed[(method, sequence, frames)]
        path = osp.join(run_directory(row, results_root), "memory_trace.json")
        if not osp.isfile(path):
            raise FileNotFoundError(path)
        with open(path) as handle:
            trace = json.load(handle)
        source_paths.append(path)
        current = []
        retained = []
        for item in trace:
            step = int(item["frame_index"])
            for frame_id in item.get("retained_frame_ids", []):
                current.append(step)
                retained.append(int(frame_id))
        axis.scatter(
            current,
            retained,
            s=2.2,
            alpha=0.55,
            color=METHOD_COLORS[method],
            rasterized=True,
        )
        axis.plot([0, frames], [0, frames], color="#888888", linewidth=0.8)
        axis.set_ylabel(METHOD_LABELS[method])
        axis.grid(alpha=0.15)
    axes[-1].set_xlabel("Current frame")
    fig.supylabel("Retained frame ID")
    outputs = save_figure(fig, figures_dir, "fig_stage4c_cache_timeline")
    case = {
        "case_id": "temporal_retention_timeline",
        "asset_type": "cache_timeline",
        "sequence": sequence,
        "num_frames": frames,
        "rationale": (
            "Compare frozen K4/K6/K8 retained-frame histories at 1000 frames."
        ),
        "required_methods": " ".join(BOUNDED),
        "source_available": "yes",
    }
    return outputs, source_paths, [case]


def build_video_depth_table(pareto_rows, paired_rows, tables_dir):
    by_dataset_method = {
        (row["dataset"], row["method"]): row for row in pareto_rows
    }
    significance = {}
    for row in paired_rows:
        if (
            row["metric"] == "abs_rel"
            and row["method_a"] == "full_cache"
            and row["method_b"] in METHODS
        ):
            value = row["significance"]
            if value == "A_BETTER":
                value = "full_better"
            elif value == "B_BETTER":
                value = "candidate_better"
            significance[(row["dataset"], row["method_b"])] = value
    output = []
    for dataset in ("bonn", "kitti", "sintel"):
        full = by_dataset_method[(dataset, "full_cache")]
        for method in METHODS:
            row = by_dataset_method[(dataset, method)]
            output.append(
                {
                    "dataset": dataset,
                    "method": method,
                    "method_label": METHOD_LABELS[method],
                    "abs_rel": number(row, "abs_rel"),
                    "delta_1": number(row, "delta_1"),
                    "fps": number(row, "fps_inference"),
                    "peak_allocated_gib": number(
                        row, "max_peak_allocated_mb"
                    )
                    / 1024.0,
                    "memory_reduction_vs_full_pct": (
                        100.0
                        * (
                            1.0
                            - number(row, "max_peak_allocated_mb")
                            / number(full, "max_peak_allocated_mb")
                        )
                    ),
                    "speedup_vs_full": (
                        number(row, "fps_inference")
                        / number(full, "fps_inference")
                    ),
                    "pareto": row["pareto_absrel_allocated_time"],
                    "absrel_significance_vs_full": (
                        "reference"
                        if method == "full_cache"
                        else significance.get((dataset, method), "")
                    ),
                }
            )
    fields = tuple(output[0])
    csv_path = osp.join(tables_dir, "table_video_depth.csv")
    write_csv(csv_path, fields, output)
    latex_rows = [
        {
            "dataset": row["dataset"].upper(),
            "method": row["method_label"],
            "abs_rel": fmt(row["abs_rel"], 4),
            "delta_1": fmt(row["delta_1"], 4),
            "fps": fmt(row["fps"], 2),
            "memory": fmt(row["peak_allocated_gib"], 2),
        }
        for row in output
    ]
    tex_path = osp.join(tables_dir, "table_video_depth.tex")
    write_latex(
        tex_path,
        ("dataset", "method", "abs_rel", "delta_1", "fps", "memory"),
        ("Dataset", "Method", "AbsRel ↓", "δ1 ↑", "FPS ↑", "GPU GiB ↓"),
        latex_rows,
        "Frozen RTX 6000 Ada VideoDepth results.",
        "tab:video_depth",
    )
    return [csv_path, tex_path]


def build_role_table(role_rows, tables_dir):
    output = []
    for row in role_rows:
        output.append(
            {
                "method": row["method"],
                "method_label": METHOD_LABELS[row["method"]],
                "final_role": row["final_role"],
                "status": row["status"],
                "benchmark_wins": row["primary_oracle_wins"],
                "macro_regret": row["mean_macro_primary_regret"],
                "video_depth_regret": row["video_depth_mean_regret"],
                "pose_regret": row["pose_mean_regret"],
                "static_recon_regret": row["static_recon_mean_regret"],
                "dynamic_recon_regret": row["dynamic_recon_mean_regret"],
                "max_peak_allocated_mb": row["max_peak_allocated_mb"],
            }
        )
    fields = tuple(output[0])
    csv_path = osp.join(tables_dir, "table_method_roles.csv")
    write_csv(csv_path, fields, output)
    latex_rows = [
        {
            "method": row["method_label"],
            "role": row["final_role"],
            "wins": row["benchmark_wins"] or "--",
            "regret": fmt(row["macro_regret"], 3),
            "memory": fmt(row["max_peak_allocated_mb"], 0),
        }
        for row in output
    ]
    tex_path = osp.join(tables_dir, "table_method_roles.tex")
    write_latex(
        tex_path,
        ("method", "role", "wins", "regret", "memory"),
        ("Method", "Frozen role", "Wins", "Macro regret ↓", "Peak MiB ↓"),
        latex_rows,
        "Frozen cross-task method roles.",
        "tab:method_roles",
    )
    return [csv_path, tex_path]


def build_long_table(rows, tables_dir):
    output = []
    for method in METHODS:
        lengths = sorted(
            {
                int(row["num_frames"])
                for row in rows
                if row["method"] == method
            }
        )
        for length in lengths:
            selected = [
                row
                for row in rows
                if row["method"] == method
                and int(row["num_frames"]) == length
            ]
            successful = [row for row in selected if row["status"] == "ok"]
            output.append(
                {
                    "method": method,
                    "method_label": METHOD_LABELS[method],
                    "num_frames": length,
                    "num_runs": len(selected),
                    "num_ok": len(successful),
                    "num_failed": len(selected) - len(successful),
                    "mean_ate": mean(number(row, "ate") for row in successful),
                    "mean_rpe_trans": mean(
                        number(row, "rpe_trans") for row in successful
                    ),
                    "mean_rpe_rot_deg": mean(
                        number(row, "rpe_rot_deg") for row in successful
                    ),
                    "mean_fps": mean(
                        number(row, "fps_inference") for row in successful
                    ),
                    "max_peak_allocated_mb": max(
                        (
                            number(row, "peak_allocated_mb")
                            for row in selected
                            if number(row, "peak_allocated_mb") is not None
                        ),
                        default=None,
                    ),
                    "max_rss_peak_mib": max(
                        (
                            number(row, "rss_peak_mib")
                            for row in successful
                            if number(row, "rss_peak_mib") is not None
                        ),
                        default=None,
                    ),
                }
            )
    fields = tuple(output[0])
    csv_path = osp.join(tables_dir, "table_long_sequence.csv")
    write_csv(csv_path, fields, output)
    latex_rows = [
        {
            "method": row["method_label"],
            "frames": row["num_frames"],
            "ok": f"{row['num_ok']}/{row['num_runs']}",
            "ate": fmt(row["mean_ate"], 3),
            "rot": fmt(row["mean_rpe_rot_deg"], 2),
            "fps": fmt(row["mean_fps"], 2),
            "memory": fmt(row["max_peak_allocated_mb"], 0),
        }
        for row in output
    ]
    tex_path = osp.join(tables_dir, "table_long_sequence.tex")
    write_latex(
        tex_path,
        ("method", "frames", "ok", "ate", "rot", "fps", "memory"),
        (
            "Method",
            "Frames",
            "Runs",
            "ATE ↓",
            "Rot. RPE ↓",
            "FPS ↑",
            "Peak MiB ↓",
        ),
        latex_rows,
        "Frozen unseen long-sequence results.",
        "tab:long_sequence",
    )
    return [csv_path, tex_path]


def build_cross_task_table(rows, tables_dir):
    fields = (
        "task",
        "dataset",
        "method",
        "evaluation_unit",
        "num_units",
        "primary_metric",
        "primary_value",
        "secondary_metric",
        "secondary_value",
        "fps_inference",
        "max_peak_allocated_mb",
        "coverage_ok",
    )
    output = [{key: row.get(key, "") for key in fields} for row in rows]
    path = osp.join(tables_dir, "table_cross_task.csv")
    write_csv(path, fields, output)
    return [path]


def build_findings(
    stage4c_rows,
    stage4e_rows,
    claim_rows,
    results_output,
):
    direct = [
        row
        for row in stage4e_rows
        if row["variant"] == "direct_k4_geometry_k8_pose"
    ]
    component = [
        row
        for row in stage4e_rows
        if row["variant"] == "component_k4_translation_k8_rotation"
    ]
    k4_rows = [
        row
        for row in stage4c_rows
        if row["method"] == "stage3_2_k4" and row["status"] == "ok"
    ]
    k8_rows = [
        row
        for row in stage4c_rows
        if row["method"] == "temporal_binned_dino_k8"
        and row["status"] == "ok"
    ]
    findings = [
        {
            "finding_id": "frozen_claim_audit",
            "status": "PASS",
            "evidence": (
                f"{sum(row['status'] == 'PASS' for row in claim_rows)} PASS, "
                f"{sum(row['status'] == 'PASS_LIMITED' for row in claim_rows)} "
                "PASS_LIMITED claims"
            ),
            "allowed_wording": (
                "Claims follow the frozen Stage 4B audit and limitations."
            ),
            "forbidden_wording": (
                "Do not upgrade PASS_LIMITED claims to universal superiority."
            ),
        },
        {
            "finding_id": "bounded_1000_frame_system",
            "status": "PASS",
            "evidence": (
                f"K4 peak={max(number(row, 'peak_allocated_mb') for row in k4_rows):.2f} "
                f"MiB; K8 peak={max(number(row, 'peak_allocated_mb') for row in k8_rows):.2f} "
                "MiB; 500-to-1000 GPU growth=0 MiB."
            ),
            "allowed_wording": (
                "The bounded streaming configurations plateau through 1000 frames."
            ),
            "forbidden_wording": (
                "Do not claim quality stability from memory stability."
            ),
        },
        {
            "finding_id": "full_cache_failure_ceiling",
            "status": "PASS",
            "evidence": (
                "Full cache succeeds at 100 frames and OOMs near 195 processed "
                "frames on all three unseen TUM sequences."
            ),
            "allowed_wording": (
                "Bounded caching extends execution beyond the full-cache ceiling."
            ),
            "forbidden_wording": (
                "Do not use failed full-cache prefixes as quality comparisons."
            ),
        },
        {
            "finding_id": "naive_pose_fusion",
            "status": "FAIL_EXPECTED_STOP",
            "evidence": (
                "Direct K8 max ATE/K4="
                f"{max(number(row, 'ate_ratio_to_k4') for row in direct):.3f}; "
                "component max translation-RPE/K8="
                f"{max(number(row, 'rpe_trans_ratio_to_k8') for row in component):.3f}."
            ),
            "allowed_wording": (
                "K4 global position and K8 local pose advantages are not "
                "composable by naive output-level fusion."
            ),
            "forbidden_wording": (
                "Do not claim a validated hybrid or online dual-branch method."
            ),
        },
    ]
    fields = (
        "finding_id",
        "status",
        "evidence",
        "allowed_wording",
        "forbidden_wording",
    )
    write_csv(results_output, fields, findings)


def build_source_manifest(paths, output_root):
    unique = sorted({osp.abspath(path) for path in paths})
    payload = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "stage": "4D",
        "new_model_inference": False,
        "selector_or_threshold_tuning": False,
        "python": platform.python_version(),
        "numpy": np.__version__,
        "matplotlib": matplotlib.__version__,
        "sources": [
            {
                "path": path,
                "size_bytes": osp.getsize(path),
                "sha256": sha256(path),
            }
            for path in unique
        ],
    }
    path = osp.join(output_root, "source_manifest.json")
    with open(path, "w") as handle:
        json.dump(payload, handle, indent=2)
    return path


def build_asset_manifest(output_root, manifest_output):
    rows = []
    for directory, _, filenames in os.walk(output_root):
        for filename in sorted(filenames):
            path = osp.join(directory, filename)
            relative = osp.relpath(path, output_root)
            category = relative.split(os.sep, 1)[0]
            rows.append(
                {
                    "relative_path": relative,
                    "category": category,
                    "size_bytes": osp.getsize(path),
                    "sha256": sha256(path),
                }
            )
    rows.sort(key=lambda row: row["relative_path"])
    fields = ("relative_path", "category", "size_bytes", "sha256")
    write_csv(manifest_output, fields, rows)


def main():
    args = parse_args()
    tables_dir = osp.join(args.output_root, "tables")
    figures_dir = osp.join(args.output_root, "figures")
    os.makedirs(tables_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)

    source_paths = [
        args.cross_task_summary,
        args.cross_task_regret,
        args.method_roles,
        args.claim_audit,
        args.pareto,
        args.paired_video_depth,
        args.stage4c_results,
        args.stage4c_gate,
        args.stage4e_results,
    ]
    for path in source_paths:
        if not osp.isfile(path):
            raise FileNotFoundError(path)

    cross_task = read_csv(args.cross_task_summary)
    regret = read_csv(args.cross_task_regret)
    roles = read_csv(args.method_roles)
    claims = read_csv(args.claim_audit)
    pareto = read_csv(args.pareto)
    paired = read_csv(args.paired_video_depth)
    stage4c = read_csv(args.stage4c_results)
    stage4e = read_csv(args.stage4e_results)

    build_video_depth_table(pareto, paired, tables_dir)
    build_role_table(roles, tables_dir)
    build_long_table(stage4c, tables_dir)
    build_cross_task_table(cross_task, tables_dir)
    plot_video_depth_pareto(pareto, figures_dir)
    plot_cross_task_regret(roles, figures_dir)
    plot_stage4c_scaling(stage4c, figures_dir)
    plot_stage4c_pose(stage4c, figures_dir)
    plot_stage4e_failure(stage4e, figures_dir)

    case_rows = []
    server_sources = []
    try:
        _, paths, cases = plot_trajectory_cases(
            stage4c, args.stage4c_results_root, figures_dir
        )
        server_sources.extend(paths)
        case_rows.extend(cases)
        _, paths, cases = plot_cache_timeline(
            stage4c, args.stage4c_results_root, figures_dir
        )
        server_sources.extend(paths)
        case_rows.extend(cases)
    except (FileNotFoundError, KeyError) as error:
        if not args.allow_missing_server_assets:
            raise
        print(f"Skipping server-only Stage 4D assets: {error}")
        for case_id, sequence, frames, rationale in TRAJECTORY_CASES:
            case_rows.append(
                {
                    "case_id": case_id,
                    "asset_type": "trajectory",
                    "sequence": sequence,
                    "num_frames": frames,
                    "rationale": rationale,
                    "required_methods": " ".join(BOUNDED),
                    "source_available": "no",
                }
            )
        case_rows.append(
            {
                "case_id": "temporal_retention_timeline",
                "asset_type": "cache_timeline",
                "sequence": SEQUENCES[-1],
                "num_frames": 1000,
                "rationale": "Compare K4/K6/K8 retained-frame histories.",
                "required_methods": " ".join(BOUNDED),
                "source_available": "no",
            }
        )

    case_fields = (
        "case_id",
        "asset_type",
        "sequence",
        "num_frames",
        "rationale",
        "required_methods",
        "source_available",
    )
    write_csv(args.case_output, case_fields, case_rows)
    build_findings(stage4c, stage4e, claims, args.results_output)
    build_source_manifest(source_paths + server_sources, args.output_root)
    build_asset_manifest(args.output_root, args.manifest_output)
    print(
        "Stage 4D build completed without model inference; "
        f"cross-task regret rows={len(regret)}."
    )


if __name__ == "__main__":
    main()
