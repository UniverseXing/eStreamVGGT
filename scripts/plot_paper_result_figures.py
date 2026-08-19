#!/usr/bin/env python3
"""Build the final figures replacing the Results tables in Sections 5.6.1/5.6.4."""

from __future__ import annotations

import argparse
import csv
import os
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/estreamvggt-matplotlib")

import matplotlib.pyplot as plt
import numpy as np


METHODS = ("Full cache", "K4", "K6", "K8")
BOUNDED = ("K4", "K6", "K8")
COLORS = {
    "Full cache": "#4D4D4D",
    "K4": "#0072B2",
    "K6": "#009E73",
    "K8": "#D55E00",
}
HATCHES = {"Full cache": "//", "K4": "", "K6": "", "K8": ""}
MARKERS = {"K4": "o", "K6": "s", "K8": "^"}
DATASETS = ("bonn", "kitti", "sintel")
DATASET_LABELS = {"bonn": "Bonn", "kitti": "KITTI", "sintel": "Sintel"}
POSE_CASES = (
    ("rgbd_dataset_freiburg1_room", 250, "F1-R/250"),
    ("rgbd_dataset_freiburg2_desk", 500, "F2-D/500"),
    (
        "rgbd_dataset_freiburg3_long_office_household",
        500,
        "F3-LO/500",
    ),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot final VideoDepth and representative pose comparisons."
    )
    parser.add_argument(
        "--video-depth",
        default="supplementary/tables/table_s01_video_depth_summary.csv",
    )
    parser.add_argument(
        "--long-sequence",
        default="supplementary/tables/table_s15_long_sequence_results.csv",
    )
    parser.add_argument("--output-dir", default="paper_assets/figures")
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def save_figure(fig, output_dir: Path, stem: str) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = output_dir / f"{stem}.pdf"
    png_path = output_dir / f"{stem}.png"
    fixed_date = datetime(2000, 1, 1, tzinfo=timezone.utc)
    fig.savefig(
        pdf_path,
        bbox_inches="tight",
        metadata={
            "Title": stem,
            "Creator": "eStreamVGGT",
            "Producer": "Matplotlib",
            "CreationDate": fixed_date,
            "ModDate": fixed_date,
        },
    )
    fig.savefig(
        png_path,
        dpi=300,
        bbox_inches="tight",
        metadata={"Software": "eStreamVGGT"},
    )
    plt.close(fig)
    return pdf_path, png_path


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.size": 9.2,
            "axes.labelsize": 10.0,
            "xtick.labelsize": 8.8,
            "ytick.labelsize": 8.5,
            "legend.fontsize": 9.0,
        }
    )


def plot_video_depth(rows: list[dict[str, str]], output_dir: Path):
    indexed = {(row["dataset"], row["method"]): row for row in rows}
    expected = {(dataset, method) for dataset in DATASETS for method in METHODS}
    if set(indexed) != expected:
        raise ValueError(
            "VideoDepth coverage mismatch; "
            f"missing={sorted(expected-set(indexed))}, "
            f"extra={sorted(set(indexed)-expected)}"
        )

    panels = (
        ("abs_rel", "AbsRel ↓", (0.0, 0.37)),
        ("delta_1", r"$\delta_1$ ↑", (0.0, 1.03)),
        ("fps_inference", "Inference throughput (FPS) ↑", (0.0, 10.0)),
        ("peak_allocated_gib", "Peak GPU memory (GiB) ↓", (0.0, 23.5)),
    )
    x = np.arange(len(DATASETS), dtype=float)
    width = 0.19
    offsets = np.linspace(-1.5 * width, 1.5 * width, len(METHODS))
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.25))
    for panel_index, (axis, (field, label, ylim)) in enumerate(
        zip(axes.flat, panels)
    ):
        for method, offset in zip(METHODS, offsets):
            values = [float(indexed[(dataset, method)][field]) for dataset in DATASETS]
            axis.bar(
                x + offset,
                values,
                width=width,
                color=COLORS[method],
                edgecolor="white" if method != "Full cache" else "#303030",
                linewidth=0.6,
                hatch=HATCHES[method],
                label=method,
                zorder=3,
            )
        axis.set_xticks(x, [DATASET_LABELS[dataset] for dataset in DATASETS])
        axis.set_ylabel(label)
        axis.set_ylim(*ylim)
        axis.grid(axis="y", alpha=0.22, linewidth=0.8, zorder=0)
        axis.text(
            0.01,
            0.98,
            f"({chr(ord('a') + panel_index)})",
            transform=axis.transAxes,
            va="top",
            ha="left",
            fontweight="bold",
        )
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        ncol=4,
        loc="upper center",
        frameon=False,
        bbox_to_anchor=(0.5, 1.01),
    )
    fig.subplots_adjust(
        top=0.88,
        bottom=0.09,
        left=0.10,
        right=0.99,
        hspace=0.33,
        wspace=0.28,
    )
    return save_figure(fig, output_dir, "fig_video_depth_results")


def plot_pose_cases(rows: list[dict[str, str]], output_dir: Path):
    indexed = {
        (row["sequence"], int(row["num_frames"]), row["method"]): row
        for row in rows
        if row["status"] == "ok"
    }
    expected = {
        (sequence, frames, method)
        for sequence, frames, _ in POSE_CASES
        for method in BOUNDED
    }
    missing = expected - set(indexed)
    if missing:
        raise ValueError(f"missing representative pose results: {sorted(missing)}")

    panels = (
        ("ate", "ATE ↓", False),
        ("rpe_trans", "Translation RPE ↓", True),
        ("rpe_rot_deg", "Rotation RPE (deg) ↓", True),
    )
    x = np.arange(len(POSE_CASES), dtype=float)
    offsets = {"K4": -0.16, "K6": 0.0, "K8": 0.16}
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 3.25))
    for panel_index, (axis, (field, label, log_scale)) in enumerate(
        zip(axes, panels)
    ):
        values_by_method = {
            method: [
                float(indexed[(sequence, frames, method)][field])
                for sequence, frames, _ in POSE_CASES
            ]
            for method in BOUNDED
        }
        for method in BOUNDED:
            values = values_by_method[method]
            axis.scatter(
                x + offsets[method],
                values,
                color=COLORS[method],
                marker=MARKERS[method],
                s=48,
                label=method,
                zorder=3,
            )
        for case_index in range(len(POSE_CASES)):
            case_values = {
                method: values_by_method[method][case_index] for method in BOUNDED
            }
            best = min(case_values.values())
            for method, value in case_values.items():
                if np.isclose(value, best, rtol=1e-10, atol=1e-12):
                    axis.scatter(
                        [x[case_index] + offsets[method]],
                        [value],
                        facecolors="none",
                        edgecolors="black",
                        linewidths=1.15,
                        s=85,
                        zorder=4,
                    )
        if log_scale:
            axis.set_yscale("log")
        axis.set_xticks(x, [label for _, _, label in POSE_CASES])
        axis.set_ylabel(label)
        axis.grid(axis="y", which="both", alpha=0.22, linewidth=0.8)
        axis.text(
            0.97,
            0.98,
            f"({chr(ord('a') + panel_index)})",
            transform=axis.transAxes,
            va="top",
            ha="right",
            fontweight="bold",
        )
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        ncol=3,
        loc="upper center",
        frameon=False,
        bbox_to_anchor=(0.5, 1.01),
    )
    fig.subplots_adjust(
        top=0.84,
        bottom=0.25,
        left=0.08,
        right=0.99,
        wspace=0.38,
    )
    return save_figure(fig, output_dir, "fig_pose_case_comparison")


def main() -> None:
    args = parse_args()
    configure_style()
    output_dir = Path(args.output_dir)
    outputs = (
        *plot_video_depth(read_csv(Path(args.video_depth)), output_dir),
        *plot_pose_cases(read_csv(Path(args.long_sequence)), output_dir),
    )
    for path in outputs:
        print(f"Wrote {path}")


if __name__ == "__main__":
    main()
