#!/usr/bin/env python3
"""Plot the final long-sequence resource trends from Supplementary Table S15."""

from __future__ import annotations

import argparse
import csv
import math
import os
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import fmean

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/estreamvggt-matplotlib")

import matplotlib.pyplot as plt


METHODS = ("K4", "K6", "K8")
LENGTHS = (100, 250, 500, 1000)
COLORS = {
    "Full cache": "#4D4D4D",
    "K4": "#0072B2",
    "K6": "#009E73",
    "K8": "#D55E00",
}
MARKERS = {"K4": "o", "K6": "s", "K8": "^"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot GPU-memory and throughput trends versus sequence length."
    )
    parser.add_argument(
        "--input",
        default="supplementary/tables/table_s15_long_sequence_results.csv",
    )
    parser.add_argument("--output-dir", default="paper_assets/figures")
    parser.add_argument("--stem", default="fig_long_sequence_scaling")
    return parser.parse_args()


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def validate_and_aggregate(rows: list[dict[str, str]]):
    bounded = defaultdict(list)
    full_success = []
    full_oom = []
    for row in rows:
        method = row["method"]
        requested = int(row["num_frames"])
        if method in METHODS and row["status"] == "ok":
            bounded[(method, requested)].append(row)
        elif method == "Full cache" and row["status"] == "ok":
            full_success.append(row)
        elif method == "Full cache" and row["status"] == "failed":
            full_oom.append(row)

    expected_keys = {(method, length) for method in METHODS for length in LENGTHS}
    if set(bounded) != expected_keys:
        raise ValueError(
            "bounded method/length coverage mismatch; "
            f"missing={sorted(expected_keys-set(bounded))}, "
            f"extra={sorted(set(bounded)-expected_keys)}"
        )
    if any(len(bounded[key]) != 3 for key in expected_keys):
        raise ValueError("each bounded method/length cell must contain three sequences")
    if len(full_success) != 3 or {int(row["num_frames"]) for row in full_success} != {100}:
        raise ValueError("expected three successful 100-frame full-cache runs")
    if len(full_oom) != 3 or {int(row["num_frames"]) for row in full_oom} != {250}:
        raise ValueError("expected three failed 250-frame full-cache runs")
    if any("out of memory" not in row["error"].lower() for row in full_oom):
        raise ValueError("full-cache failures must be CUDA out-of-memory records")

    peaks = {
        method: [
            max(float(row["peak_allocated_mb"]) for row in bounded[(method, length)])
            / 1024.0
            for length in LENGTHS
        ]
        for method in METHODS
    }
    fps = {
        method: [
            fmean(float(row["fps_inference"]) for row in bounded[(method, length)])
            for length in LENGTHS
        ]
        for method in METHODS
    }
    full_100_peak = max(float(row["peak_allocated_mb"]) for row in full_success) / 1024.0
    full_100_fps = fmean(float(row["fps_inference"]) for row in full_success)
    processed = {int(row["processed_frames"]) for row in full_oom}
    if len(processed) != 1:
        raise ValueError(f"inconsistent full-cache OOM frames: {sorted(processed)}")
    oom_frame = next(iter(processed))
    oom_peak = max(float(row["peak_allocated_mb"]) for row in full_oom) / 1024.0

    for method in METHODS:
        if not math.isclose(peaks[method][2], peaks[method][3], abs_tol=1e-9):
            raise ValueError(f"{method} GPU peak is not flat from 500 to 1000 frames")
    return peaks, fps, full_100_peak, full_100_fps, oom_frame, oom_peak


def plot(rows: list[dict[str, str]], output_dir: Path, stem: str) -> tuple[Path, Path]:
    peaks, fps, full_peak, full_fps, oom_frame, oom_peak = validate_and_aggregate(rows)

    plt.rcParams.update(
        {
            "font.size": 9.5,
            "axes.labelsize": 10.5,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
        }
    )
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.25))

    for method in METHODS:
        axes[0].plot(
            LENGTHS,
            peaks[method],
            color=COLORS[method],
            marker=MARKERS[method],
            linewidth=1.8,
            markersize=5.2,
            label=method,
        )
        axes[1].plot(
            LENGTHS,
            fps[method],
            color=COLORS[method],
            marker=MARKERS[method],
            linewidth=1.8,
            markersize=5.2,
            label=method,
        )

    axes[0].plot(
        [100, oom_frame],
        [full_peak, oom_peak],
        color=COLORS["Full cache"],
        linestyle="--",
        marker="o",
        linewidth=1.5,
        markersize=5.0,
        label="Full cache",
    )
    axes[0].scatter(
        [oom_frame],
        [oom_peak],
        color="#CC0000",
        marker="x",
        s=72,
        linewidths=2.0,
        zorder=5,
    )
    axes[0].annotate(
        "OOM at frame 195\n(250 requested)",
        xy=(oom_frame, oom_peak),
        xytext=(315, 26.7),
        arrowprops={"arrowstyle": "->", "color": "#8B0000", "lw": 1.0},
        color="#8B0000",
        fontsize=8.3,
        ha="left",
    )
    axes[1].scatter(
        [100],
        [full_fps],
        color=COLORS["Full cache"],
        marker="o",
        s=30,
        label="Full cache",
        zorder=4,
    )

    axes[0].set_ylabel("Peak GPU memory (GiB) ↓")
    axes[1].set_ylabel("Inference throughput (FPS) ↑")
    for index, axis in enumerate(axes):
        axis.set_xlabel("Sequence prefix length (frames)")
        axis.set_xticks(LENGTHS)
        axis.set_xlim(60, 1040)
        axis.grid(alpha=0.22, linewidth=0.8)
        axis.text(
            0.01,
            0.98,
            f"({chr(ord('a') + index)})",
            transform=axis.transAxes,
            va="top",
            ha="left",
            fontweight="bold",
        )
    axes[0].set_ylim(6.8, 30.2)
    axes[1].set_ylim(2.6, 11.3)

    handles, labels = axes[0].get_legend_handles_labels()
    order = [labels.index(name) for name in ("Full cache", "K4", "K6", "K8")]
    fig.legend(
        [handles[index] for index in order],
        [labels[index] for index in order],
        ncol=4,
        loc="upper center",
        frameon=False,
        bbox_to_anchor=(0.5, 1.02),
    )
    fig.subplots_adjust(top=0.80, bottom=0.18, left=0.09, right=0.99, wspace=0.27)

    output_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = output_dir / f"{stem}.pdf"
    png_path = output_dir / f"{stem}.png"
    fixed_date = datetime(2000, 1, 1, tzinfo=timezone.utc)
    fig.savefig(
        pdf_path,
        bbox_inches="tight",
        metadata={
            "Title": "Long-sequence resource scaling",
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


def main() -> None:
    args = parse_args()
    pdf_path, png_path = plot(
        read_rows(Path(args.input)), Path(args.output_dir), args.stem
    )
    print(f"Wrote {pdf_path}")
    print(f"Wrote {png_path}")


if __name__ == "__main__":
    main()
