#!/usr/bin/env python3
"""Plot Stage 5B memory curves and the two decomposed contributions."""

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt


LABELS = {
    "full_accumulated": "Full + accumulated outputs",
    "full_release": "Full + streaming release",
    "k4_accumulated": "K4 + accumulated outputs",
    "k4_release": "K4 + streaming release",
}
COLORS = {
    "full_accumulated": "#b2182b", "full_release": "#ef8a62",
    "k4_accumulated": "#2166ac", "k4_release": "#67a9cf",
}


def read_csv(path):
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    root = args.repo_root.resolve()
    trace = read_csv(root / "stage5b_memory_trace.csv")
    contributions = read_csv(root / "stage5b_memory_contributions.csv")
    figure, axes = plt.subplots(1, 2, figsize=(10.2, 3.8), constrained_layout=True)
    for cell in LABELS:
        rows = [row for row in trace if row["cell"] == cell]
        axes[0].plot(
            [int(row["frame_index"]) + 1 for row in rows],
            [float(row["cuda_allocated_mib"]) / 1024.0 for row in rows],
            label=LABELS[cell], color=COLORS[cell], linewidth=2,
        )
    axes[0].set_xlabel("Processed frames")
    axes[0].set_ylabel("CUDA allocated memory (GiB)")
    axes[0].grid(alpha=0.25)
    axes[0].legend(frameon=False, fontsize=8)
    axes[0].text(
        0.01, 0.98, "(a)", transform=axes[0].transAxes,
        va="top", ha="left", fontweight="bold",
    )
    selected = [
        next(row for row in contributions if row["effect"] == "kv_pruning_with_streaming_release"),
        next(row for row in contributions if row["effect"] == "output_release_with_k4"),
    ]
    values = [float(row["peak_allocated_saved_mib"]) / 1024.0 for row in selected]
    bars = axes[1].bar(["KV pruning", "Output release"], values, color=["#2166ac", "#4daf4a"])
    axes[1].set_ylabel("Peak allocated memory saved (GiB)")
    axes[1].grid(axis="y", alpha=0.25)
    axes[1].text(
        0.99, 0.98, "(b)", transform=axes[1].transAxes,
        va="top", ha="right", fontweight="bold",
    )
    for bar, value in zip(bars, values):
        axes[1].text(bar.get_x() + bar.get_width() / 2, value, f"{value:.2f}", ha="center", va="bottom")
    output_dir = root / "paper_assets/figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    for suffix in ("pdf", "svg", "png"):
        path = output_dir / f"fig_stage5b_memory_decomposition.{suffix}"
        figure.savefig(path, dpi=300)
        print(f"Wrote {path}")
    plt.close(figure)


if __name__ == "__main__":
    main()
