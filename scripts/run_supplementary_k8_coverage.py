#!/usr/bin/env python3
"""Compare temporal-bin coverage of Hierarchical K8 and Non-hierarchical DINO-8."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/estreamvggt-k8-coverage-matplotlib")

import matplotlib.pyplot as plt
import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))
from run_supplementary_selector_trace import (  # noqa: E402
    run_method,
    selected_images,
    write_payload,
)


METHODS = (
    (
        "hierarchical_k8",
        "Hierarchical K8",
        8,
        "anchor_recent_dino_diverse_k8",
    ),
    (
        "nonhierarchical_dino8",
        "Non-hierarchical DINO-8",
        8,
        "anchor_recent_dino_diverse",
    ),
)
TEMPORAL_BINS = ("recent", "near", "middle", "long")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument(
        "--images-dir", type=Path, default=Path("data/eval/7scenes/chess/seq-01")
    )
    parser.add_argument("--image-glob", default="*.color.png")
    parser.add_argument("--sampling-stride", type=int, default=5)
    parser.add_argument("--max-frames", type=int, default=110)
    parser.add_argument(
        "--steady-start-frame",
        type=int,
        default=50,
        help="One-based first view included in steady-state coverage statistics.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("eval_results/supplementary_k8_coverage"),
    )
    return parser.parse_args()


def temporal_bin(age: int) -> str:
    if age <= 3:
        return "recent"
    if age <= 15:
        return "near"
    if age <= 47:
        return "middle"
    return "long"


def coverage_rows(results: list[dict], steady_start_frame: int) -> list[dict]:
    rows = []
    for result in results:
        for trace in result["memory_trace"]:
            current = int(trace["frame_index"])
            retained = [int(value) for value in trace["retained_frame_ids"]]
            nonanchor = [value for value in retained if value != 0]
            assignments = {name: [] for name in TEMPORAL_BINS}
            for frame_id in nonanchor:
                assignments[temporal_bin(current - frame_id)].append(frame_id)
            covered = {name: bool(assignments[name]) for name in TEMPORAL_BINS}
            rows.append(
                {
                    "method": result["method"],
                    "method_label": result["method_label"],
                    "cache_policy": result["cache_policy"],
                    "frame_number": current + 1,
                    "frame_id": current,
                    "steady_state": "yes" if current + 1 >= steady_start_frame else "no",
                    "retained_frame_ids": json.dumps(retained),
                    "anchor_covered": int(0 in retained),
                    **{f"{name}_covered": int(covered[name]) for name in TEMPORAL_BINS},
                    **{
                        f"{name}_frame_ids": json.dumps(assignments[name])
                        for name in TEMPORAL_BINS
                    },
                    "covered_temporal_bins": sum(covered.values()),
                    "all_four_temporal_bins_covered": int(all(covered.values())),
                    "oldest_nonanchor_age": max(
                        (current - frame_id for frame_id in nonanchor), default=0
                    ),
                }
            )
    return rows


def summary_rows(rows: list[dict], steady_start_frame: int) -> list[dict]:
    summaries = []
    method_order = [method[0] for method in METHODS]
    for method in method_order:
        selected = [
            row for row in rows
            if row["method"] == method and row["steady_state"] == "yes"
        ]
        if not selected:
            raise RuntimeError(f"no steady-state rows for {method}")
        summary = {
            "method": method,
            "method_label": selected[0]["method_label"],
            "cache_policy": selected[0]["cache_policy"],
            "steady_start_frame": steady_start_frame,
            "steady_end_frame": max(int(row["frame_number"]) for row in selected),
            "num_steady_steps": len(selected),
            "anchor_coverage_fraction": np.mean(
                [row["anchor_covered"] for row in selected]
            ),
        }
        for name in TEMPORAL_BINS:
            summary[f"{name}_coverage_fraction"] = np.mean(
                [row[f"{name}_covered"] for row in selected]
            )
        summary.update(
            all_four_coverage_fraction=np.mean(
                [row["all_four_temporal_bins_covered"] for row in selected]
            ),
            mean_covered_temporal_bins=np.mean(
                [row["covered_temporal_bins"] for row in selected]
            ),
            minimum_covered_temporal_bins=min(
                row["covered_temporal_bins"] for row in selected
            ),
            mean_oldest_nonanchor_age=np.mean(
                [row["oldest_nonanchor_age"] for row in selected]
            ),
            minimum_oldest_nonanchor_age=min(
                row["oldest_nonanchor_age"] for row in selected
            ),
        )
        summaries.append(summary)
    return summaries


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty CSV: {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {path}")


def gate_rows(summaries: list[dict]) -> list[dict]:
    hierarchical = next(
        row for row in summaries if row["method"] == "hierarchical_k8"
    )
    control = next(
        row for row in summaries if row["method"] == "nonhierarchical_dino8"
    )
    all_four_delta = float(hierarchical["all_four_coverage_fraction"]) - float(
        control["all_four_coverage_fraction"]
    )
    mean_bin_delta = float(hierarchical["mean_covered_temporal_bins"]) - float(
        control["mean_covered_temporal_bins"]
    )
    return [
        {
            "gate": "strict_all_step_guarantee",
            "threshold": "hierarchical all-four coverage = 1.0",
            "observed": hierarchical["all_four_coverage_fraction"],
            "passed": "yes" if float(hierarchical["all_four_coverage_fraction"]) == 1.0 else "no",
            "permitted_claim": (
                "guarantees all four temporal bins at every steady-state step"
            ),
        },
        {
            "gate": "comparative_all_four_coverage",
            "threshold": "hierarchical - non-hierarchical >= 0.20",
            "observed": all_four_delta,
            "passed": "yes" if all_four_delta >= 0.20 else "no",
            "permitted_claim": "more consistent all-bin temporal coverage",
        },
        {
            "gate": "comparative_mean_bin_coverage",
            "threshold": "hierarchical - non-hierarchical >= 0.25 bins",
            "observed": mean_bin_delta,
            "passed": "yes" if mean_bin_delta >= 0.25 else "no",
            "permitted_claim": "more consistent multi-scale temporal coverage",
        },
    ]


def plot_coverage(
    results: list[dict], summaries: list[dict], steady_start_frame: int, output_dir: Path
) -> None:
    figure, axes = plt.subplots(
        3, 1, figsize=(11.5, 9.2), dpi=180,
        gridspec_kw={"height_ratios": (1.0, 1.0, 0.9)},
    )
    bin_spans = (
        (0, 3, "#D9F0D3", "recent"),
        (4, 15, "#C6DBEF", "near"),
        (16, 47, "#FDD0A2", "middle"),
        (48, 109, "#DADAEB", "long"),
    )
    for axis, result in zip(axes[:2], results):
        for low, high, colour, _name in bin_spans:
            axis.axhspan(low, high, color=colour, alpha=0.42, linewidth=0)
        for trace in result["memory_trace"]:
            current = int(trace["frame_index"])
            retained = [int(value) for value in trace["retained_frame_ids"] if int(value) != 0]
            ages = [current - frame_id for frame_id in retained]
            axis.scatter([current + 1] * len(ages), ages, s=7, color="#1F4E79", alpha=0.72)
        axis.axvline(steady_start_frame, color="black", linestyle="--", linewidth=0.8)
        axis.set_xlim(1, results[0]["num_frames"])
        axis.set_ylim(-1, results[0]["num_frames"])
        axis.set_ylabel("Retained age")
        axis.set_title(result["method_label"], loc="left", fontsize=10)
        axis.grid(alpha=0.15)
    axes[1].set_xlabel("Current sampled view (one-based)")

    labels = ["Recent", "Near", "Middle", "Long", "All four"]
    keys = [f"{name}_coverage_fraction" for name in TEMPORAL_BINS] + [
        "all_four_coverage_fraction"
    ]
    x = np.arange(len(labels))
    width = 0.36
    colours = ("#0072B2", "#D55E00")
    for index, (summary, colour) in enumerate(zip(summaries, colours)):
        values = [float(summary[key]) for key in keys]
        offset = (index - 0.5) * width
        axes[2].bar(
            x + offset, values, width=width, label=summary["method_label"], color=colour
        )
    axes[2].set_xticks(x, labels)
    axes[2].set_ylim(0, 1.08)
    axes[2].set_ylabel("Coverage fraction")
    axes[2].set_title(
        f"Temporal-bin occupancy over views {steady_start_frame}--{results[0]['num_frames']}",
        loc="left", fontsize=10,
    )
    axes[2].legend(frameon=False, ncol=2)
    axes[2].grid(axis="y", alpha=0.2)
    figure.suptitle("Matched K=8 temporal-coverage diagnostic")
    figure.tight_layout()
    for suffix in ("pdf", "png"):
        path = output_dir / f"figure_k8_temporal_coverage.{suffix}"
        figure.savefig(path, bbox_inches="tight")
        print(f"Wrote {path}")
    plt.close(figure)


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if args.steady_start_frame < 50:
        raise ValueError("--steady-start-frame must be at least 50 for a non-anchor long bin")
    if args.steady_start_frame > args.max_frames:
        raise ValueError("--steady-start-frame must not exceed --max-frames")

    root = args.repo_root.resolve()
    weights = args.weights if args.weights.is_absolute() else root / args.weights
    images_dir = args.images_dir if args.images_dir.is_absolute() else root / args.images_dir
    output_dir = args.output_dir if args.output_dir.is_absolute() else root / args.output_dir
    if not weights.is_file():
        raise FileNotFoundError(weights)
    paths = selected_images(
        images_dir, args.image_glob, args.sampling_stride, args.max_frames
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    sys.path.insert(0, str(root / "src"))
    from streamvggt.models.streamvggt import StreamVGGT

    model = StreamVGGT().eval().to("cuda")
    checkpoint = torch.load(weights, map_location="cuda")
    model.load_state_dict(checkpoint, strict=True)
    del checkpoint

    results = []
    for slug, label, window, policy in METHODS:
        print(f"[K8 coverage] {label}: {len(paths)} views")
        result = run_method(model, paths, slug, label, window, policy, root)
        write_payload(result, paths, output_dir)
        results.append(result)

    rows = coverage_rows(results, args.steady_start_frame)
    summaries = summary_rows(rows, args.steady_start_frame)
    gates = gate_rows(summaries)
    write_csv(output_dir / "k8_temporal_coverage_steps.csv", rows)
    write_csv(output_dir / "k8_temporal_coverage_summary.csv", summaries)
    write_csv(output_dir / "k8_temporal_coverage_gate.csv", gates)
    plot_coverage(results, summaries, args.steady_start_frame, output_dir)
    metadata = {
        "dataset": "7-Scenes chess/seq-01",
        "image_glob": args.image_glob,
        "sampling_stride": args.sampling_stride,
        "num_frames": len(paths),
        "steady_start_frame_one_based": args.steady_start_frame,
        "temporal_bins_by_age": {
            "recent": "0--3",
            "near": "4--15",
            "middle": "16--47",
            "long": ">=48 (frame 0 is counted only as the separate anchor)",
        },
        "pre_registered_gates": {
            "strict_guarantee": "hierarchical all-four coverage = 1.0",
            "comparative_all_four": (
                "hierarchical minus non-hierarchical all-four coverage >= 0.20"
            ),
            "comparative_mean_bins": (
                "hierarchical minus non-hierarchical mean occupied bins >= 0.25"
            ),
        },
        "claim_limit": (
            "This diagnostic verifies temporal coverage, not downstream accuracy causality."
        ),
    }
    (output_dir / "k8_temporal_coverage_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
