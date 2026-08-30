#!/usr/bin/env python3
"""Summarize the matched 1000-frame TUM K8 pose comparison."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/estreamvggt-k8-pose-matplotlib")

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
METHODS = {
    "anchor_recent_dino_diverse_k8": (
        "Hierarchical K8", 8, "anchor_recent_dino_diverse_k8"
    ),
    "nonhierarchical_dino8": (
        "Non-hierarchical DINO-8", 8, "anchor_recent_dino_diverse"
    ),
}
SEQUENCES = (
    "rgbd_dataset_freiburg1_room",
    "rgbd_dataset_freiburg2_desk",
    "rgbd_dataset_freiburg3_long_office_household",
)
METRICS = ("ate", "rpe_trans", "rpe_rot_deg")
RESULT_FIELDS = (
    "method", "method_label", "sequence", "num_frames", "status", "pose_status",
    "ate", "rpe_trans", "rpe_rot_deg", "fps_inference", "peak_allocated_mb",
    "cache_window_size", "cache_policy", "gpu_name", "torch_version",
    "cuda_version", "python_version", "slurm_job_id", "hostname", "run_id",
    "result_dir",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-root", type=Path,
        default=Path("eval_results/supplementary_k8_pose"),
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--expected-run-id", required=True)
    return parser.parse_args()


def write_csv(path: Path, rows: list[dict], fields=None) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields or list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {path}")


def load_results(root: Path, expected_run_id: str) -> list[dict]:
    rows = []
    for method, (label, window, policy) in METHODS.items():
        for sequence in SEQUENCES:
            result_dir = root / method / sequence / "1000"
            path = result_dir / "stage4c_metrics.json"
            if not path.is_file():
                raise FileNotFoundError(f"missing planned result: {path}")
            payload = json.loads(path.read_text(encoding="utf-8"))
            if payload.get("run_id") != expected_run_id:
                raise RuntimeError(
                    f"stale result for {method}/{sequence}: expected run_id "
                    f"{expected_run_id!r}, found {payload.get('run_id')!r}"
                )
            if payload.get("method") != method or payload.get("sequence") != sequence:
                raise RuntimeError(f"result identity mismatch: {path}")
            if int(payload.get("num_frames", 0)) != 1000:
                raise RuntimeError(f"non-1000-frame result: {path}")
            if int(payload.get("processed_frames", 0)) != 1000:
                raise RuntimeError(f"incomplete result: {path}")
            if payload.get("status") != "ok" or payload.get("pose_status") != "ok":
                raise RuntimeError(f"failed inference or pose evaluation: {path}")
            if int(payload.get("cache_window_size", 0)) != window:
                raise RuntimeError(f"cache-window mismatch: {path}")
            if payload.get("cache_policy") != policy:
                raise RuntimeError(f"cache-policy mismatch: {path}")
            if any(payload.get(metric) is None for metric in METRICS):
                raise RuntimeError(f"missing pose metric: {path}")
            row = {field: payload.get(field, "") for field in RESULT_FIELDS}
            row.update(method_label=label, result_dir=str(result_dir))
            rows.append(row)

    for field in ("gpu_name", "torch_version", "cuda_version", "python_version"):
        values = {str(row[field]) for row in rows}
        if len(values) != 1 or not next(iter(values)):
            raise RuntimeError(f"inconsistent {field}: {values}")
    rows.sort(key=lambda row: (SEQUENCES.index(row["sequence"]), list(METHODS).index(row["method"])))
    return rows


def comparisons(rows: list[dict]) -> tuple[list[dict], list[dict]]:
    by_key = {(row["method"], row["sequence"]): row for row in rows}
    proposed_id = "anchor_recent_dino_diverse_k8"
    control_id = "nonhierarchical_dino8"
    paired = []
    summary = []
    for metric in METRICS:
        proposed_values = []
        control_values = []
        wins = 0
        for sequence in SEQUENCES:
            proposed = float(by_key[(proposed_id, sequence)][metric])
            control = float(by_key[(control_id, sequence)][metric])
            improvement = 100.0 * (control - proposed) / max(abs(control), 1e-12)
            winner = (
                "Hierarchical K8" if proposed < control
                else "Non-hierarchical DINO-8" if control < proposed
                else "tie"
            )
            wins += int(proposed < control)
            proposed_values.append(proposed)
            control_values.append(control)
            paired.append(
                {
                    "sequence": sequence,
                    "num_frames": 1000,
                    "metric": metric,
                    "hierarchical_k8": proposed,
                    "nonhierarchical_dino8": control,
                    "hierarchical_minus_nonhierarchical": proposed - control,
                    "hierarchical_relative_improvement_percent": improvement,
                    "winner": winner,
                }
            )
        proposed_mean = float(np.mean(proposed_values))
        control_mean = float(np.mean(control_values))
        summary.append(
            {
                "metric": metric,
                "num_sequences": len(SEQUENCES),
                "hierarchical_macro_mean": proposed_mean,
                "nonhierarchical_macro_mean": control_mean,
                "hierarchical_relative_improvement_percent": (
                    100.0 * (control_mean - proposed_mean) / max(abs(control_mean), 1e-12)
                ),
                "hierarchical_wins": wins,
                "ties": sum(
                    proposed == control
                    for proposed, control in zip(proposed_values, control_values)
                ),
                "hierarchical_losses": len(SEQUENCES) - wins - sum(
                    proposed == control
                    for proposed, control in zip(proposed_values, control_values)
                ),
                "maximum_hierarchical_to_control_ratio": max(
                    proposed / max(control, 1e-12)
                    for proposed, control in zip(proposed_values, control_values)
                ),
            }
        )
    return paired, summary


def gates(summary: list[dict]) -> list[dict]:
    by_metric = {row["metric"]: row for row in summary}
    strict_checks = [
        float(by_metric[metric]["hierarchical_relative_improvement_percent"]) > 0
        and int(by_metric[metric]["hierarchical_wins"]) >= 2
        for metric in METRICS
    ]
    rotation = by_metric["rpe_rot_deg"]
    ate = by_metric["ate"]
    translation = by_metric["rpe_trans"]
    rotation_checks = (
        float(rotation["hierarchical_relative_improvement_percent"]) >= 10.0,
        int(rotation["hierarchical_wins"]) >= 2,
        float(ate["hierarchical_relative_improvement_percent"]) >= -20.0,
        float(translation["hierarchical_relative_improvement_percent"]) >= -20.0,
        float(ate["maximum_hierarchical_to_control_ratio"]) <= 2.0,
    )
    return [
        {
            "gate": "strict_overall_pose_superiority",
            "threshold": "all three macro means improve and each metric wins >=2/3 sequences",
            "observed": json.dumps({
                metric: {
                    "improvement_percent": by_metric[metric]["hierarchical_relative_improvement_percent"],
                    "wins": by_metric[metric]["hierarchical_wins"],
                }
                for metric in METRICS
            }),
            "passed": "yes" if all(strict_checks) else "no",
            "permitted_claim": "overall 1000-frame pose superiority",
        },
        {
            "gate": "rotation_pose_specialist",
            "threshold": (
                "rotation RPE improves >=10% and wins >=2/3; macro ATE/translation "
                "regress <=20%; maximum per-sequence ATE ratio <=2"
            ),
            "observed": json.dumps({
                "rotation_improvement_percent": rotation["hierarchical_relative_improvement_percent"],
                "rotation_wins": rotation["hierarchical_wins"],
                "ate_improvement_percent": ate["hierarchical_relative_improvement_percent"],
                "translation_improvement_percent": translation["hierarchical_relative_improvement_percent"],
                "maximum_ate_ratio": ate["maximum_hierarchical_to_control_ratio"],
            }),
            "passed": "yes" if all(rotation_checks) else "no",
            "permitted_claim": "1000-frame rotation-pose specialist",
        },
    ]


def plot_comparison(paired: list[dict], output_dir: Path) -> None:
    short = {
        "rgbd_dataset_freiburg1_room": "F1-Room",
        "rgbd_dataset_freiburg2_desk": "F2-Desk",
        "rgbd_dataset_freiburg3_long_office_household": "F3-Office",
    }
    titles = {"ate": "ATE", "rpe_trans": "Translation RPE", "rpe_rot_deg": "Rotation RPE (deg)"}
    figure, axes = plt.subplots(1, 3, figsize=(11.5, 3.7), dpi=180)
    x = np.arange(len(SEQUENCES))
    width = 0.36
    for axis, metric in zip(axes, METRICS):
        values = [row for row in paired if row["metric"] == metric]
        hierarchical = [float(row["hierarchical_k8"]) for row in values]
        control = [float(row["nonhierarchical_dino8"]) for row in values]
        axis.bar(x - width / 2, hierarchical, width, label="Hierarchical K8", color="#0072B2")
        axis.bar(x + width / 2, control, width, label="Non-hierarchical DINO-8", color="#D55E00")
        axis.set_xticks(x, [short[sequence] for sequence in SEQUENCES], rotation=18)
        axis.set_title(titles[metric])
        axis.grid(axis="y", alpha=0.2)
    axes[0].set_ylabel("Error (lower is better)")
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="upper center", ncol=2, frameon=False)
    figure.suptitle("Matched 1000-frame TUM pose comparison", y=1.04)
    figure.tight_layout()
    for suffix in ("pdf", "png"):
        path = output_dir / f"figure_k8_pose_comparison.{suffix}"
        figure.savefig(path, bbox_inches="tight")
        print(f"Wrote {path}")
    plt.close(figure)


def main() -> None:
    args = parse_args()
    root = args.results_root if args.results_root.is_absolute() else ROOT / args.results_root
    output = args.output_dir or root
    if not output.is_absolute():
        output = ROOT / output
    rows = load_results(root, args.expected_run_id)
    paired, summary = comparisons(rows)
    gate_rows = gates(summary)
    write_csv(output / "k8_pose_results.csv", rows, RESULT_FIELDS)
    write_csv(output / "k8_pose_comparison.csv", paired)
    write_csv(output / "k8_pose_summary.csv", summary)
    write_csv(output / "k8_pose_gate.csv", gate_rows)
    plot_comparison(paired, output)
    metadata = {
        "protocol": "frozen Stage 4C raw-TUM protocol",
        "num_frames": 1000,
        "sequences": list(SEQUENCES),
        "methods": {key: value[0] for key, value in METHODS.items()},
        "expected_run_id": args.expected_run_id,
        "claim_rule": (
            "Use only the claim permitted by a passed pre-registered gate; "
            "coverage and pose causality are not established by association alone."
        ),
    }
    (output / "k8_pose_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
