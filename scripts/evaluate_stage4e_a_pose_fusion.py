#!/usr/bin/env python3
"""Stage 4E-A offline K4/K8 pose-composability screen."""

import argparse
import csv
import json
import os
import os.path as osp
import sys
import traceback

import numpy as np


REPO_ROOT = osp.abspath(osp.join(osp.dirname(__file__), ".."))
SRC_ROOT = osp.join(REPO_ROOT, "src")
if SRC_ROOT not in sys.path:
    sys.path.append(SRC_ROOT)

from eval.pose_evaluation.trajectory_metrics import evaluate_trajectory


K4 = "stage3_2_k4"
K8 = "temporal_binned_dino_k8"
VARIANTS = (
    "direct_k4_geometry_k8_pose",
    "component_k4_translation_k8_rotation",
)
FIELDS = (
    "sequence",
    "num_frames",
    "variant",
    "status",
    "error",
    "k4_ate",
    "k4_rpe_trans",
    "k4_rpe_rot_deg",
    "k8_ate",
    "k8_rpe_trans",
    "k8_rpe_rot_deg",
    "output_ate",
    "output_rpe_trans",
    "output_rpe_rot_deg",
    "output_align_scale",
    "ate_ratio_to_k4",
    "rpe_trans_ratio_to_k8",
    "rpe_rot_ratio_to_k8",
    "baseline_max_abs_diff",
    "k4_inference_sec",
    "k8_inference_sec",
    "dual_inference_sec_proxy",
    "dual_fps_proxy",
    "full_100_mean_fps",
    "k4_peak_allocated_mb",
    "k8_peak_allocated_mb",
    "sequential_peak_allocated_mb_proxy",
    "k4_aggregator_kv_mib",
    "k8_aggregator_kv_mib",
    "combined_aggregator_kv_mib",
    "projected_online_peak_allocated_mb",
    "resource_evidence",
    "output_path",
)
SUMMARY_FIELDS = (
    "candidate",
    "successful_units",
    "mean_k4_ate",
    "mean_k8_ate",
    "mean_output_ate",
    "max_ate_ratio_to_k4",
    "mean_k4_rpe_trans",
    "mean_k8_rpe_trans",
    "mean_output_rpe_trans",
    "max_rpe_trans_ratio_to_k8",
    "mean_k4_rpe_rot_deg",
    "mean_k8_rpe_rot_deg",
    "mean_output_rpe_rot_deg",
    "max_rpe_rot_ratio_to_k8",
    "mean_dual_fps_proxy",
    "max_sequential_peak_allocated_mb_proxy",
    "max_projected_online_peak_allocated_mb",
    "resource_evidence",
)


def read_csv(path):
    with open(path, newline="") as handle:
        return list(csv.DictReader(handle))


def number(row, key):
    value = row.get(key)
    return None if value in (None, "") else float(value)


def trajectory_path(
    results_root, method, sequence, frames, recorded_result_dir=None
):
    filename = sequence.replace("/", "_") + ".npz"
    candidates = []
    if recorded_result_dir:
        candidates.append(
            osp.join(recorded_result_dir, "trajectories", filename)
        )
    candidates.append(
        osp.join(
            results_root,
            method,
            sequence,
            str(frames),
            "trajectories",
            filename,
        )
    )
    return next((path for path in candidates if osp.isfile(path)), candidates[-1])


def load_trajectory(path):
    if not osp.isfile(path):
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=False) as payload:
        return {
            "gt_c2w": payload["gt_c2w"],
            "pred_c2w": payload["pred_c2w"],
            "image_paths": payload["image_paths"],
        }


def metric_subset(metrics):
    return {
        "ate": float(metrics["ate"]),
        "rpe_trans": float(metrics["rpe_trans"]),
        "rpe_rot_deg": float(metrics["rpe_rot_deg"]),
        "align_scale": float(metrics["align_scale"]),
    }


def evaluate(ground_truth, prediction):
    metrics = evaluate_trajectory(ground_truth, prediction)
    return metric_subset(metrics)


def mean(rows, key):
    values = [number(row, key) for row in rows]
    if not values or any(value is None for value in values):
        return None
    return sum(values) / len(values)


def project_rotation(rotation):
    u, _, vh = np.linalg.svd(rotation)
    nearest = u @ vh
    if np.linalg.det(nearest) < 0:
        u[:, -1] *= -1
        nearest = u @ vh
    return nearest


def component_fusion(k4_c2w, k8_c2w):
    """Use K4 camera centers and K8 orientations in the K4 initial gauge."""
    if k4_c2w.shape != k8_c2w.shape:
        raise ValueError(
            f"K4/K8 trajectory shape mismatch: {k4_c2w.shape}/{k8_c2w.shape}"
        )
    gauge_rotation = project_rotation(
        k4_c2w[0, :3, :3] @ k8_c2w[0, :3, :3].T
    )
    fused = np.array(k4_c2w, dtype=np.float64, copy=True)
    fused[:, :3, :3] = np.einsum(
        "ij,njk->nik", gauge_rotation, k8_c2w[:, :3, :3]
    )
    fused[:, 3] = np.array([0.0, 0.0, 0.0, 1.0])
    return fused


def validate_pair(k4, k8, expected_frames):
    if len(k4["gt_c2w"]) != expected_frames:
        raise ValueError(
            f"trajectory has {len(k4['gt_c2w'])} frames, expected {expected_frames}"
        )
    if not np.array_equal(k4["image_paths"], k8["image_paths"]):
        raise ValueError("K4/K8 image paths differ")
    if not np.allclose(
        k4["gt_c2w"], k8["gt_c2w"], rtol=0.0, atol=1e-12
    ):
        raise ValueError("K4/K8 ground-truth trajectories differ")


def baseline_difference(computed, row):
    return max(
        abs(computed[key] - number(row, key))
        for key in ("ate", "rpe_trans", "rpe_rot_deg")
    )


def main():
    parser = argparse.ArgumentParser("Stage 4E-A pose fusion")
    parser.add_argument("--stage4c-results", default="stage4c_results.csv")
    parser.add_argument(
        "--stage4c-results-root",
        default="eval_results/stage4c_tum_long",
    )
    parser.add_argument(
        "--output-root", default="eval_results/stage4e_a_pose_fusion"
    )
    parser.add_argument(
        "--output", default="stage4e_a_sequence_results.csv"
    )
    parser.add_argument(
        "--summary-output", default="stage4e_a_results.csv"
    )
    parser.add_argument(
        "--lengths", type=int, nargs="+", default=(250, 500, 1000)
    )
    parser.add_argument("--sequences", nargs="+")
    args = parser.parse_args()

    source_rows = read_csv(args.stage4c_results)
    successful = [
        row for row in source_rows if row.get("status") == "ok"
    ]
    full_100_fps = [
        number(row, "fps_inference")
        for row in successful
        if row["method"] == "full_cache"
        and int(row["num_frames"]) == 100
    ]
    if not full_100_fps:
        raise ValueError("Stage 4C has no successful full-cache 100-frame FPS")
    full_100_mean_fps = sum(full_100_fps) / len(full_100_fps)
    sequences = args.sequences or sorted(
        {
            row["sequence"]
            for row in successful
            if row["method"] in (K4, K8)
        }
    )
    by_key = {
        (row["method"], row["sequence"], int(row["num_frames"])): row
        for row in source_rows
    }

    output_rows = []
    for sequence in sequences:
        for frames in args.lengths:
            k4_row = by_key.get((K4, sequence, frames))
            k8_row = by_key.get((K8, sequence, frames))
            common = {
                "sequence": sequence,
                "num_frames": frames,
                "full_100_mean_fps": full_100_mean_fps,
            }
            try:
                if (
                    k4_row is None
                    or k8_row is None
                    or k4_row.get("status") != "ok"
                    or k8_row.get("status") != "ok"
                ):
                    raise ValueError(
                        f"missing successful Stage 4C K4/K8 rows for "
                        f"{sequence}/{frames}"
                    )
                k4 = load_trajectory(
                    trajectory_path(
                        args.stage4c_results_root,
                        K4,
                        sequence,
                        frames,
                        k4_row.get("result_dir"),
                    )
                )
                k8 = load_trajectory(
                    trajectory_path(
                        args.stage4c_results_root,
                        K8,
                        sequence,
                        frames,
                        k8_row.get("result_dir"),
                    )
                )
                validate_pair(k4, k8, frames)
                k4_metrics = evaluate(k4["gt_c2w"], k4["pred_c2w"])
                k8_metrics = evaluate(k8["gt_c2w"], k8["pred_c2w"])
                baseline_max_abs_diff = max(
                    baseline_difference(k4_metrics, k4_row),
                    baseline_difference(k8_metrics, k8_row),
                )

                fused = component_fusion(
                    k4["pred_c2w"], k8["pred_c2w"]
                )
                predictions = {
                    "direct_k4_geometry_k8_pose": k8["pred_c2w"],
                    "component_k4_translation_k8_rotation": fused,
                }
                output_dir = osp.join(
                    args.output_root, sequence, str(frames)
                )
                os.makedirs(output_dir, exist_ok=True)
                output_path = osp.join(output_dir, "stage4e_a_poses.npz")
                np.savez_compressed(
                    output_path,
                    gt_c2w=k4["gt_c2w"],
                    k4_c2w=k4["pred_c2w"],
                    k8_c2w=k8["pred_c2w"],
                    direct_c2w=k8["pred_c2w"],
                    component_fused_c2w=fused,
                    image_paths=k4["image_paths"],
                )

                k4_sec = number(k4_row, "inference_sec")
                k8_sec = number(k8_row, "inference_sec")
                dual_sec = k4_sec + k8_sec
                k4_peak = number(k4_row, "peak_allocated_mb")
                k8_peak = number(k8_row, "peak_allocated_mb")
                k4_kv = number(k4_row, "max_aggregator_kv_mib")
                k8_kv = number(k8_row, "max_aggregator_kv_mib")
                sequential_peak = max(k4_peak, k8_peak)
                combined_kv = k4_kv + k8_kv
                # K8 peak already includes its own KV. Adding the measured K4
                # KV is only a first-order projection, not an online
                # measurement or a substitute for Stage 4E-B profiling.
                projected_online_peak = k8_peak + k4_kv

                for variant, prediction in predictions.items():
                    metrics = evaluate(k4["gt_c2w"], prediction)
                    output_rows.append(
                        {
                            **common,
                            "variant": variant,
                            "status": "ok",
                            "error": "",
                            "k4_ate": k4_metrics["ate"],
                            "k4_rpe_trans": k4_metrics["rpe_trans"],
                            "k4_rpe_rot_deg": k4_metrics[
                                "rpe_rot_deg"
                            ],
                            "k8_ate": k8_metrics["ate"],
                            "k8_rpe_trans": k8_metrics["rpe_trans"],
                            "k8_rpe_rot_deg": k8_metrics[
                                "rpe_rot_deg"
                            ],
                            "output_ate": metrics["ate"],
                            "output_rpe_trans": metrics["rpe_trans"],
                            "output_rpe_rot_deg": metrics[
                                "rpe_rot_deg"
                            ],
                            "output_align_scale": metrics[
                                "align_scale"
                            ],
                            "ate_ratio_to_k4": metrics["ate"]
                            / k4_metrics["ate"],
                            "rpe_trans_ratio_to_k8": metrics[
                                "rpe_trans"
                            ]
                            / k8_metrics["rpe_trans"],
                            "rpe_rot_ratio_to_k8": metrics[
                                "rpe_rot_deg"
                            ]
                            / k8_metrics["rpe_rot_deg"],
                            "baseline_max_abs_diff": baseline_max_abs_diff,
                            "k4_inference_sec": k4_sec,
                            "k8_inference_sec": k8_sec,
                            "dual_inference_sec_proxy": dual_sec,
                            "dual_fps_proxy": frames / dual_sec,
                            "k4_peak_allocated_mb": k4_peak,
                            "k8_peak_allocated_mb": k8_peak,
                            "sequential_peak_allocated_mb_proxy": sequential_peak,
                            "k4_aggregator_kv_mib": k4_kv,
                            "k8_aggregator_kv_mib": k8_kv,
                            "combined_aggregator_kv_mib": combined_kv,
                            "projected_online_peak_allocated_mb": projected_online_peak,
                            "resource_evidence": (
                                "offline_proxy_from_two_stage4c_runs"
                            ),
                            "output_path": output_path,
                        }
                    )
            except Exception as error:
                traceback.print_exc()
                for variant in VARIANTS:
                    output_rows.append(
                        {
                            **common,
                            "variant": variant,
                            "status": "failed",
                            "error": f"{type(error).__name__}: {error}",
                        }
                    )

    expected = len(sequences) * len(args.lengths) * len(VARIANTS)
    if len(output_rows) != expected:
        raise RuntimeError(
            f"generated {len(output_rows)} rows, expected {expected}"
        )
    with open(args.output, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(output_rows)
    summary_rows = []
    for variant in VARIANTS:
        rows = [
            row
            for row in output_rows
            if row["variant"] == variant and row["status"] == "ok"
        ]
        summary_rows.append(
            {
                "candidate": variant,
                "successful_units": len(rows),
                "mean_k4_ate": mean(rows, "k4_ate"),
                "mean_k8_ate": mean(rows, "k8_ate"),
                "mean_output_ate": mean(rows, "output_ate"),
                "max_ate_ratio_to_k4": max(
                    (number(row, "ate_ratio_to_k4") for row in rows),
                    default=None,
                ),
                "mean_k4_rpe_trans": mean(rows, "k4_rpe_trans"),
                "mean_k8_rpe_trans": mean(rows, "k8_rpe_trans"),
                "mean_output_rpe_trans": mean(
                    rows, "output_rpe_trans"
                ),
                "max_rpe_trans_ratio_to_k8": max(
                    (
                        number(row, "rpe_trans_ratio_to_k8")
                        for row in rows
                    ),
                    default=None,
                ),
                "mean_k4_rpe_rot_deg": mean(rows, "k4_rpe_rot_deg"),
                "mean_k8_rpe_rot_deg": mean(rows, "k8_rpe_rot_deg"),
                "mean_output_rpe_rot_deg": mean(
                    rows, "output_rpe_rot_deg"
                ),
                "max_rpe_rot_ratio_to_k8": max(
                    (
                        number(row, "rpe_rot_ratio_to_k8")
                        for row in rows
                    ),
                    default=None,
                ),
                "mean_dual_fps_proxy": mean(rows, "dual_fps_proxy"),
                "max_sequential_peak_allocated_mb_proxy": max(
                    (
                        number(
                            row,
                            "sequential_peak_allocated_mb_proxy",
                        )
                        for row in rows
                    ),
                    default=None,
                ),
                "max_projected_online_peak_allocated_mb": max(
                    (
                        number(
                            row,
                            "projected_online_peak_allocated_mb",
                        )
                        for row in rows
                    ),
                    default=None,
                ),
                "resource_evidence": (
                    "offline_proxy_from_two_stage4c_runs"
                ),
            }
        )
    with open(args.summary_output, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        writer.writerows(summary_rows)
    metadata = {
        "sequences": sequences,
        "lengths": args.lengths,
        "variants": VARIANTS,
        "fusion_uses_ground_truth": False,
        "resource_values_are_online_measurements": False,
    }
    with open(osp.splitext(args.output)[0] + "_metadata.json", "w") as handle:
        json.dump(metadata, handle, indent=2)
    print(f"Wrote {len(output_rows)} Stage 4E-A rows to {args.output}")
    print(f"Wrote Stage 4E-A summary to {args.summary_output}")


if __name__ == "__main__":
    main()
