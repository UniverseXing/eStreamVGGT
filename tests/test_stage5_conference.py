"""CPU-only regression tests for the conference Stage 5 summarizers."""

from __future__ import annotations

import csv
import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SUMMARY_5B = REPO_ROOT / "scripts" / "summarize_stage5b_memory.py"
PLOT_5B = REPO_ROOT / "scripts" / "plot_stage5b_memory.py"
SPEC = importlib.util.spec_from_file_location(
    "summarize_stage5a", REPO_ROOT / "scripts" / "summarize_stage5a.py"
)
SUMMARY_5A = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SUMMARY_5A)


class Stage5ConferenceTest(unittest.TestCase):
    def test_stage5a_paired_advantage_uses_control_minus_k4(self):
        rows = []
        for sequence, proposed, recent, anchor_recent in (
            ("a", 0.10, 0.13, 0.12),
            ("b", 0.20, 0.22, 0.24),
        ):
            for method, value in (
                ("proposed_k4", proposed),
                ("recent4", recent),
                ("anchor_recent4", anchor_recent),
            ):
                rows.append(
                    {"dataset": "bonn", "sequence": sequence, "method": method, "abs_rel": value}
                )
        outputs = SUMMARY_5A.paired_statistics(rows, samples=1000)
        anchor = next(row for row in outputs if row["control"] == "anchor_recent4")
        self.assertAlmostEqual(anchor["mean_advantage_proposed"], 0.03)
        self.assertEqual(anchor["wins_proposed"], 2)

    def test_stage5b_factorial_validates_hashes_and_writes_contributions(self):
        configs = {
            "full_accumulated": ("stream_accumulate", None, "full_cache", "retained", 20000.0),
            "full_release": ("stream_release", None, "full_cache", "sink", 18000.0),
            "k4_accumulated": ("stream_accumulate", 4, "anchor_recent_dino_diverse_k4", "retained", 10000.0),
            "k4_release": ("stream_release", 4, "anchor_recent_dino_diverse_k4", "sink", 9000.0),
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for cell, (mode, window, policy, output_mode, peak) in configs.items():
                result_dir = root / "eval_results/stage5b_memory" / cell
                result_dir.mkdir(parents=True)
                cache = "full" if cell.startswith("full") else "k4"
                payload = {
                    "method": cell, "mode": mode, "status": "ok", "dataset": "bonn",
                    "sequence": "person_tracking2", "num_frames": 110,
                    "processed_frames": 110, "cache_window_size": window,
                    "cache_policy": policy, "input_mode": "streaming",
                    "output_mode": output_mode, "peak_allocated_mb": peak,
                    "peak_reserved_mb": peak + 1000, "camera_pose_sha256": f"pose-{cache}",
                    "depth_sha256": f"depth-{cache}",
                    "gpu_name": "NVIDIA RTX 6000 Ada Generation",
                    "torch_version": "2.3.1", "cuda_version": "12.1",
                }
                (result_dir / "stage5b_metrics.json").write_text(json.dumps(payload))
                trace = [
                    {
                        "frame_index": index,
                        "retained_frame_ids": [index],
                        "cuda_allocated_mib": peak - 100 + index,
                        "cuda_reserved_mib": peak + 900,
                    }
                    for index in range(110)
                ]
                (result_dir / "memory_trace.json").write_text(json.dumps(trace))
            result = subprocess.run(
                [sys.executable, str(SUMMARY_5B), "--repo-root", str(root)],
                capture_output=True, text=True, check=False,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            with (root / "stage5b_memory_contributions.csv").open(newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 4)
            kv_release = next(
                row for row in rows if row["effect"] == "kv_pruning_with_streaming_release"
            )
            self.assertAlmostEqual(float(kv_release["peak_allocated_saved_mib"]), 9000.0)
            plot_result = subprocess.run(
                [sys.executable, str(PLOT_5B), "--repo-root", str(root)],
                capture_output=True, text=True, check=False,
            )
            self.assertEqual(plot_result.returncode, 0, plot_result.stderr)
            self.assertTrue(
                (root / "paper_assets/figures/fig_stage5b_memory_decomposition.svg").is_file()
            )
            self.assertTrue(
                (root / "paper_assets/figures/fig_stage5b_memory_decomposition.png").is_file()
            )


if __name__ == "__main__":
    unittest.main()
