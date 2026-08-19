"""CPU-only checks for Stage 5A paired statistics."""

import importlib.util
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "summarize_stage5a", REPO_ROOT / "scripts" / "summarize_stage5a.py"
)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class Stage5ASummaryTest(unittest.TestCase):
    def test_random_seeds_are_averaged_before_pairing(self):
        rows = []
        proposed = {"a": 0.10, "b": 0.20}
        seed_values = {
            "random4_seed0": {"a": 0.12, "b": 0.21},
            "random4_seed1": {"a": 0.14, "b": 0.23},
            "random4_seed2": {"a": 0.16, "b": 0.25},
        }
        for sequence, value in proposed.items():
            rows.append(
                {"task": "video_depth", "dataset": "kitti", "sequence": sequence,
                 "method": "proposed_k4", "abs_rel": value}
            )
        for method, values in seed_values.items():
            for sequence, value in values.items():
                rows.append(
                    {"task": "video_depth", "dataset": "kitti", "sequence": sequence,
                     "method": method, "abs_rel": value}
                )
        comparisons = MODULE.paired_rows(rows, samples=1000)
        result = next(item for item in comparisons if item["control"] == "random4_mean")
        self.assertAlmostEqual(result["mean_control"], 0.185)
        self.assertAlmostEqual(result["mean_proposed"], 0.15)
        self.assertAlmostEqual(result["mean_advantage_proposed"], 0.035)
        self.assertEqual(result["wins_proposed"], 2)
        self.assertEqual(result["losses_proposed"], 0)


if __name__ == "__main__":
    unittest.main()
