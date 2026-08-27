"""CPU-only checks for the worksheet-driven supplementary package."""

from __future__ import annotations

import csv
import importlib.util
import sys
import unittest
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "supplementary material"
sys.path.insert(0, str(ROOT / "src"))
SPEC = importlib.util.spec_from_file_location(
    "streamvggt_model", ROOT / "src/streamvggt/models/streamvggt.py"
)
MODEL_MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODEL_MODULE)


def rows(path: Path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


class RequestedSupplementaryMaterialTest(unittest.TestCase):
    def test_manifest_covers_ten_p0_and_four_p1_requirements(self):
        manifest = rows(PACKAGE / "MATERIAL_STATUS.csv")
        self.assertEqual(10, sum(row["priority"] == "P0" for row in manifest))
        self.assertEqual(4, sum(row["priority"] == "P1" for row in manifest))

    def test_protocol_and_paired_tables_have_frozen_coverage(self):
        protocol = rows(PACKAGE / "tables/p0_01_dataset_and_protocol.csv")
        self.assertEqual(129, len(protocol))
        held_out = {
            row["dataset"]
            for row in protocol
            if row["policy_frozen_before_evaluation"] == "yes"
        }
        self.assertEqual({"kitti", "tum_rgbd_raw"}, held_out)
        paired = rows(PACKAGE / "tables/p0_03_paired_absrel_values.csv")
        self.assertEqual(41 * 5, len(paired))

    def test_memory_table_records_output_bytes_slopes_and_hash_parity(self):
        memory = rows(PACKAGE / "tables/p0_08_memory_factorial_complete.csv")
        self.assertEqual(4, len(memory))
        self.assertTrue(all(row["pose_hash_equal_with_lifecycle_pair"] == "true" for row in memory))
        self.assertTrue(all(row["depth_hash_equal_with_lifecycle_pair"] == "true" for row in memory))
        accumulated = [row for row in memory if row["output_mode"] == "retained"]
        self.assertTrue(all(int(row["final_output_tensor_bytes"]) > 0 for row in accumulated))

    def test_selector_dot_product_accounting_matches_frozen_layouts(self):
        features = torch.nn.functional.normalize(torch.randn(9, 1024), dim=-1)
        ids = torch.arange(9)
        self.assertEqual(
            3,
            MODEL_MODULE._selection_dot_product_count(features[:5], ids[:5], 4, "anchor_recent_dino_diverse_k4"),
        )
        self.assertEqual(
            9,
            MODEL_MODULE._selection_dot_product_count(features[:7], ids[:7], 6, "anchor_recent_dino_diverse_k6"),
        )
        k8_ids = torch.tensor([0, 10, 30, 50, 80, 97, 98, 99, 100])
        self.assertEqual(
            15,
            MODEL_MODULE._selection_dot_product_count(features, k8_ids, 8, "anchor_recent_dino_diverse_k8"),
        )


if __name__ == "__main__":
    unittest.main()
