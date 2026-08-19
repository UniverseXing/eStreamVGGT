"""CPU-only regression tests for the Stage 5A same-budget controls."""

import sys
import unittest
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from streamvggt.models.streamvggt import _cache_keep_frame_indices  # noqa: E402


class Stage5ACachePolicyTest(unittest.TestCase):
    device = torch.device("cpu")

    def test_anchor_uniform_matches_k4_layout_without_dino(self):
        keep = _cache_keep_frame_indices(
            5, 4, "anchor_uniform_k4", self.device, frame_ids=torch.arange(5)
        )
        self.assertEqual(keep.tolist(), [0, 1, 3, 4])

    def test_dino_only_can_evict_frame_zero(self):
        # Frame 0 duplicates the current descriptor; frames 1--3 are diverse.
        features = torch.tensor(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
                [1.0, 0.0, 0.0, 0.0],
            ]
        )
        keep = _cache_keep_frame_indices(
            5,
            4,
            "dino_diverse_no_anchor_k4",
            self.device,
            frame_features=features,
            frame_ids=torch.arange(5),
        )
        self.assertEqual(keep.tolist(), [1, 2, 3, 4])

    def test_k6_no_recent_has_anchor_four_dino_history_and_current(self):
        features = torch.eye(7)
        keep = _cache_keep_frame_indices(
            7,
            6,
            "anchor_dino_diverse_no_recent_k6",
            self.device,
            frame_features=features,
            frame_ids=torch.arange(7),
        )
        self.assertEqual(keep.numel(), 6)
        self.assertEqual(keep[0].item(), 0)
        self.assertEqual(keep[-1].item(), 6)

    @staticmethod
    def _run_random_stream(seed):
        retained = torch.empty(0, dtype=torch.long)
        for frame_id in range(40):
            retained = torch.cat([retained, torch.tensor([frame_id])])
            keep = _cache_keep_frame_indices(
                retained.numel(),
                4,
                "random_reservoir_k4",
                torch.device("cpu"),
                frame_ids=retained,
                random_seed=seed,
            )
            if keep is not None:
                retained = retained.index_select(0, keep)
            assert retained.numel() <= 4
            assert retained[-1].item() == frame_id
        return retained

    def test_random_reservoir_is_reproducible_and_seeded(self):
        seed0_a = self._run_random_stream(0)
        seed0_b = self._run_random_stream(0)
        seed1 = self._run_random_stream(1)
        self.assertTrue(torch.equal(seed0_a, seed0_b))
        self.assertFalse(torch.equal(seed0_a[:-1], seed1[:-1]))
        self.assertEqual(seed0_a.numel(), 4)
        self.assertEqual(seed0_a[-1].item(), 39)

    def test_stage5a_policies_reject_wrong_window(self):
        configs = {
            "anchor_uniform_k4": 4,
            "random_reservoir_k4": 4,
            "dino_diverse_no_anchor_k4": 4,
            "anchor_dino_diverse_no_recent_k6": 6,
        }
        for policy, window in configs.items():
            with self.subTest(policy=policy):
                with self.assertRaisesRegex(ValueError, "requires cache_window_size"):
                    _cache_keep_frame_indices(
                        window + 1,
                        window + 1,
                        policy,
                        self.device,
                        frame_features=torch.eye(window + 1),
                        frame_ids=torch.arange(window + 1),
                    )


if __name__ == "__main__":
    unittest.main()
