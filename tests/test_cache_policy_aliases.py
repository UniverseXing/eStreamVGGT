"""CPU-only checks for the public paper-facing cache-policy names."""

import sys
import unittest
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from streamvggt.models.streamvggt import (  # noqa: E402
    _cache_keep_frame_indices,
    _canonical_cache_policy,
)


PUBLIC_POLICIES = {
    "anchor_recent_dino_diverse_k4": (
        "anchor_recent_dino_diverse_2old_1recent",
        4,
    ),
    "anchor_recent_dino_diverse_k6": ("anchor_recent_dino_diverse", 6),
    "anchor_recent_dino_diverse_k8": ("temporal_binned_dino_k8", 8),
}


class CachePolicyAliasTest(unittest.TestCase):
    def test_public_names_resolve_to_frozen_implementations(self):
        for public_name, (implementation_name, _) in PUBLIC_POLICIES.items():
            with self.subTest(policy=public_name):
                self.assertEqual(
                    _canonical_cache_policy(public_name), implementation_name
                )

    def test_public_names_match_legacy_selection(self):
        num_frames = 65
        frame_features = torch.eye(num_frames)
        frame_ids = torch.arange(num_frames)
        device = torch.device("cpu")

        for public_name, (implementation_name, window_size) in PUBLIC_POLICIES.items():
            with self.subTest(policy=public_name):
                public_selection = _cache_keep_frame_indices(
                    num_frames,
                    window_size,
                    public_name,
                    device,
                    frame_features=frame_features,
                    frame_ids=frame_ids,
                )
                legacy_selection = _cache_keep_frame_indices(
                    num_frames,
                    window_size,
                    implementation_name,
                    device,
                    frame_features=frame_features,
                    frame_ids=frame_ids,
                )
                self.assertTrue(torch.equal(public_selection, legacy_selection))
                self.assertEqual(public_selection.numel(), window_size)
                self.assertEqual(public_selection[0].item(), 0)
                self.assertEqual(public_selection[-1].item(), num_frames - 1)

    def test_public_names_reject_the_wrong_window_size(self):
        frame_features = torch.eye(9)
        frame_ids = torch.arange(9)
        device = torch.device("cpu")

        for public_name, (_, window_size) in PUBLIC_POLICIES.items():
            with self.subTest(policy=public_name):
                with self.assertRaisesRegex(ValueError, "requires cache_window_size"):
                    _cache_keep_frame_indices(
                        9,
                        window_size + 1,
                        public_name,
                        device,
                        frame_features=frame_features,
                        frame_ids=frame_ids,
                    )


if __name__ == "__main__":
    unittest.main()
