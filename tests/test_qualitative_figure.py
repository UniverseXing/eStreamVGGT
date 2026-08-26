"""CPU-only checks for the single-scene qualitative panel exporter."""

from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts/reproduce/run_qualitative_figure.py"
SPEC = importlib.util.spec_from_file_location("qualitative_figure", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class QualitativeFigureTest(unittest.TestCase):
    def test_method_order_and_labels_are_frozen(self):
        self.assertEqual(
            ("full_cache", "k4", "k6", "k8"),
            tuple(method[0] for method in MODULE.METHODS),
        )
        self.assertEqual((None, 4, 6, 8), tuple(method[2] for method in MODULE.METHODS))

    def test_image_listing_requires_exact_requested_coverage(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for index in range(9):
                Image.new("RGB", (16, 12), color=(index, index, index)).save(
                    root / f"frame_{index:04d}.jpg"
                )
            self.assertEqual(9, len(MODULE.list_images(root, 9)))
            with self.assertRaisesRegex(ValueError, "found only 9"):
                MODULE.list_images(root, 10)

    def test_sampling_and_export_produce_exactly_twelve_pngs(self):
        rng = np.random.default_rng(7)
        pointmap = rng.normal(size=(28, 42, 3)).astype(np.float32)
        confidence = rng.uniform(size=(28, 42)).astype(np.float32)
        rgb = rng.uniform(size=(28, 42, 3)).astype(np.float32)
        sampled, colors = MODULE.sample_pointmap(pointmap, confidence, rgb, 2, 50)
        self.assertEqual(sampled.shape, colors.shape)
        results = []
        for slug, label, window, policy in MODULE.METHODS:
            results.append(
                {
                    "method_slug": slug,
                    "method_label": label,
                    "frame_number": 9,
                    "processed_frames": 9,
                    "rgb": rgb,
                    "depth": np.linalg.norm(pointmap, axis=-1),
                    "points": pointmap.reshape(-1, 3),
                    "colors": rgb.reshape(-1, 3),
                }
            )
        with tempfile.TemporaryDirectory() as directory:
            outputs = MODULE.export_panels(
                results, Path(directory), "test_scene", 10_000, 18.0, -70.0
            )
            self.assertEqual(12, len(outputs))
            self.assertTrue(all(path.is_file() for path in outputs))
            self.assertEqual(12, len(list(Path(directory).glob("*.png"))))


if __name__ == "__main__":
    unittest.main()
