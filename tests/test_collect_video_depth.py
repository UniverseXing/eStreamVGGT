"""CPU-only tests for the public VideoDepth result collector."""

from __future__ import annotations

import csv
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
COLLECTOR = REPO_ROOT / "scripts" / "reproduce" / "collect_video_depth.py"
FROZEN_SEQUENCE_FRAMES = {
    "bonn": {
        "balloon2": 110,
        "crowd2": 110,
        "crowd3": 110,
        "person_tracking2": 110,
        "synchronous": 110,
    },
    "kitti": {
        "2011_09_26_drive_0002_sync_02": 67,
        "2011_09_26_drive_0005_sync_02": 110,
        "2011_09_26_drive_0013_sync_02": 110,
        "2011_09_26_drive_0020_sync_02": 76,
        "2011_09_26_drive_0023_sync_02": 110,
        "2011_09_26_drive_0036_sync_02": 110,
        "2011_09_26_drive_0079_sync_02": 90,
        "2011_09_26_drive_0095_sync_02": 110,
        "2011_09_26_drive_0113_sync_02": 77,
        "2011_09_28_drive_0037_sync_02": 79,
        "2011_09_29_drive_0026_sync_02": 110,
        "2011_09_30_drive_0016_sync_02": 110,
        "2011_10_03_drive_0047_sync_02": 110,
    },
    "sintel": {
        "alley_1": 50,
        "alley_2": 50,
        "ambush_2": 21,
        "ambush_4": 33,
        "ambush_5": 50,
        "ambush_6": 20,
        "ambush_7": 50,
        "bamboo_1": 50,
        "bamboo_2": 50,
        "bandage_1": 50,
        "bandage_2": 50,
        "cave_2": 50,
        "cave_4": 50,
        "market_2": 50,
        "market_5": 50,
        "market_6": 40,
        "mountain_1": 50,
        "shaman_2": 50,
        "shaman_3": 50,
        "sleeping_1": 50,
        "sleeping_2": 50,
        "temple_2": 50,
        "temple_3": 50,
    },
}
DATASET_COUNTS = {key: len(value) for key, value in FROZEN_SEQUENCE_FRAMES.items()}
METHOD_CONFIGS = {
    "full_cache": ("", "full_cache"),
    "anchor_recent_dino_diverse_k4": ("4", "anchor_recent_dino_diverse_k4"),
    "anchor_recent_dino_diverse_k6": ("6", "anchor_recent_dino_diverse_k6"),
    "anchor_recent_dino_diverse_k8": ("8", "anchor_recent_dino_diverse_k8"),
}
METHODS = tuple(METHOD_CONFIGS)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


class VideoDepthCollectorTest(unittest.TestCase):
    def _write_method(
        self,
        root: Path,
        dataset: str,
        method: str,
        *,
        count: int | None = None,
        frame_limit: int | None = None,
    ) -> None:
        dataset_index = tuple(DATASET_COUNTS).index(dataset)
        method_index = METHODS.index(method)
        window, policy = METHOD_CONFIGS[method]
        result_dir = root / dataset / method
        result_dir.mkdir(parents=True)
        runtime_sequences = []
        metric_sequences = []
        count = DATASET_COUNTS[dataset] if count is None else count
        frozen_items = tuple(FROZEN_SEQUENCE_FRAMES[dataset].items())
        for index in range(count):
            name, frozen_frames = frozen_items[index]
            frames = frozen_frames if frame_limit is None else min(frame_limit, frozen_frames)
            inference_sec = 1.0 + 0.1 * index + 0.05 * method_index
            metrics = {
                "valid_pixels": 100 + index,
                "Abs Rel": 0.10 + 0.01 * dataset_index + 0.004 * method_index + 0.001 * index,
                "Sq Rel": 0.20 + 0.01 * dataset_index + 0.003 * method_index + 0.001 * index,
                "RMSE": 0.30 + 0.01 * dataset_index + 0.002 * method_index + 0.001 * index,
                "Log RMSE": 0.08 + 0.002 * dataset_index + 0.001 * method_index,
                "δ < 1.": 0.20 + 0.001 * index,
                "δ < 1.25": 0.80 - 0.003 * method_index + 0.001 * index,
                "δ < 1.25^2": 0.90 - 0.002 * method_index + 0.001 * index,
                "δ < 1.25^3": 0.95 - 0.001 * method_index + 0.001 * index,
            }
            runtime_sequences.append(
                {
                    "seq": name,
                    "status": "ok",
                    "num_frames": frames,
                    "inference_sec": inference_sec,
                    "fps_inference": frames / inference_sec,
                    "peak_allocated_mb": 1000.0 + 50 * method_index + index,
                    "peak_reserved_mb": 1200.0 + 50 * method_index + index,
                    "cache_window_size": window,
                    "cache_policy": policy,
                }
            )
            metric_sequences.append(
                {"sequence": name, "num_frames": frames, "metrics": metrics}
            )

        total_valid = sum(item["metrics"]["valid_pixels"] for item in metric_sequences)
        weighted = {
            key: sum(
                item["metrics"][key] * item["metrics"]["valid_pixels"]
                for item in metric_sequences
            )
            / total_valid
            for key in metric_sequences[0]["metrics"]
            if key != "valid_pixels"
        }
        total_frames = sum(item["num_frames"] for item in runtime_sequences)
        total_inference_sec = sum(item["inference_sec"] for item in runtime_sequences)
        summary = {
            "gpu_name": "Fixture RTX 6000 Ada",
            "torch_version": "2.3.1+cu121",
            "cuda_version": "12.1",
            "python_version": "3.11.9",
            "slurm_job_id": f"fixture-{dataset}-{method}",
            "hostname": f"fixture-node-{method}",
            "dataset": dataset,
            "input_size": 518,
            "pose_eval_stride": 1,
            "no_crop": True,
            "requested_max_frames": frame_limit,
            "num_sequences": count,
            "num_ok": count,
            "num_oom": 0,
            "total_frames": total_frames,
            "total_inference_sec": total_inference_sec,
            "fps_inference": total_frames / total_inference_sec,
            "max_peak_allocated_mb": max(
                item["peak_allocated_mb"] for item in runtime_sequences
            ),
            "max_peak_reserved_mb": max(
                item["peak_reserved_mb"] for item in runtime_sequences
            ),
            "cache_window_size": window,
            "cache_policy": policy,
        }
        (result_dir / "runtime_memory_rank0.json").write_text(
            json.dumps({"summary": summary, "sequences": runtime_sequences}),
            encoding="utf-8",
        )
        (result_dir / "result_scale.json").write_text(
            json.dumps(weighted), encoding="utf-8"
        )
        (result_dir / "result_scale_sequences.json").write_text(
            json.dumps(
                {
                    "dataset": dataset,
                    "align": "scale",
                    "num_sequences": count,
                    "weighted_average": weighted,
                    "sequences": metric_sequences,
                }
            ),
            encoding="utf-8",
        )

    def _run_collector(
        self,
        root: Path,
        datasets: tuple[str, ...],
        methods: tuple[str, ...],
        *,
        allow_subset: bool = False,
        check: bool = True,
    ) -> subprocess.CompletedProcess[str]:
        command = [
            sys.executable,
            str(COLLECTOR),
            "--results-root",
            str(root),
            "--output",
            str(root / "video_depth_results.csv"),
            "--datasets",
            *datasets,
            "--methods",
            *methods,
            "--bootstrap-samples",
            "1000",
            "--seed",
            "0",
        ]
        if allow_subset:
            command.append("--allow-subset")
        return subprocess.run(command, check=check, capture_output=True, text=True)

    def test_full_matrix_writes_six_stable_csvs(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for dataset in DATASET_COUNTS:
                for method in METHODS:
                    self._write_method(root, dataset, method)

            # Deliberately scramble CLI order: output order remains frozen.
            self._run_collector(
                root,
                ("sintel", "bonn", "kitti"),
                tuple(reversed(METHODS)),
            )

            aggregate = read_csv(root / "video_depth_results.csv")
            sequences = read_csv(root / "video_depth_sequence_results.csv")
            paired = read_csv(root / "video_depth_paired_bootstrap.csv")
            statistics = read_csv(root / "video_depth_sequence_statistics.csv")
            regret = read_csv(root / "video_depth_regret.csv")
            pareto = read_csv(root / "video_depth_pareto.csv")
            self.assertEqual(
                tuple(map(len, (aggregate, sequences, paired, statistics, regret, pareto))),
                (12, 164, 90, 132, 32, 12),
            )
            self.assertEqual(
                [(row["dataset"], row["method"]) for row in aggregate[:4]],
                [("bonn", method) for method in METHODS],
            )
            self.assertEqual(aggregate[0]["cache_window_size"], "")
            self.assertEqual({row["run_scope"] for row in aggregate}, {"frozen"})
            self.assertEqual(aggregate[0]["result_dir"], "bonn/full_cache")
            self.assertFalse(Path(aggregate[0]["result_dir"]).is_absolute())
            self.assertEqual(sequences[0]["source"], "bonn/full_cache")

    def test_single_sequence_statistics_are_finite(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self._write_method(root, "bonn", "full_cache", count=1)
            self._run_collector(
                root, ("bonn",), ("full_cache",), allow_subset=True
            )
            statistics = read_csv(root / "video_depth_sequence_statistics.csv")
            self.assertEqual(len(statistics), 11)
            for row in statistics:
                self.assertEqual(row["std"], "0.0")
                self.assertEqual(row["ci95_low"], row["ci95_high"])
            self.assertEqual(read_csv(root / "video_depth_paired_bootstrap.csv"), [])

    def test_single_pair_is_marked_insufficient(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for method in METHODS[:2]:
                self._write_method(root, "bonn", method, count=1)
            self._run_collector(
                root, ("bonn",), METHODS[:2], allow_subset=True
            )
            paired = read_csv(root / "video_depth_paired_bootstrap.csv")
            self.assertEqual(len(paired), 5)
            self.assertEqual(
                {row["significance"] for row in paired}, {"INSUFFICIENT_PAIRS"}
            )

    def test_truncated_full_coverage_needs_explicit_subset_mode(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self._write_method(root, "bonn", "full_cache", frame_limit=10)
            rejected = self._run_collector(
                root, ("bonn",), ("full_cache",), check=False
            )
            self.assertNotEqual(rejected.returncode, 0)
            self.assertIn("requested_max_frames=10", rejected.stderr)

            accepted = self._run_collector(
                root, ("bonn",), ("full_cache",), allow_subset=True
            )
            self.assertEqual(accepted.returncode, 0)
            self.assertEqual(
                {row["run_scope"] for row in read_csv(root / "video_depth_results.csv")},
                {"debug_subset"},
            )

    def test_duplicate_sequence_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self._write_method(root, "bonn", "full_cache", count=1)
            runtime_path = root / "bonn" / "full_cache" / "runtime_memory_rank0.json"
            payload = json.loads(runtime_path.read_text(encoding="utf-8"))
            payload["sequences"].append(dict(payload["sequences"][0]))
            runtime_path.write_text(json.dumps(payload), encoding="utf-8")
            result = self._run_collector(
                root,
                ("bonn",),
                ("full_cache",),
                allow_subset=True,
                check=False,
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("duplicate", result.stderr.lower())


if __name__ == "__main__":
    unittest.main()
