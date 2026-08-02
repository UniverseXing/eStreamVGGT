"""CPU-only regression tests for the public result-completeness gates."""

from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
POSE_SUMMARIZER = REPO_ROOT / "scripts" / "summarize_stage3_3_pose.py"
RECON_SUMMARIZER = REPO_ROOT / "scripts" / "summarize_stage3_3b_recon.py"
RECON_RUNNER = REPO_ROOT / "scripts" / "reproduce" / "run_reconstruction.sh"
LONG_SUMMARIZER = REPO_ROOT / "scripts" / "summarize_stage4c.py"


def run_script(script: Path, *arguments: object) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(script), *(str(argument) for argument in arguments)],
        check=False,
        capture_output=True,
        text=True,
    )


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


class PoseResultGateTest(unittest.TestCase):
    METHOD_CONFIGS = {
        "full_cache": ("full_cache", None),
        "anchor_recent_dino_diverse_k4": ("anchor_recent_dino_diverse_k4", 4),
        "anchor_recent_dino_diverse_k6": ("anchor_recent_dino_diverse_k6", 6),
        "anchor_recent_dino_diverse_k8": ("anchor_recent_dino_diverse_k8", 8),
    }

    def _write_run(
        self,
        root: Path,
        method: str,
        sequences: list[dict[str, object]],
        *,
        num_failed: int = 0,
        gpu_name: str = "NVIDIA RTX 6000 Ada Generation",
        torch_version: str = "2.3.1",
        cuda_version: str = "12.1",
        python_version: str = "3.11.9",
        metadata_method: str | None = None,
        requested_max_frames: int | None = None,
    ) -> None:
        cache_policy, cache_window_size = self.METHOD_CONFIGS[
            metadata_method or method
        ]
        write_json(
            root / f"sintel_reproduce_{method}" / "pose_metrics.json",
            {
                "summary": {
                    "dataset": "sintel",
                    "cache_policy": cache_policy,
                    "cache_window_size": cache_window_size,
                    "gpu_name": gpu_name,
                    "torch_version": torch_version,
                    "cuda_version": cuda_version,
                    "python_version": python_version,
                    "slurm_job_id": f"fixture-{method}",
                    "hostname": f"fixture-node-{method}",
                    "input_size": 518,
                    "stride": 1,
                    "requested_max_frames": requested_max_frames,
                    "num_sequences": len(sequences),
                    "num_successful": len(sequences) - num_failed,
                    "num_failed": num_failed,
                },
                "sequences": sequences,
            },
        )

    @staticmethod
    def _ok_sequences(second_frames: int = 20) -> list[dict[str, object]]:
        return [
            {"sequence": "alley_1", "num_frames": 10, "status": "ok"},
            {
                "sequence": "alley_2",
                "num_frames": second_frames,
                "status": "ok",
            },
        ]

    @staticmethod
    def _frozen_sintel_sequences() -> list[dict[str, object]]:
        frames = {
            "alley_2": 50,
            "ambush_4": 33,
            "ambush_5": 50,
            "ambush_6": 20,
            "cave_2": 50,
            "cave_4": 50,
            "market_2": 50,
            "market_5": 50,
            "market_6": 40,
            "shaman_3": 50,
            "sleeping_1": 50,
            "sleeping_2": 50,
            "temple_2": 50,
            "temple_3": 50,
        }
        return [
            {"sequence": sequence, "num_frames": count, "status": "ok"}
            for sequence, count in frames.items()
        ]

    def _summarize(
        self,
        root: Path,
        *,
        allow_subset: bool = True,
    ) -> subprocess.CompletedProcess[str]:
        arguments: list[object] = [
            "--results-root",
            root,
            "--name-filter",
            "reproduce",
            "--output",
            root / "pose.csv",
            "--expected-runs",
            2,
            "--require-all-success",
        ]
        if allow_subset:
            arguments.append("--allow-subset")
        return run_script(
            POSE_SUMMARIZER,
            *arguments,
        )

    def test_complete_paired_matrix_passes(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self._write_run(root, "full_cache", self._ok_sequences())
            self._write_run(
                root, "anchor_recent_dino_diverse_k4", self._ok_sequences()
            )

            result = self._summarize(root)

            self.assertEqual(result.returncode, 0, result.stderr)
            with (root / "pose.csv").open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 2)
            self.assertEqual({row["run_scope"] for row in rows}, {"debug_subset"})

    def test_failed_sequence_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self._write_run(root, "full_cache", self._ok_sequences())
            failed = self._ok_sequences()
            failed[1]["status"] = "error"
            self._write_run(
                root, "anchor_recent_dino_diverse_k4", failed, num_failed=1
            )

            result = self._summarize(root)

            self.assertNotEqual(result.returncode, 0)
            self.assertIn("failed pose sequence", result.stderr)

    def test_mismatched_sequence_coverage_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self._write_run(root, "full_cache", self._ok_sequences())
            self._write_run(
                root,
                "anchor_recent_dino_diverse_k4",
                self._ok_sequences(second_frames=19),
            )

            result = self._summarize(root)

            self.assertNotEqual(result.returncode, 0)
            self.assertIn("coverage differs", result.stderr)

    def test_missing_core_provenance_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self._write_run(root, "full_cache", self._ok_sequences())
            self._write_run(
                root,
                "anchor_recent_dino_diverse_k4",
                self._ok_sequences(),
                cuda_version="",
            )

            result = self._summarize(root)

            self.assertNotEqual(result.returncode, 0)
            self.assertIn("missing pose cuda_version", result.stderr)

    def test_inconsistent_core_provenance_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self._write_run(root, "full_cache", self._ok_sequences())
            self._write_run(
                root,
                "anchor_recent_dino_diverse_k4",
                self._ok_sequences(),
                gpu_name="Different GPU",
            )

            result = self._summarize(root)

            self.assertNotEqual(result.returncode, 0)
            self.assertIn("inconsistent pose gpu_name", result.stderr)

    def test_frozen_signature_and_configuration_pass(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            sequences = self._frozen_sintel_sequences()
            self._write_run(root, "full_cache", sequences)
            self._write_run(root, "anchor_recent_dino_diverse_k4", sequences)

            result = self._summarize(root, allow_subset=False)

            self.assertEqual(result.returncode, 0, result.stderr)
            with (root / "pose.csv").open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual({row["run_scope"] for row in rows}, {"frozen"})

    def test_nonfrozen_signature_is_rejected_without_allow_subset(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self._write_run(root, "full_cache", self._ok_sequences())
            self._write_run(
                root, "anchor_recent_dino_diverse_k4", self._ok_sequences()
            )

            result = self._summarize(root, allow_subset=False)

            self.assertNotEqual(result.returncode, 0)
            self.assertIn("frozen sequence/frame signature mismatch", result.stderr)

    def test_frozen_max_frames_metadata_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            sequences = self._frozen_sintel_sequences()
            self._write_run(root, "full_cache", sequences)
            self._write_run(
                root,
                "anchor_recent_dino_diverse_k4",
                sequences,
                requested_max_frames=50,
            )

            result = self._summarize(root, allow_subset=False)

            self.assertNotEqual(result.returncode, 0)
            self.assertIn("frozen configuration mismatch", result.stderr)

    def test_directory_method_metadata_mismatch_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self._write_run(root, "full_cache", self._ok_sequences())
            self._write_run(
                root,
                "anchor_recent_dino_diverse_k4",
                self._ok_sequences(),
                metadata_method="anchor_recent_dino_diverse_k6",
            )

            result = self._summarize(root)

            self.assertNotEqual(result.returncode, 0)
            self.assertIn("method metadata mismatch", result.stderr)


class ReconstructionResultGateTest(unittest.TestCase):
    METHOD_CONFIGS = {
        "full_cache": ("full_cache", None),
        "anchor_recent_dino_diverse_k4": (
            "anchor_recent_dino_diverse_k4",
            4,
        ),
        "anchor_recent_dino_diverse_k6": (
            "anchor_recent_dino_diverse_k6",
            6,
        ),
        "anchor_recent_dino_diverse_k8": (
            "anchor_recent_dino_diverse_k8",
            8,
        ),
    }

    def _write_run(
        self,
        root: Path,
        method: str,
        sequences: list[dict[str, object]],
        *,
        num_failed: int = 0,
        gpu_name: str = "NVIDIA RTX 6000 Ada Generation",
        torch_version: str = "2.3.1",
        cuda_version: str = "12.1",
        python_version: str = "3.11.9",
        dataset: str = "7scenes",
        protocol: str = "dense",
        sampling_stride: object = 50,
        prefix_frames: tuple[int, ...] = (4, 6, 8, 10),
        metadata_method: str | None = None,
        payload_overrides: dict[str, object] | None = None,
        summary_overrides: dict[str, object] | None = None,
    ) -> None:
        metadata_method = metadata_method or method
        cache_policy, cache_window = self.METHOD_CONFIGS[metadata_method]
        summary = {
            "dataset": dataset,
            "protocol": protocol,
            "sampling_stride": sampling_stride,
            "cache_policy": cache_policy,
            "cache_window_size": cache_window,
            "num_sequences": len(sequences),
            "num_successful": len(sequences) - num_failed,
            "num_failed": num_failed,
        }
        summary.update(summary_overrides or {})
        payload = {
            "model_name": "StreamVGGT",
            "gpu_name": gpu_name,
            "torch_version": torch_version,
            "cuda_version": cuda_version,
            "python_version": python_version,
            "slurm_job_id": f"fixture-{method}",
            "hostname": f"fixture-node-{method}",
            "input_size": 518,
            "use_proj": False,
            "max_scenes": None,
            "max_frames": None,
            "icp_threshold": 0.1,
            "protocol": protocol,
            "sampling_strides": {dataset: sampling_stride},
            "prefix_frames": list(prefix_frames),
            "cache_policy": cache_policy,
            "cache_window_size": cache_window,
            "seed": 0,
            "datasets": {
                dataset: {
                    "summary": summary,
                    "prefix_summaries": [
                        {
                            "dataset": dataset,
                            "protocol": protocol,
                            "prefix_frames": prefix,
                            "cache_policy": cache_policy,
                            "cache_window_size": cache_window,
                            "num_sequences": len(sequences),
                        }
                        for prefix in prefix_frames
                    ],
                    "sequences": sequences,
                }
            },
        }
        payload.update(payload_overrides or {})
        write_json(
            root
            / f"streamvggt_stage3_3b_{method}"
            / "reconstruction_metrics.json",
            payload,
        )

    @staticmethod
    def _ok_sequences(second_frames: int = 5) -> list[dict[str, object]]:
        return [
            {
                "sequence": "chess/seq-01",
                "num_frames": 4,
                "status": "ok",
                "pose_status": "ok",
            },
            {
                "sequence": "office/seq-01",
                "num_frames": second_frames,
                "status": "ok",
                "pose_status": "ok",
            },
        ]

    @staticmethod
    def _frozen_eth3d_sequences() -> list[dict[str, object]]:
        return [
            {
                "sequence": sequence,
                "num_frames": 10,
                "status": "ok",
                "pose_status": "ok",
            }
            for sequence in (
                "courtyard",
                "delivery_area",
                "electro",
                "facade",
                "kicker",
                "meadow",
                "office",
                "pipes",
                "playground",
                "relief",
                "relief_2",
                "terrace",
                "terrains",
            )
        ]

    def _summarize(
        self, root: Path, *, allow_subset: bool = True
    ) -> subprocess.CompletedProcess[str]:
        arguments: list[object] = [
            "--results-root",
            root,
            "--name-filter",
            "stage3_3b",
            "--output",
            root / "reconstruction.csv",
            "--expected-runs",
            2,
            "--require-all-success",
        ]
        if allow_subset:
            arguments.append("--allow-subset")
        return run_script(
            RECON_SUMMARIZER,
            *arguments,
        )

    def test_complete_paired_matrix_with_valid_pose_metrics_passes(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self._write_run(root, "full_cache", self._ok_sequences())
            self._write_run(
                root,
                "anchor_recent_dino_diverse_k4",
                self._ok_sequences(),
            )

            result = self._summarize(root)

            self.assertEqual(result.returncode, 0, result.stderr)
            with (root / "reconstruction.csv").open(
                newline="", encoding="utf-8"
            ) as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 12)
            self.assertEqual({row["run_scope"] for row in rows}, {"debug_subset"})
            self.assertEqual(
                {row["gpu_name"] for row in rows},
                {"NVIDIA RTX 6000 Ada Generation"},
            )
            self.assertEqual({row["input_size"] for row in rows}, {"518"})
            self.assertEqual({row["use_proj"] for row in rows}, {"False"})

    def test_failed_sequence_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self._write_run(root, "full_cache", self._ok_sequences())
            failed = self._ok_sequences()
            failed[0]["status"] = "error"
            self._write_run(
                root,
                "anchor_recent_dino_diverse_k4",
                failed,
                num_failed=1,
            )

            result = self._summarize(root)

            self.assertNotEqual(result.returncode, 0)
            self.assertIn("failed reconstruction sequence", result.stderr)

    def test_mismatched_sequence_coverage_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self._write_run(root, "full_cache", self._ok_sequences())
            self._write_run(
                root,
                "anchor_recent_dino_diverse_k4",
                self._ok_sequences(second_frames=6),
            )

            result = self._summarize(root)

            self.assertNotEqual(result.returncode, 0)
            self.assertIn("coverage differs", result.stderr)

    def test_failed_pose_metric_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self._write_run(root, "full_cache", self._ok_sequences())
            bad_pose = self._ok_sequences()
            bad_pose[1]["pose_status"] = "failed"
            self._write_run(root, "anchor_recent_dino_diverse_k4", bad_pose)

            result = self._summarize(root)

            self.assertNotEqual(result.returncode, 0)
            self.assertIn("failed reconstruction pose metric", result.stderr)

    def test_missing_core_provenance_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self._write_run(root, "full_cache", self._ok_sequences())
            self._write_run(
                root,
                "anchor_recent_dino_diverse_k4",
                self._ok_sequences(),
                cuda_version="",
            )

            result = self._summarize(root)

            self.assertNotEqual(result.returncode, 0)
            self.assertIn("missing reconstruction cuda_version", result.stderr)

    def test_inconsistent_core_provenance_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self._write_run(root, "full_cache", self._ok_sequences())
            self._write_run(
                root,
                "anchor_recent_dino_diverse_k4",
                self._ok_sequences(),
                gpu_name="Different GPU",
            )

            result = self._summarize(root)

            self.assertNotEqual(result.returncode, 0)
            self.assertIn("inconsistent reconstruction gpu_name", result.stderr)

    def test_frozen_signature_and_configuration_pass(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            sequences = self._frozen_eth3d_sequences()
            for method in ("full_cache", "anchor_recent_dino_diverse_k4"):
                self._write_run(
                    root,
                    method,
                    sequences,
                    dataset="eth3d",
                    sampling_stride="random_10",
                )

            result = self._summarize(root, allow_subset=False)

            self.assertEqual(result.returncode, 0, result.stderr)
            with (root / "reconstruction.csv").open(
                newline="", encoding="utf-8"
            ) as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual({row["run_scope"] for row in rows}, {"frozen"})

    def test_nonfrozen_signature_is_rejected_without_allow_subset(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self._write_run(root, "full_cache", self._ok_sequences())
            self._write_run(
                root,
                "anchor_recent_dino_diverse_k4",
                self._ok_sequences(),
            )

            result = self._summarize(root, allow_subset=False)

            self.assertNotEqual(result.returncode, 0)
            self.assertIn("frozen reconstruction sequence/frame signature", result.stderr)

    def test_frozen_cap_metadata_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            sequences = self._frozen_eth3d_sequences()
            self._write_run(
                root,
                "full_cache",
                sequences,
                dataset="eth3d",
                sampling_stride="random_10",
            )
            self._write_run(
                root,
                "anchor_recent_dino_diverse_k4",
                sequences,
                dataset="eth3d",
                sampling_stride="random_10",
                payload_overrides={"max_frames": 10},
            )

            result = self._summarize(root, allow_subset=False)

            self.assertNotEqual(result.returncode, 0)
            self.assertIn("non-frozen reconstruction max_frames", result.stderr)

    def test_directory_method_metadata_mismatch_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self._write_run(root, "full_cache", self._ok_sequences())
            self._write_run(
                root,
                "anchor_recent_dino_diverse_k4",
                self._ok_sequences(),
                metadata_method="anchor_recent_dino_diverse_k6",
            )

            result = self._summarize(root)

            self.assertNotEqual(result.returncode, 0)
            self.assertIn("method metadata mismatch", result.stderr)

    def test_runner_marks_an_explicit_override_as_debug_subset(self):
        base_environment = {
            "PATH": os.environ["PATH"],
            "HOME": os.environ.get("HOME", str(REPO_ROOT)),
            "DRY_RUN": "1",
            "METHODS": "full_cache",
            "DATASETS": "tum",
        }
        frozen = subprocess.run(
            ["bash", str(RECON_RUNNER)],
            cwd=REPO_ROOT,
            env=base_environment,
            check=False,
            capture_output=True,
            text=True,
        )
        debug_environment = {**base_environment, "SEED": "0"}
        debug = subprocess.run(
            ["bash", str(RECON_RUNNER)],
            cwd=REPO_ROOT,
            env=debug_environment,
            check=False,
            capture_output=True,
            text=True,
        )

        self.assertEqual(frozen.returncode, 0, frozen.stderr)
        self.assertNotIn("--allow-subset", frozen.stdout)
        self.assertEqual(debug.returncode, 0, debug.stderr)
        self.assertIn("--allow-subset", debug.stdout)
        self.assertIn("debug_subset reconstruction overrides: SEED", debug.stdout)


class LongSequenceResultGateTest(unittest.TestCase):
    def _write_cell(
        self,
        root: Path,
        method: str,
        sequence: str,
        frames: int,
        *,
        gpu_name: str = "NVIDIA RTX 6000 Ada Generation",
        torch_version: str = "2.3.1",
        cuda_version: str = "12.1",
        python_version: str = "3.11.9",
        pose_status: str = "ok",
        run_id: str = "fixture-run",
    ) -> None:
        write_json(
            root / method / sequence / str(frames) / "stage4c_metrics.json",
            {
                "run_scope": "frozen",
                "run_id": run_id,
                "method": method,
                "sequence": sequence,
                "num_frames": frames,
                "status": "ok",
                "pose_status": pose_status,
                "gpu_name": gpu_name,
                "torch_version": torch_version,
                "cuda_version": cuda_version,
                "python_version": python_version,
            },
        )

    def test_include_cells_ignore_stale_results_and_validate_provenance(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self._write_cell(root, "full_cache", "freiburg1_room", 100)
            self._write_cell(root, "anchor_recent_dino_diverse_k4", "freiburg1_room", 100)
            self._write_cell(
                root,
                "stale_method",
                "stale_sequence",
                50,
                gpu_name="Different stale GPU",
                torch_version="0.0",
                cuda_version="0.0",
            )
            output = root / "long.csv"

            result = run_script(
                LONG_SUMMARIZER,
                "--results-root",
                root,
                "--output",
                output,
                "--include-cell",
                "full_cache|freiburg1_room|100",
                "--include-cell",
                "anchor_recent_dino_diverse_k4|freiburg1_room|100",
                "--require-consistent-provenance",
                "--require-pose-success",
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            with output.open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 2)
            self.assertNotIn("stale_method", {row["method"] for row in rows})

    def test_missing_planned_cell_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self._write_cell(root, "full_cache", "freiburg1_room", 100)

            result = run_script(
                LONG_SUMMARIZER,
                "--results-root",
                root,
                "--output",
                root / "long.csv",
                "--include-cell",
                "full_cache|freiburg1_room|100",
                "--include-cell",
                "anchor_recent_dino_diverse_k4|freiburg1_room|100",
            )

            self.assertNotEqual(result.returncode, 0)
            self.assertIn("missing planned Stage 4C result", result.stderr)

    def test_inconsistent_included_provenance_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self._write_cell(root, "full_cache", "freiburg1_room", 100)
            self._write_cell(
                root,
                "anchor_recent_dino_diverse_k4",
                "freiburg1_room",
                100,
                gpu_name="Different GPU",
            )

            result = run_script(
                LONG_SUMMARIZER,
                "--results-root",
                root,
                "--output",
                root / "long.csv",
                "--include-cell",
                "full_cache|freiburg1_room|100",
                "--include-cell",
                "anchor_recent_dino_diverse_k4|freiburg1_room|100",
                "--require-consistent-provenance",
            )

            self.assertNotEqual(result.returncode, 0)
            self.assertIn("inconsistent Stage 4C gpu_name", result.stderr)

    def test_successful_inference_with_failed_pose_metric_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self._write_cell(
                root,
                "anchor_recent_dino_diverse_k4",
                "freiburg1_room",
                100,
                pose_status="failed",
            )

            result = run_script(
                LONG_SUMMARIZER,
                "--results-root",
                root,
                "--output",
                root / "long.csv",
                "--include-cell",
                "anchor_recent_dino_diverse_k4|freiburg1_room|100",
                "--require-pose-success",
            )

            self.assertNotEqual(result.returncode, 0)
            self.assertIn(
                "successful Stage 4C inference with failed pose metric",
                result.stderr,
            )

    def test_run_scope_mismatch_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self._write_cell(root, "full_cache", "freiburg1_room", 100)
            result = run_script(
                LONG_SUMMARIZER,
                "--results-root",
                root,
                "--output",
                root / "long.csv",
                "--include-cell",
                "full_cache|freiburg1_room|100",
                "--expected-run-scope",
                "debug_subset",
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("run_scope mismatch", result.stderr)

    def test_stale_run_id_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self._write_cell(
                root, "full_cache", "freiburg1_room", 100, run_id="old-run"
            )
            result = run_script(
                LONG_SUMMARIZER,
                "--results-root",
                root,
                "--output",
                root / "long.csv",
                "--include-cell",
                "full_cache|freiburg1_room|100",
                "--expected-run-id",
                "full_cache|freiburg1_room|100|new-run",
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("stale Stage 4C result", result.stderr)


if __name__ == "__main__":
    unittest.main()
