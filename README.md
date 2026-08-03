# eStreamVGGT

**DINO-guided bounded-memory streaming 3D perception built on StreamVGGT**

eStreamVGGT is a training-free extension of
[StreamVGGT](https://github.com/wzzheng/StreamVGGT). It replaces the unbounded
historical key-value (KV) cache with a small, frame-level cache selected from
the existing DINO features, couples the retained history of the aggregator and
camera head, and supports streaming input/output release for long sequences.

This is an independent research extension, not an official release by the
StreamVGGT authors. The original architecture, checkpoint, paper, project page,
and demo remain the work of the upstream authors:

- [StreamVGGT paper](https://arxiv.org/abs/2507.11539)
- [StreamVGGT project page](https://wzzheng.net/StreamVGGT)
- [StreamVGGT repository](https://github.com/wzzheng/StreamVGGT)
- [Upstream checkpoint](https://huggingface.co/lch01/StreamVGGT/)

The eStreamVGGT paper and project citation are coming soon.

## What this repository adds

- Fixed-budget DINO-guided cache configurations with 4, 6, or 8 retained frame
  states.
- Coupled pruning of aggregator and camera-head KV caches.
- A streaming-release path that does not retain consumed inputs or dense
  per-frame outputs.
- Reproducible evaluation entry points for VideoDepth, camera pose,
  multi-view reconstruction, and 1000-frame streaming.
- CSV supplementary tables with complete results and an exact note for
  normalised regret/oracle-win calculations.

No additional training or modified checkpoint is required.

## Frozen configurations

The following names are the public configuration identifiers used by the
reproduction scripts and generated metadata.

| Display name | Configuration identifier | Retained frame states | Intended role |
|---|---|---:|---|
| Full cache | `full_cache` | Unbounded | Quality/resource reference only |
| K4 | `anchor_recent_dino_diverse_k4` | 4 | Primary compact configuration |
| K6 | `anchor_recent_dino_diverse_k6` | 6 | Reconstruction-oriented robust alternative |
| K8 | `anchor_recent_dino_diverse_k8` | 8 | Long-sequence local-pose specialist |

K4 combines a stable anchor, DINO-diverse historical states, and the current
context. K6 assigns additional capacity to recent context. Once warmed up, K8
uses an anchor, long/middle/near temporal landmarks, and four recent frames.
The value of `K` limits cached frame states, not the total input length.

## Results snapshot

The table below reports the frozen VideoDepth results on one NVIDIA RTX 6000
Ada GPU. AbsRel is lower-is-better; allocated memory is the maximum CUDA memory
allocated by the run. K6 and K8, all other metrics, sequence-level values, and
paired statistics are available in [`supplementary/`](supplementary/).

| Dataset | Method | AbsRel ↓ | FPS ↑ | Peak allocated GiB ↓ |
|---|---|---:|---:|---:|
| Bonn | Full cache | 0.0746 | 3.08 | 21.13 |
| Bonn | K4 | 0.0755 | 7.63 | 10.32 |
| KITTI | Full cache | 0.1726 | 5.99 | 12.43 |
| KITTI | K4 | 0.1334 | 9.06 | 8.00 |
| Sintel | Full cache | 0.3232 | 6.87 | 10.30 |
| Sintel | K4 | 0.3161 | 8.54 | 7.90 |

On three held-out TUM RGB-D sequences, full cache completed the 100-frame runs
but exhausted GPU memory while processing the requested 250-frame runs at
about frame 195. K4, K6, and K8 completed every 1000-frame run. Their maximum
1000-frame allocated peaks were 8026, 8406, and 8783 MiB, respectively, with
zero additional GPU peak from 500 to 1000 frames. Memory boundedness does not,
by itself, imply that pose error remains bounded; the pose limitations and
per-sequence results are reported in the supplementary package.

These resource values are hardware- and software-dependent. Reproduce method
comparisons on the same GPU and environment rather than comparing isolated
numbers across machines.

## Quick start

```bash
git clone https://github.com/UniverseXing/eStreamVGGT.git
cd eStreamVGGT

conda create -n StreamVGGT python=3.11 cmake=3.14.0
conda activate StreamVGGT
python -m pip install -r requirements_eval.txt
conda install 'llvm-openmp<16'
```

Download the original StreamVGGT `checkpoints.pth` from the
[upstream checkpoint page](https://huggingface.co/lch01/StreamVGGT/) and place
it at:

```bash
hf download lch01/StreamVGGT checkpoints.pth --local-dir ckpt
```

Then run the small validation entry point:

```bash
bash scripts/reproduce/smoke_test.sh
```

See [`docs/INSTALLATION.md`](docs/INSTALLATION.md) for the complete environment
setup and [`docs/DATASETS.md`](docs/DATASETS.md) before running full benchmarks.

## Reproducing the evaluation

The public wrappers are ordinary Bash scripts: they do not contain SLURM
directives, module commands, user-specific paths, or Conda activation. Activate
the intended environment before invoking them.

| Entry point | Purpose |
|---|---|
| `scripts/reproduce/smoke_test.sh` | Validate imports, checkpoint access, method mappings, and a short inference path |
| `scripts/reproduce/download_kitti.sh` | Download, prepare, and validate the frozen KITTI VideoDepth protocol |
| `scripts/reproduce/download_tum_long.sh` | Download and validate the three held-out raw-TUM sequences |
| `scripts/reproduce/run_video_depth.sh` | Bonn, KITTI, and Sintel VideoDepth evaluation |
| `scripts/reproduce/run_pose.sh` | Sintel, ScanNet, and TUM camera-pose evaluation |
| `scripts/reproduce/run_reconstruction.sh` | 7-Scenes, NRGBD, ETH3D, and TUM reconstruction evaluation |
| `scripts/reproduce/run_long_sequence.sh` | Raw-TUM 100/250/500/1000-frame streaming evaluation |
| `scripts/reproduce/build_supplementary.sh` | Rebuild supplementary CSV tables and calculation note from result sources |
| `scripts/reproduce/run_all.sh` | Run the selected evaluation groups in sequence |

For example:

```bash
bash scripts/reproduce/run_video_depth.sh

# Inspect the commands for one dataset and method without allocating a GPU.
DRY_RUN=1 \
DATASETS=bonn \
METHODS=anchor_recent_dino_diverse_k4 \
  bash scripts/reproduce/run_video_depth.sh
```

The benchmark wrappers evaluate `full_cache`,
`anchor_recent_dino_diverse_k4`,
`anchor_recent_dino_diverse_k6`, and
`anchor_recent_dino_diverse_k8` unless their documented method selection is
overridden. The smoke test intentionally runs one bounded method. Full details,
overrides, output files, frozen protocols, and expected failure behavior are in
[`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md).
By default, inference results are written below `eval_results/reproduce/`.

## Repository layout

```text
eStreamVGGT/
├── ckpt/                       # local checkpoints (not redistributed)
├── data/eval/                  # local evaluation datasets
├── docs/                       # installation, data, and reproduction guides
├── scripts/reproduce/          # stable public reproduction entry points
├── src/streamvggt/             # model and bounded-cache implementation
├── src/eval/                   # task evaluators
└── supplementary/              # Complete CSV results and calculation note
```

The historical `run_stage*.sh` files document the internal experiment
development process. New users should use `scripts/reproduce/`; stage-numbered
scripts are not the stable public interface.

## Supplementary package

[`supplementary/README.md`](supplementary/README.md) indexes 15 CSV tables with
the complete VideoDepth, pose, reconstruction, cross-task, and long-sequence
results. [`supplementary/CALCULATION_METHODS.md`](supplementary/CALCULATION_METHODS.md)
defines the normalised-regret equations, bounded-oracle scope, tie rule,
evaluation units, and aggregation procedure.

The generated package is committed to the repository. Rebuilding it requires
the author-side result CSV/archives at their documented default paths:

```bash
bash scripts/reproduce/build_supplementary.sh
```

If a source is unavailable, the builder reports the missing file instead of
silently producing a partial package.

## Documentation

- [Installation and checkpoint setup](docs/INSTALLATION.md)
- [Evaluation dataset layouts](docs/DATASETS.md)
- [End-to-end reproducibility guide](docs/REPRODUCIBILITY.md)
- [Relationship to upstream StreamVGGT](docs/UPSTREAM.md)

## Training and demo

The focus of this repository is training-free bounded-cache inference and its
evaluation. The original training, fine-tuning, and Gradio demo code is
retained for compatibility. For the authoritative training and demo
instructions, use the
[upstream StreamVGGT documentation](https://github.com/wzzheng/StreamVGGT).

## License and attribution

This repository retains the upstream
[CC BY-NC-SA 4.0 license](LICENSE.txt). Dataset, checkpoint, and third-party
component licenses apply separately. See [`docs/UPSTREAM.md`](docs/UPSTREAM.md)
for the provenance of the base repository and a summary of eStreamVGGT changes.

If this code is useful, please cite the original StreamVGGT paper. A separate
eStreamVGGT citation will be added when its paper is public.

```bibtex
@article{streamVGGT,
  title={Streaming 4D Visual Geometry Transformer},
  author={Dong Zhuo and Wenzhao Zheng and Jiahe Guo and Yuqi Wu and Jie Zhou and Jiwen Lu},
  journal={arXiv preprint arXiv:2507.11539},
  year={2025}
}
```

## Acknowledgements

eStreamVGGT inherits substantial code and evaluation infrastructure from
StreamVGGT and its dependencies, including VGGT, DUSt3R, MonST3R, Spann3R,
CUT3R, and Point3R. We thank their authors and contributors.
