# Reproducibility guide

This document describes the stable public entry points for reproducing the
frozen eStreamVGGT evaluation. It separates portable experiment commands from
site-specific scheduler setup.

## Scope

The final comparison contains exactly four configurations:

```text
full_cache
anchor_recent_dino_diverse_k4
anchor_recent_dino_diverse_k6
anchor_recent_dino_diverse_k8
```

The same original StreamVGGT checkpoint and `518` input size are used by every
method. K4/K6/K8 are inference-time cache policies; no method-specific training
or fine-tuning is performed.

Historical implementation names may remain in old result manifests so the
development history stays auditable. New runs and paper-facing output use only
the public names above.

## Before running

1. Complete [`INSTALLATION.md`](INSTALLATION.md) and activate the environment.
2. Place the original checkpoint at `ckpt/checkpoints.pth`.
3. Prepare the required datasets according to [`DATASETS.md`](DATASETS.md).
4. Run `bash scripts/reproduce/smoke_test.sh`.
5. Confirm that the intended GPU is idle and visible to PyTorch.

All commands below are invoked from the repository root.

The wrappers take configuration through environment variables. They accept
`--help`, but do not use task configuration as positional command-line
arguments. Set `DRY_RUN=1` to print the resolved commands without checking the
GPU or checkpoint:

```bash
DRY_RUN=1 bash scripts/reproduce/run_video_depth.sh
```

## Common controls

| Variable | Default | Meaning |
|---|---|---|
| `METHODS` | All four frozen methods | Space-separated canonical method identifiers |
| `WEIGHTS` | `ckpt/checkpoints.pth` | StreamVGGT checkpoint path |
| `RESULTS_ROOT` | `eval_results/reproduce` | Parent directory for inference results |
| `MAX_FRAMES` | Unset | Optional task-level frame limit; for long sequences it replaces `LENGTHS` |
| `DRY_RUN` | `0` | Set to `1` to print commands without executing them |
| `PYTHON_BIN` | Active-environment Python | Explicit Python executable override |
| `ACCELERATE_BIN` | Active-environment `accelerate` | Explicit Accelerate executable override |

Real runs prefer the interpreter and `accelerate` executable under the active
`CONDA_PREFIX`. Space-separated values must be quoted, for example:

```bash
METHODS="full_cache anchor_recent_dino_diverse_k4" \
DATASETS="bonn sintel" \
  bash scripts/reproduce/run_video_depth.sh
```

When changing a dataset, method, or sequence subset, use a fresh task-specific
output root. The public collectors deliberately reject extra stale run
directories, failed cells, unequal sequence coverage, or unequal frame counts
instead of silently averaging a mixed matrix.

## Public entry points

### Smoke test

```bash
bash scripts/reproduce/smoke_test.sh
```

The smoke test is a pre-flight check, not a reportable benchmark. It verifies
the active Python environment, checkpoint, canonical method mapping, and a
short inference path. It defaults to K4 and the repository example images in
`examples/example_building`. Override these with `METHOD` and `IMAGES_DIR`.
When `MAX_FRAMES` is set, it must exceed the selected cache window so the smoke
test actually exercises pruning.

### VideoDepth

```bash
bash scripts/reproduce/run_video_depth.sh
```

The frozen VideoDepth suite contains Bonn, KITTI, and Sintel. It stores
per-method predictions and official depth metrics below
`eval_results/reproduce/video_depth/` by default.
Sequence-level aggregation, paired bootstrap statistics, regret, and Pareto
tables are derived from those raw results; do not substitute an unpaired mean
comparison for the paired analysis.

VideoDepth controls are:

| Variable | Default | Meaning |
|---|---|---|
| `DATASETS` | `bonn kitti sintel` | Space-separated dataset keys |
| `VIDEO_DEPTH_RESULTS_ROOT` | `<RESULTS_ROOT>/video_depth` | Task output override |
| `KITTI_ROOT` | `data/eval/kitti` | Prepared KITTI root |
| `SEQ_LIST` | Unset | Optional space-separated sequence subset; requires exactly one selected dataset |
| `MAX_FRAMES` | Unset | Optional per-sequence frame cap |
| `SIZE` | `518` | Fixed at `518`; other values are rejected by the frozen evaluator |
| `ALIGN` | `scale` | Frozen at `scale`; any other value is rejected |
| `BOOTSTRAP_SAMPLES` | `10000` | Number of paired-bootstrap draws (minimum `1000`) |
| `BOOTSTRAP_SEED` | `0` | Paired-bootstrap random seed |

After all selected cells finish, the wrapper validates sequence coverage,
the exact frozen sequence names and frame counts, method metadata, provenance,
and valid-pixel-weighted aggregates. Setting `SEQ_LIST` or `MAX_FRAMES` marks
the output as a debug/subset run: coverage must still match across methods, but
it cannot pass as the frozen full matrix.
It then writes six portable CSVs under `<VIDEO_DEPTH_RESULTS_ROOT>/`:

- `video_depth_results.csv` (official valid-pixel-weighted aggregates);
- `video_depth_sequence_results.csv`;
- `video_depth_paired_bootstrap.csv`;
- `video_depth_sequence_statistics.csv`;
- `video_depth_regret.csv`; and
- `video_depth_pareto.csv`.

Paths in the `result_dir` and `source` columns are relative to the VideoDepth
results root, so moving a complete result directory does not invalidate the
tables. The frozen full matrix contains 12 aggregate rows, 164 per-sequence
rows, 90 paired comparisons, 132 sequence-statistic rows, 32 regret rows, and
12 Pareto rows.
Every generated CSV includes `run_scope`: `frozen` for exact protocol coverage
or `debug_subset` when `SEQ_LIST`/`MAX_FRAMES` is used.
The statistics and paired-comparison tables also record the bootstrap sample
count and RNG seed used to form their confidence intervals.

The frozen KITTI scorer requires the complete prepared ground-truth protocol.
If `DATASETS` contains `kitti`, setting `MAX_FRAMES` is rejected rather than
silently scoring a mismatched truncation. Use Bonn or Sintel for a shortened
debug run, and run KITTI on its full prepared set.

### Camera pose

```bash
bash scripts/reproduce/run_pose.sh
```

The pose suite contains Sintel, ScanNet, and processed TUM sequences. Its
primary output for each method/dataset is `pose_metrics.json`, with raw and
aligned trajectories under a `trajectories/` directory. A sequence-level
metric failure is recorded as a failure and must not be silently dropped.

Pose controls are:

| Variable | Default | Meaning |
|---|---|---|
| `DATASETS` | `sintel scannet tum` | Space-separated dataset keys |
| `POSE_RESULTS_ROOT` | `<RESULTS_ROOT>/pose` | Task output override |
| `DATA_ROOT` | Unset | Root override when running exactly one dataset |
| `SINTEL_ROOT` | `data/eval/sintel/training/final` | Sintel RGB root |
| `SINTEL_ANNO_ROOT` | `data/eval/sintel/training/camdata_left` | Sintel camera root |
| `SCANNET_ROOT` | `data/eval/scannetv2` | Processed ScanNet root |
| `TUM_ROOT` | `data/eval/tum` | Processed TUM root |
| `SEQ_LIST` | Unset | Optional debug subset; requires exactly one selected dataset |
| `SINTEL_SEQUENCES` | Frozen 14 | Sequence-list override; marks the aggregate as debug |
| `SCANNET_SEQUENCES` | Frozen 6 | Sequence-list override; marks the aggregate as debug |
| `TUM_SEQUENCES` | Frozen 8 | Sequence-list override; marks the aggregate as debug |
| `MAX_FRAMES` | Unset | Optional debug-only per-sequence frame cap |
| `STRIDE` | `1` | Fixed at `1`; other values are rejected by the frozen runner |
| `SIZE` | `518` | Fixed at `518`; other values are rejected by the frozen runner |
| `LOG_SELECTIONS` | `1` | Save cache-selection logs |
| `RESUME` | `0` | Set to `1` to replace selected sequences in existing metrics while retaining the rest |

After all selected cells finish, the wrapper writes
`<POSE_RESULTS_ROOT>/pose_results.csv`. Raw method directories follow
`<dataset>_reproduce_<method>/`. The CSV `run_scope` column is `frozen` for the
default protocol and `debug_subset` whenever `SEQ_LIST`, `MAX_FRAMES`, or any
dataset-specific sequence-list variable was explicitly set. Debug rows still
have to succeed completely and use identical coverage and software/hardware
provenance across methods, but they are not paper-table reproductions.

The defaults explicitly pass the frozen 14 Sintel, 6 ScanNet, and 8 TUM
sequence names; they do not scan and silently include additional prepared
scenes. Their required sequence/frame signatures are:

- Sintel: `alley_2=50`, `ambush_4=33`, `ambush_5=50`, `ambush_6=20`,
  `cave_2=50`, `cave_4=50`, `market_2=50`, `market_5=50`, `market_6=40`,
  `shaman_3=50`, `sleeping_1=50`, `sleeping_2=50`, `temple_2=50`, and
  `temple_3=50`;
- ScanNet: `scene0707_00` through `scene0710_00` have 90 frames,
  `scene0711_00` has 87, and `scene0712_00` has 90; and
- TUM: each of the eight
  `rgbd_dataset_freiburg3_{sitting,walking}_{halfsphere,rpy,static,xyz}`
  sequences has 90 frames.

The frozen collector also requires size 518, stride 1, no requested frame cap,
and an exact match between each `<dataset>_reproduce_<method>` directory and
the summary's public cache policy/window. Override a sequence list only when
intentionally producing a non-paper comparison.

### Multi-view reconstruction

```bash
bash scripts/reproduce/run_reconstruction.sh
```

The reconstruction suite covers static 7-Scenes, NRGBD, and ETH3D, plus TUM
Dynamics. Outputs include dataset-level `metrics.json`, trajectories,
reconstruction artifacts when enabled, and merged
`reconstruction_metrics.json` metadata.

The `paper` and `dense` protocols have different frame sampling. Compare
methods only within the same protocol. In particular, a cache budget can be
larger than a short reconstruction sample; in that case no pruning occurs
before the sample ends, which is valid and should be reported rather than
artificially extending the input.

Reconstruction controls are:

| Variable | Default | Meaning |
|---|---|---|
| `DATASETS` | `7scenes nrgbd eth3d tum` | Space-separated dataset keys |
| `RECON_RESULTS_ROOT` | `<RESULTS_ROOT>/mv_recon` | Task output override |
| `SEVEN_SCENES_ROOT` | `data/eval/7scenes` | 7-Scenes root |
| `NRGBD_ROOT` | `data/eval/neural_rgbd` | NRGBD root |
| `ETH3D_ROOT` | `data/eval/eth3d` | ETH3D root |
| `TUM_ROOT` | `data/eval/tum` | TUM Dynamics root |
| `PREFIX_FRAMES` | `4 6 8 10` | Static reconstruction prefix lengths |
| `TUM_PREFIX_FRAMES` | `10 20 30 40 50` | Dynamic reconstruction prefix lengths |
| `TUM_FRAMES` | `50` | Number of TUM Dynamics frames |
| `TUM_SAMPLING` | `first` | TUM frame sampling; `first` or `uniform` |
| `SEVEN_SCENES_KF_EVERY` | `50` | Dense 7-Scenes sampling stride |
| `NRGBD_KF_EVERY` | `100` | Dense NRGBD sampling stride |
| `MAX_SCENES`, `MAX_FRAMES` | Unset | Smoke/debug caps; not for a formal frozen run |
| `SEVEN_SCENES_SEQUENCES` | Frozen common 12 | Space-separated 7-Scenes sequence set |
| `TUM_SEQUENCES` | Frozen 8 TUM Dynamics sequences | TUM sequence-list override |
| `SAVE_ARTIFACTS` | `0` | Set to `1` to save large point-cloud/array artifacts |
| `LOG_SELECTIONS` | `1` | Save cache-selection logs |
| `USE_PROJ` | `0` | Fixed at direct point output; projected mode is rejected by the frozen runner |
| `SIZE`, `SEED` | `518`, `0` | Image size is fixed at `518`; dataset sampling seed defaults to `0` |

The wrapper automatically separates static `dense` runs from the TUM `paper`
run; it does not mix protocols in one aggregate. It writes
`reconstruction_static_results.csv` and `reconstruction_tum_results.csv` under
the reconstruction output root when the corresponding groups are selected.
Raw runs use `static/streamvggt_reproduce_<method>/` and
`tum/streamvggt_reproduce_<method>/`.

With no frozen-control override, each CSV row is labelled
`run_scope=frozen`. If any of `MAX_SCENES`, `MAX_FRAMES`, `PREFIX_FRAMES`,
`TUM_PREFIX_FRAMES`,
`TUM_FRAMES`, `TUM_SAMPLING`, `SEVEN_SCENES_SEQUENCES`, `TUM_SEQUENCES`,
`SEED`, `SEVEN_SCENES_KF_EVERY`, or `NRGBD_KF_EVERY` is explicitly set, the
wrapper passes `--allow-subset` and labels every row `run_scope=debug_subset`.
This happens even when an explicitly supplied value equals its default, so a
formal table is always tied to the unmodified entry point. Debug collection
still rejects failures, differing sequence/frame coverage across methods,
missing or mixed machine provenance, and mismatches between the
result-directory method and the recorded cache policy/window.

The frozen collector checks resolution 518, direct point output, no scene/frame
caps, seed 0, ICP threshold 0.1, the expected dense/paper protocol, sampling
stride, requested prefix list, and the following exact sequence/frame
signatures:

| Dataset | Frozen sequence/frame signature |
|---|---|
| 7-Scenes | `chess/seq-03` 20; `chess/seq-05` 8; `fire/seq-03` 20; `fire/seq-04` 20; `heads/seq-01` 20; `office/seq-02` 20; `pumpkin/seq-01` 20; `pumpkin/seq-07` 20; `redkitchen/seq-03` 20; `redkitchen/seq-04` 20; `stairs/seq-01` 10; `stairs/seq-04` 10 |
| NRGBD | `breakfast_room` 12; `complete_kitchen` 13; `green_room` 15; `grey_white_room` 15; `kitchen` 16; `morning_apartment` 10; `staircase` 12; `thin_geometry` 4; `whiteroom` 17 |
| ETH3D | `courtyard`, `delivery_area`, `electro`, `facade`, `kicker`, `meadow`, `office`, `pipes`, `playground`, `relief`, `relief_2`, `terrace`, and `terrains`: 10 frames each |
| TUM Dynamics | `rgbd_dataset_freiburg3_sitting_halfsphere`, `rgbd_dataset_freiburg3_sitting_rpy`, `rgbd_dataset_freiburg3_sitting_static`, `rgbd_dataset_freiburg3_sitting_xyz`, `rgbd_dataset_freiburg3_walking_halfsphere`, `rgbd_dataset_freiburg3_walking_rpy`, `rgbd_dataset_freiburg3_walking_static`, and `rgbd_dataset_freiburg3_walking_xyz`: 50 frames each |

The static runner intentionally evaluates the common 12 successful 7-Scenes
sequences for every method. It reproduces the comparable quality rows, not the
six historical ineligible one-frame failure records retained in supplementary
S12 for Full/K4/K6 provenance.

### Held-out long sequences

```bash
bash scripts/reproduce/run_long_sequence.sh
```

The frozen run evaluates 100, 250, 500, and 1000-frame prefixes of three raw
TUM RGB-D sequences. It uses lazy frame input, an output sink, and
`retain_outputs=False`/`retain_views=False` so input and dense predictions do
not grow with sequence length.

Full cache is expected to stop after its first out-of-memory result for a
sequence. That failure is a measured resource ceiling, not a reason for the
wrapper to discard completed results or abort all bounded methods. Do not use a
failed full-cache prefix for a pose-quality comparison.

Long-sequence controls are:

| Variable | Default | Meaning |
|---|---|---|
| `LONG_RESULTS_ROOT` | `<RESULTS_ROOT>/long_sequence` | Task output override |
| `LONG_DATA_ROOT` | `data/eval/stage4c_tum` | Raw held-out TUM root |
| `DATA_ROOT` | Unset | Alias for a long-sequence root override |
| `SEQUENCES` | The three frozen raw-TUM sequences | Space-separated sequence names |
| `LENGTHS` | `100 250 500 1000` | Evaluated prefix lengths |
| `MAX_FRAMES` | Unset | If set, evaluate only that prefix instead of `LENGTHS` |
| `SIZE` | `518` | Fixed at `518`; other values are rejected by the frozen runner |
| `MAX_ASSOC_DIFF` | `0.02` | Maximum RGB/pose timestamp difference in seconds |

The collected matrix is written to
`<LONG_RESULTS_ROOT>/long_sequence_results.csv`. The default matrix is labelled
`run_scope=frozen`; overriding `SEQUENCES`, `LENGTHS`, `MAX_FRAMES`, or
`MAX_ASSOC_DIFF` labels every produced row `debug_subset`.

### Supplementary assets

```bash
bash scripts/reproduce/build_supplementary.sh
```

This command performs no model inference. It deterministically rebuilds the
CSV tables, PDF figures, audits, and SHA256 manifests under `supplementary/`
from frozen result sources and archived trajectory/memory inputs.

Stage 4E-A fusion outputs are deliberately excluded from the final package.

Supplementary-builder controls are `OUTPUT_ROOT` (default `supplementary`),
`FIGURE_SOURCE_ARCHIVE` (default `stage4_supp_figure_sources.tar.gz`), and
`SKIP_FIGURES=1` for a CSV-only build.

The repository commits the already generated supplementary package, but not
all raw experiment-source CSV/tar archives. To reproduce the asset build
itself, first stage every file and matching SHA256 listed in
`supplementary/source_manifest.csv` at the expected repository-relative path.
The builder fails with a missing-source report when that author-side source
bundle is absent; a fresh clone is therefore sufficient to inspect and verify
the committed assets, but not to regenerate all of them from raw results.

### Selected or complete suite

```bash
bash scripts/reproduce/run_all.sh
```

The combined wrapper runs selected evaluation groups in sequence. A complete
run requires all datasets and substantial GPU/storage time. Prefer the
individual wrappers while validating a new installation. Set the
space-separated `TASKS` variable to choose groups rather than implicitly
running unavailable datasets. Its default tasks are `video_depth pose
reconstruction long_sequence`; supplementary generation is opt-in. Dataset
sets can be overridden independently with `VIDEO_DEPTH_DATASETS`,
`POSE_DATASETS`, and `RECON_DATASETS`.

## Method/configuration mapping

At the model API level, use the cache window matching the public policy:

| Method | `cache_window_size` | `cache_policy` |
|---|---:|---|
| Full cache | `None` | Not used |
| K4 | `4` | `anchor_recent_dino_diverse_k4` |
| K6 | `6` | `anchor_recent_dino_diverse_k6` |
| K8 | `8` | `anchor_recent_dino_diverse_k8` |

Passing a canonical policy with the wrong cache window raises an error. This
is intentional: it prevents an output directory labelled K4/K6/K8 from using a
different budget.

For ordinary short-sequence inference, the model retains outputs by default.
For an end-to-end bounded long-sequence process, provide an `output_sink`, set
`retain_outputs=False` and `retain_views=False`, and stream frames rather than
preloading the complete sequence. Bounding only the KV cache does not release
other application-owned tensors.

## Frozen protocols

| Evaluation group | Datasets | Frozen unit/protocol |
|---|---|---|
| VideoDepth | Bonn, KITTI, Sintel | Official valid-pixel depth aggregation plus sequence-level paired analysis |
| Pose | Sintel, ScanNet, TUM | Sequence-equal ATE and translation/rotation RPE |
| Static reconstruction | 7-Scenes, NRGBD, ETH3D | Dense reconstruction protocol |
| Dynamic reconstruction | TUM Dynamics | First 50 aligned frames, paper protocol |
| Long streaming | Three raw TUM sequences | Prefixes of 100, 250, 500, and 1000 frames |

The evaluators use one GPU process. Reconstruction uses seed `0`. Do not change
frame sampling, resolution, dataset coverage, ICP settings, or valid-pixel
masks and still label the output as a reproduction of the frozen table.

## Outputs and provenance

Keep raw result directories until the aggregate tables and manifests have been
built. The key machine-readable products are:

- VideoDepth prediction directories and `result_scale.json` outputs;
- VideoDepth `video_depth_results.csv`;
- pose `pose_metrics.json`, trajectory NPZ files, and `pose_results.csv`;
- reconstruction `metrics.json`, `reconstruction_metrics.json`, and collected
  static/TUM CSV files;
- long-sequence `stage4c_metrics.json`, trajectory NPZ, memory traces, and
  `long_sequence_results.csv`; and
- supplementary `source_manifest.csv` and `asset_manifest.csv`.

Each formal run should retain:

- canonical method name and cache window;
- checkpoint path or hash;
- dataset/sequence and number of processed frames;
- GPU name, PyTorch/CUDA/Python versions, and job identifier when available;
- inference time and allocated/reserved CUDA peaks; and
- explicit success, metric failure, or OOM status.

Generated results should not be committed wholesale to Git. Commit compact
tables, figures, manifests, and the minimum raw trajectory/memory sources
needed to verify the published claims.

## Comparing with the frozen results

Use the tables in `supplementary/tables/` as the machine-readable reference.
Important aggregation rules are recorded in
`supplementary/README.md`:

- VideoDepth official aggregates are valid-pixel weighted; sequence-level
  statistics are reported separately.
- Pose and reconstruction means are sequence-equal means.
- 7-Scenes quality comparison uses the common successful sequence set; K8's
  shorter failure record must not be interpreted as better robustness.
- Runtime and memory comparisons must use the same hardware/software run.

Small floating-point differences can occur across GPU and library versions.
Coverage, method/window metadata, qualitative decisions, and large resource
trends should remain consistent. If a result differs materially, first audit
dataset layout, sampling, checkpoint, method name, and active interpreter.

## Scheduler template

Keep cluster-specific setup outside the repository's stable wrappers:

```bash
#!/usr/bin/env bash
# Add scheduler directives required by your site above this line.
set -euo pipefail

# Initialize modules/Conda using the commands approved for your cluster.
conda activate StreamVGGT

cd /path/to/eStreamVGGT
bash scripts/reproduce/run_video_depth.sh
```

The repository does not publish a universal `SBATCH` file because partitions,
accounts, module stacks, filesystems, and Conda initialization are site
specific.

## Claim boundaries

The frozen evidence supports bounded execution through 1000 frames on the
tested inputs and a strong K4 quality-memory-runtime trade-off. It does not
support the following stronger claims:

- constant pose error as sequence length increases;
- universal accuracy superiority over full cache;
- K8 as a universal replacement for K4;
- a validated K4/K8 output-level fusion method; or
- hardware-independent absolute runtime or memory values.
