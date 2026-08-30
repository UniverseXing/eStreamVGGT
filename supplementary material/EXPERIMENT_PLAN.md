# Remaining supplementary work

## Priority order

### P0-A: refresh the same-budget CSVs (required, no model inference)

The existing Stage 5A raw directories already contain `result_scale.json`,
`result_scale_sequences.json`, and `runtime_memory_rank0.json`. The old
summarizer discarded four secondary metrics and did not test K4 against Full.
The updated summarizer now exports all seven depth metrics and records the
10,000-draw bootstrap seed.

Run only the finalize path on the server:

```bash
STREAMVGGT_RUN_TARGET=supplementary \
STREAMVGGT_SUPPLEMENTARY_PARTS=refresh_stage5a \
sbatch run.sh
```

This must not launch any `eval/video_depth/launch.py` process. Expected refreshed
files in the repository root:

```text
stage5a_same_budget_results.csv
stage5a_same_budget_sequence_results.csv
stage5a_paired_statistics.csv
```

Download these three files and rebuild the curated folder locally. P0-A passes
only if the aggregate and sequence CSVs contain
`abs_rel,sq_rel,rmse,log_rmse,delta_1,delta_2,delta_3`, all eight Stage 5A method
rows per dataset, and the paired CSV contains K4-vs-Full, K4-vs-Recent-4,
K4-vs-Anchor+Recent-4, K4-vs-Uniform-4, and K4-vs-three-seed-Random-4 for all
three datasets.

### P1-A: selector trace, thumbnails, cosine scores, and overhead (recommended)

Use one fixed 110-view object-centred sequence, 7-Scenes `chess/seq-01`, sampled
at source stride 5 (`000000, 000005, ..., 000545`). Run K4/K6/K8 with identical
inputs. The diagnostic records retained IDs, evicted IDs, the score used for
selection, descriptor/KV bytes, selector CUDA time, and total frame time. It
then creates the true cache timeline and checkpoint thumbnail panels.

```bash
STREAMVGGT_RUN_TARGET=supplementary \
STREAMVGGT_SUPPLEMENTARY_PARTS=selector_trace \
sbatch run.sh
```

The diagnostic is descriptive. It must not change selector decisions or be
used to tune K, similarity thresholds, or K8 bank boundaries. It passes when
all three methods process exactly 110 views, retain no more than 4/6/8 states,
and the output contains finite timing and descriptor-size records.

### P1-B: K8 matched controls (optional after P0/P1-A)

The existing K4 study already includes Recent-4, Anchor+Recent-4, Uniform-4,
three Random-4 seeds, and K4. The remaining clean P1 component comparison is
fixed K=8:

- Hierarchical K8: `anchor_recent_dino_diverse_k8`;
- Recent-8: `fifo`, window 8;
- Non-hierarchical DINO-8: legacy generic
  `anchor_recent_dino_diverse`, window 8.

Run these on the three VideoDepth datasets only if schedule permits. Report all
results regardless of outcome and do not alter the final K8 configuration.

```bash
STREAMVGGT_RUN_TARGET=supplementary \
STREAMVGGT_SUPPLEMENTARY_PARTS=k8_controls \
sbatch run.sh
```

This launches only the two missing controls; the frozen K8 rows are reused from
Stage 4A/4B and checked for the same RTX 6000 Ada provenance.

The matched VideoDepth table is not a basis for claiming that Hierarchical K8
is the depth-optimal selector. Non-hierarchical DINO-8 is numerically stronger
on Bonn and Sintel, and its paired AbsRel advantage on KITTI is clear. The
hierarchical configuration remains a pre-frozen temporal/pose specialist.

### P1-C: matched K8 temporal-coverage diagnostic

The existing K8 trace demonstrates the hierarchical policy in isolation, but
does not compare its temporal coverage with Non-hierarchical DINO-8. Run both
selectors on the same fixed 110-view `7scenes/chess/seq-01` input:

```bash
STREAMVGGT_RUN_TARGET=supplementary \
STREAMVGGT_SUPPLEMENTARY_PARTS=k8_coverage \
sbatch run.sh
```

Both methods use the same checkpoint, ordered images, preprocessing and
eight-state budget. Frame 0 is treated as the separate anchor. For sampled
views 50--110, the four non-anchor age bins are recent (0--3), near (4--15),
middle (16--47), and long ($\geq48$). The pre-registered claim gates are:

- use *guarantees all four temporal bins* only if Hierarchical K8 has 100%
  all-four coverage over all 61 measured steps;
- otherwise use *more consistent multi-scale temporal coverage* only if its
  all-four coverage exceeds Non-hierarchical DINO-8 by at least 0.20 and its
  mean occupied-bin count exceeds the control by at least 0.25;
- if either comparative threshold fails, report the trace descriptively and
  make no comparative coverage claim;
- no boundary, descriptor rule, start frame or K value may be changed after
  seeing the matched output.

The run writes per-step, summary and gate CSVs, both raw selector traces, and a
timeline/coverage figure under `eval_results/supplementary_k8_coverage/`.
The current isolated hierarchical trace covers all four bins in 58/61 measured
steps (95.1%), with three middle-bin transition gaps at views 82--84. Therefore
the strict word *guarantees* is already disallowed unless a corrected future
method is evaluated under a new protocol. The matched experiment can support
only a weaker comparative coverage claim, and does not establish that coverage
causes better depth, pose, or reconstruction accuracy.

### P1-D: matched 1000-frame K8 TUM pose comparison

The temporal-coverage diagnostic measures selector behaviour, not downstream
geometry. Connect the coverage result to long-sequence pose using the frozen
Stage 4C raw-TUM protocol. Run Hierarchical K8 and Non-hierarchical DINO-8 on
the same three sequences at exactly 1000 frames, with the same checkpoint,
RGB/ground-truth association tolerance, resolution, streaming-release path and
RTX 6000 Ada environment:

```bash
STREAMVGGT_RUN_TARGET=supplementary \
STREAMVGGT_SUPPLEMENTARY_PARTS=k8_pose \
sbatch run.sh
```

This is a six-cell experiment: two selectors by three sequences. It reports
ATE, translation RPE and rotation RPE. The claim gates are fixed before the
run:

- *overall 1000-frame pose superiority* requires lower macro mean for all
  three metrics and Hierarchical K8 wins on at least two of three sequences for
  every metric;
- *1000-frame rotation-pose specialist* requires at least 10% lower macro
  rotation RPE and wins on at least two sequences, while macro ATE and
  translation RPE regress by no more than 20% and no per-sequence ATE exceeds
  twice the matched control;
- if neither gate passes, retain only the selector-level temporal-coverage
  result and make no downstream pose-advantage claim.

Do not change the K8 age bins or rerun only favourable sequences after seeing
the comparison. The experiment tests association between the frozen selector
and pose performance; it does not by itself establish causal mediation by
temporal coverage.

Expected outputs are under `eval_results/supplementary_k8_pose/`:

```text
k8_pose_results.csv
k8_pose_comparison.csv
k8_pose_summary.csv
k8_pose_gate.csv
figure_k8_pose_comparison.pdf
figure_k8_pose_comparison.png
k8_pose_metadata.json
```

### P1-E: budget sweep (deferred)

The requested K=2/4/6/8/12/16/Full sweep is not a clean one-dimensional ablation
because the frozen K4, K6, and K8 use different slot layouts. Mixing those
layouts into one curve would confound budget and policy. A defensible sweep
requires one scalable selector rule used unchanged at every K, shown separately
from the frozen K4/K6/K8 curve. This is lower priority than the P0 evidence and
P1 selector diagnostic and should be omitted rather than presented as a
misleading continuous-budget result if time is limited.
