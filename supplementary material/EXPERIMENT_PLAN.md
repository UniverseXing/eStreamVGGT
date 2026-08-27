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

- hierarchical K8: `anchor_recent_dino_diverse_k8`;
- Recent-8: `fifo`, window 8;
- non-hierarchical DINO-8: legacy generic
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

### P1-C: budget sweep (deferred)

The requested K=2/4/6/8/12/16/Full sweep is not a clean one-dimensional ablation
because the frozen K4, K6, and K8 use different slot layouts. Mixing those
layouts into one curve would confound budget and policy. A defensible sweep
requires one scalable selector rule used unchanged at every K, shown separately
from the frozen K4/K6/K8 curve. This is lower priority than the P0 evidence and
P1 selector diagnostic and should be omitted rather than presented as a
misleading continuous-budget result if time is limited.
