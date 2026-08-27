# Supplementary material work package

This directory is the curated package requested by `核心实验与补充材料.xlsx`.
It is intentionally separate from the older generated `supplementary/`
directory. Existing assets are copied, not removed, because the old paths are
still used by repository tests and paper scripts.

## Reading order

1. `MATERIAL_STATUS.csv` maps every P0/P1 worksheet row to its current status,
   asset paths, and remaining server action.
2. `tables/` contains full-precision CSV data.
3. `figures/` contains existing or derived visual evidence.
4. `methods/` records calculations and exact selector pseudocode.
5. `EXPERIMENT_PLAN.md` gives the remaining commands and stopping rules.

## Current headline status

- P0 model inference is complete. The only outstanding P0 action is to rerun
  the Stage 5A **summarizer**, not the model, on the server-side raw result
  directories. This exposes the already stored SqRel, log-RMSE, delta-2 and
  delta-3 values and adds K4-vs-Full paired bootstrap rows.
- P1 cache-selection visualisation and measured selector latency are grouped
  into one small diagnostic run.
- P1 K=8 controls and the broad budget sweep remain optional. They must not be
  used to retune the frozen K4/K6/K8 method after seeing held-out results.

Rebuild the curated package without inference:

```bash
python scripts/build_requested_supplementary_material.py
```

The builder overwrites only its known derived/copied assets and does not delete
unrecognised files.
