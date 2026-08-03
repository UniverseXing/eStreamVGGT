# Normalised Regret and Oracle Wins

This note defines exactly how the values in
`tables/table_s14_cross_task_regret.csv` were calculated.

## Comparison scope

The oracle is restricted to the three bounded methods $\mathcal{M}=\{\mathrm{K4},
\mathrm{K6},\mathrm{K8}\}$. Full cache is an unbounded reference and is not
eligible for a bounded oracle win. All methods compared within one row use the
same evaluation units and metric definition.

## Unit-level normalised regret

Let $x_{m,u}$ be the value obtained by method $m$ on evaluation unit $u$. For a
metric where lower values are better, the bounded oracle and regret are

\begin{equation}
o_u=\min_{m\in\mathcal{M}}x_{m,u},\qquad
r_{m,u}=\frac{x_{m,u}-o_u}{\max(|o_u|,10^{-12})}.
\end{equation}

For a metric where higher values are better, they are

\begin{equation}
o_u=\max_{m\in\mathcal{M}}x_{m,u},\qquad
r_{m,u}=\frac{o_u-x_{m,u}}{\max(|o_u|,10^{-12})}.
\end{equation}

The denominator makes regret dimensionless and comparable across metrics. The
$10^{-12}$ floor only prevents division by zero. For each reported group,
`mean_normalized_regret`, `median_normalized_regret`, and
`max_normalized_regret` are the corresponding statistics over its evaluation
units.

## Oracle wins

A method receives one oracle win on unit $u$ when its value equals $o_u$ under
`math.isclose` with relative tolerance $10^{-8}$ and absolute tolerance
$10^{-10}$. Tied methods each receive a win, so the total number of wins can
exceed the number of units. The `oracle_wins` column is the sum of these
unit-level indicators.

## Evaluation units and metrics

| Task | Dataset coverage | Unit used in dataset rows | Primary metric | Secondary metric |
|---|---|---|---|---|
| VideoDepth | Bonn, KITTI, Sintel | sequence | AbsRel (lower) | $\delta_1$ (higher) |
| Camera pose | ScanNet, Sintel, TUM | dataset aggregate | ATE (lower) | rotation RPE in degrees (lower) |
| Static reconstruction | 7-Scenes, NRGBD, ETH3D | successful sequence | overall error (lower) | normal consistency (higher) |
| Dynamic reconstruction | TUM Dynamics | sequence | overall error (lower) | normal consistency (higher) |

Rows with `dataset=all` pool the comparable units across datasets within the
same task. The `cross_task_macro` rows are different: they use only the primary
metric from each of the ten task--dataset benchmark cells (three VideoDepth,
three pose, three static reconstruction, and one dynamic reconstruction cell).
The oracle is recomputed independently in each cell, after which the ten
normalised regrets are summarised. This produces the paper's bounded primary
oracle-win counts of 7 for K4, 1 for K6, and 2 for K8.

The aggregate values supplied to this calculation are listed in
`tables/table_s13_cross_task_summary.csv`; the complete regret output is in
`tables/table_s14_cross_task_regret.csv`.
