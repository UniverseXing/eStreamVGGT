# Stage 5: conference evidence-completion plan

更新日期：2026-08-24

## 1. 新目标

Stage 5 不再承担完整期刊补证据任务，而只为 8 页会议论文补两个最关键、最能直接
回答审稿疑问的实验。原来的同预算全矩阵、K6/K8 component、外部竞争方法和 KITTI
机制分析全部顺延至 `STAGE6_JOURNAL_EXPERIMENT_PLAN.md`。

正式 K4 配置继续冻结为 `anchor_recent_dino_diverse_k4`。不得根据 Stage 5 结果修改
K、DINO descriptor、候选范围或数据划分。已有 1000-frame bounded completion 与
Full cache 约 195 帧 OOM 的 Stage 4C 结果必须保留，它是会议论文最直观的长序列
selling point，不需要重跑。

Stage 5 只包含：

1. **Stage 5A：same-budget baseline**，回答 DINO selection 是否有实际意义；
2. **Stage 5B：memory decomposition**，把 KV pruning 与 output release 的贡献拆开。

统一使用 RTX 6000 Ada、同一 checkpoint、$518$ 输入和既有数据顺序。集群仍只提交
根目录 `run.sh`。

## 2. Stage 5A：same-budget baseline

### 2.1 核心矩阵

| 会议表中名称 | 代码策略 | K4 在线布局 | 优先级 |
|---|---|---|---|
| Full | `full_cache` | 全历史；原版 StreamVGGT | 必跑参照 |
| Recent-4 | `fifo` | 最近 3 个历史状态 + current | 必跑 |
| Anchor+Recent-4 | `anchor_recent` | frame 0 + 最近 2 个历史状态 + current | 必跑 |
| K4 | `anchor_recent_dino_diverse_k4` | frame 0 + 2 个 DINO 历史状态 + current | 必跑主方法 |
| Uniform-4 | `anchor_uniform_k4` | frame 0 + 2 个 deterministic 历史状态 + current | 时间允许时追加 |
| Random-4 | `random_reservoir_k4` | 3 个 seeded reservoir 历史状态 + current | 时间允许时追加，seed 0/1/2 |

`Anchor+Recent-4` 是最关键的同槽位非 DINO 对照：它与 K4 都保留 anchor、两个历史
槽和 current，只把两个历史槽的选择从最近帧改为 DINO diversity。因此二者的配对
差异能最直接回答 DINO selection 是否有意义。Full 不是同预算方法，只用于标出原版
StreamVGGT 的质量和资源参照。

### 2.2 数据与输出

运行既有三域 VideoDepth：Bonn 5、Sintel 23、KITTI 13。只报告会议正文真正需要的：

Full 与 K4 直接复用已经在同一 RTX 6000 Ada 协议下完成的 Stage 4A/4B 聚合与逐序列
数据；Stage 5A 默认只新跑 Recent-4 和 Anchor+Recent-4。这样最终表仍包含四组，
但不会浪费时间重复 Full/K4。只有设置 `STREAMVGGT_STAGE5A_RERUN_REFERENCES=1`
时才重新运行两组参照。

- AbsRel、RMSE、$\delta_1$；
- FPS、peak CUDA allocated/reserved；
- K4 相对 Recent-4、Anchor+Recent-4 的逐序列 wins/ties/losses 和 10,000 次 paired
  bootstrap 95% CI；
- Optional 运行时，Random-4 先对同一序列的三个 seed 取均值，再进入 paired test。

正式文件：

```text
stage5a_same_budget_results.csv
stage5a_same_budget_sequence_results.csv
stage5a_paired_statistics.csv
```

### 2.3 决策门槛

- 若 K4 相对 Anchor+Recent-4 在至少两个域 AbsRel 方向更好，且其中至少一个域的
  paired CI 排除零，可写“DINO selection provides measurable same-budget benefit”。
- 若改善只在 KITTI 成立，只写“the benefit is most evident under outdoor domain
  shift”，不得写普遍优势。
- 若三域均无清晰改善，DINO 从会议论文核心创新降为低成本选择策略；主贡献改为
  bounded coupled-state execution 与 long-sequence completion，不再追加调参实验。
- 无论结果如何，都不修改正式 K4。

## 3. Stage 5B：memory decomposition

### 3.1 四格设计

在同一条 Bonn `person_tracking2` 110 帧序列、同一作业中运行：

| Cell | KV 策略 | 输入方式 | 稠密输出生命周期 |
|---|---|---|---|
| Full + accumulated outputs | Full | 逐帧 lazy load | GPU 输出持续累积 |
| Full + streaming release | Full | 逐帧 lazy load | 预测后立即写入 CPU sink 并释放 |
| K4 + accumulated outputs | K4 | 逐帧 lazy load | GPU 输出持续累积 |
| K4 + streaming release | K4 | 逐帧 lazy load | 预测后立即写入 CPU sink 并释放 |

四组都使用逐帧 lazy load，避免把“预加载全部输入”的差异误算成 output release。
因此：

- 固定 lifecycle 比较 Full 与 K4，得到 **KV pruning contribution**；
- 固定 KV 比较 accumulated 与 release，得到 **output release contribution**；
- Full 的 accumulated/release 预测 hash 必须相同，K4 的两组 hash 也必须相同。

逐帧保存 aggregator KV、camera KV、descriptor、input、retained outputs、CUDA
allocated/reserved；汇总 peak allocated/reserved、RSS、FPS 和 pose 指标。正式产物：

```text
stage5b_memory_decomposition.csv
stage5b_memory_trace.csv
stage5b_memory_contributions.csv
paper_assets/figures/fig_stage5b_memory_decomposition.pdf
paper_assets/figures/fig_stage5b_memory_decomposition.svg
```

图左展示四条逐帧 CUDA allocated 曲线，图右分别展示在 streaming-release 口径下的
KV pruning 节省，以及在 K4 口径下的 output release 节省。

### 3.2 完成与失败门槛

- 四组都应在 110 帧完成；任一组非预期 OOM 时保留错误记录，但不能把不完整曲线
  当作严格 factorial 结果。
- 同一 KV 策略的 accumulated/release pose 和 depth SHA256 必须一致；不一致先修
  执行路径，不能进入论文。
- 只有四组输入均为 `streaming`，才允许把差异归因于 KV pruning/output release。
- 正文保留 Stage 4C 的 1000-frame completion/OOM scaling 图；Stage 5B 图解释其
  内存机制，两张图回答不同问题，不能互相替代。

## 4. 执行顺序

本地先做策略 smoke，不使用 Slurm：

```bash
python scripts/reproduce/smoke_inference.py \
    --repo-root . \
    --weights ckpt/checkpoints.pth \
    --images-dir examples/example_building \
    --method anchor_recent_dino_diverse_k4 \
    --max-frames 5
```

会议版核心实验：

```bash
sbatch run.sh
```

若要拆分作业：

```bash
STREAMVGGT_STAGE5_PARTS=same_budget sbatch run.sh
STREAMVGGT_STAGE5_PARTS=memory sbatch run.sh
STREAMVGGT_STAGE5_PARTS=finalize sbatch run.sh
```

只有时间允许时才追加 Random-4/Uniform-4：

```bash
STREAMVGGT_STAGE5_PARTS=same_budget \
STREAMVGGT_STAGE5A_INCLUDE_OPTIONAL=1 \
sbatch run.sh

STREAMVGGT_STAGE5_PARTS=finalize \
STREAMVGGT_STAGE5A_INCLUDE_OPTIONAL=1 \
sbatch run.sh
```

## 5. Stage 5 停止规则

- 不在会议截止前运行 Stage 6 的 TUM pose、TUM Dynamics、K6 no-recent、K8
  no-hierarchy、外部 OVGGT/FrameVGGT/STAC 或 KITTI 机制相关分析；
- 不增加新数据集、新 K 或新 selector；
- 不删除或弱化已有 1000-frame completion 与 Full OOM 结果；
- 两个实验完成、汇总和作图通过后，立即回到 8 页会议论文压缩与写作。
