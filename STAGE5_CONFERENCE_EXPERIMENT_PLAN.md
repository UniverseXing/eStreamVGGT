# Stage 5: conference evidence-completion plan

更新日期：2026-08-25

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
paper_assets/figures/fig_stage5b_memory_decomposition.png
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

## 6. 最终结果与决策（已冻结）

Stage 5A 通过预设门槛：K4 相对 Anchor+Recent-4 在 Bonn、Sintel、KITTI 三域
的 aggregate AbsRel 分别降低 21.6\%、6.3\%、28.2\%，逐序列 paired CI 分别为
[0.0115, 0.0294]、[0.0030, 0.0358]、[0.0341, 0.0603]，全部排除零。正文允许写
“DINO selection provides measurable same-budget benefit across all three domains”。
Uniform-4/Random-4 只进入补充材料；由于 KITTI 上 Random-4 与 K4 无清晰差异，
不得写 K4 对所有 selector 普遍最优。

Stage 5B 的四个单元均完成 110 帧，输入均为 streaming，且同一 KV 策略的两种
output lifecycle 预测 hashes 一致。固定 streaming release 时，K4 相对 Full
节省 11058.3 MiB peak allocated（57.9\%）与 35812 MiB peak reserved
（80.3\%），并获得 $3.24\times$ FPS；固定 K4 时 output release 额外节省
492.0 MiB（5.8\%）。正文结论冻结为“KV pruning 是主要显存贡献，output release
是独立但较小的长度相关贡献”。

Stage 5 总门槛已通过，停止追加 selector、数据集或调参实验。下一步只进行主文
表图更新与 8 页会议稿压缩；期刊扩展继续留在 Stage 6。

## 7. Stage 5E：紧急直接 SOTA 对比（2026-08-28 修订）

### 7.1 对手与实验边界

由于 STAC 的独立 CUDA/PyTorch 环境超出服务器剩余磁盘预算，直接竞争方法改为已被
ECCV 2026 接收的 **OVGGT**（官方仓库 `https://github.com/VAISR/OVGGT`）。OVGGT
和本文方法均为训练免费的 StreamVGGT 缓存管理方法，双方读取完全相同的
`ckpt/checkpoints.pth`；不下载或复制第二份 checkpoint。

紧急核心矩阵固定为：

| 方法 | Backbone | 数据 | 指标 |
|---|---|---|---|
| K4 | StreamVGGT | Bonn 5 + Sintel 23 + KITTI 13 | AbsRel、SqRel、RMSE、LogRMSE、$\delta_{1,2,3}$、FPS、peak allocated/reserved |
| OVGGT | 同一 StreamVGGT checkpoint | 完全相同序列与帧 | 同上 |

OVGGT 不调参，直接使用官方 `OVGGT()` 与 `model.inference()` 默认设置：总 token
budget 200000、camera budget 384、coverage history-anchor strategy。该配置不是与 K4
相同的四帧预算，因此正文应称“direct competing bounded method”，不能称
“same-budget comparison”；同预算 selector 结论仍由 Stage 5A 提供。

### 7.2 公平性与决策门槛

正式推理前在 Bonn `balloon2` 前 10 帧做实现 parity：本项目 Full 与 OVGGT 代码在
关闭 eviction、保留完整因果缓存时使用同一 checkpoint、输入、depth evaluator 和 scale alignment，AbsRel
绝对差不得超过 `max(本项目 Full 的 2%, 0.002)`。不通过时只排查 preprocessing、
checkpoint、dtype 或实现差异，不能发布 K4/OVGGT 排名。

必须完整报告七个质量指标、FPS 和两种峰值显存；AbsRel 给出逐序列 10,000 次
paired bootstrap、95% CI 和 wins/ties/losses。有效性门槛为 parity 通过、双方覆盖
完全一致、OVGGT 零失败且同在 RTX 6000 Ada 上运行。“至少一个维度 K4 更好”只
决定可以陈述哪些具体优势，不能据此宣称全面优于 SOTA。

### 7.3 轻量安装

OVGGT 代码仅约 24 MB。直接放在本项目 `external/` 下并复用当前 StreamVGGT 环境；
Stage 5E adapter 直接读取本项目数据与 checkpoint，不需要数据或权重软链接。

```bash
git clone --depth 1 https://github.com/VAISR/OVGGT.git external/OVGGT
git -C external/OVGGT checkout b582391f3dc6ec734aaa3a8fde3b4baadaf7800a

conda run -n StreamVGGT python -c \
"import torch, numpy, cv2, scipy, einops, accelerate, transformers, roma, evo; print('OVGGT dependencies: OK')"
```

服务器使用且只保留 `opencv-python-headless==4.10.0.84`，不安装 OVGGT 的完整
`requirements.txt`，也不重装 torch、open3d 或 gsplat。

### 7.4 集群运行

一次完成 parity、三数据集正式推理与汇总：

```bash
STREAMVGGT_RUN_TARGET=stage5e \
STREAMVGGT_STAGE5E_OVGGT_ROOT="$(pwd)/external/OVGGT" \
sbatch --mem=32G --time=08:00:00 run.sh
```

如需拆分，依次执行：

```bash
STREAMVGGT_RUN_TARGET=stage5e STREAMVGGT_STAGE5E_PARTS=parity \
STREAMVGGT_STAGE5E_OVGGT_ROOT="$(pwd)/external/OVGGT" \
sbatch --mem=32G --time=01:00:00 run.sh

STREAMVGGT_RUN_TARGET=stage5e STREAMVGGT_STAGE5E_PARTS=inference \
STREAMVGGT_STAGE5E_OVGGT_ROOT="$(pwd)/external/OVGGT" \
sbatch --mem=32G --time=08:00:00 run.sh

STREAMVGGT_RUN_TARGET=stage5e STREAMVGGT_STAGE5E_PARTS=finalize \
STREAMVGGT_STAGE5E_OVGGT_ROOT="$(pwd)/external/OVGGT" \
sbatch --mem=32G --time=01:00:00 run.sh
```

正式下载回本地分析的文件为：

```text
stage5e_results.csv
stage5e_sequence_results.csv
stage5e_comparison.csv
stage5e_paired_statistics.csv
stage5e_gate.csv
```
