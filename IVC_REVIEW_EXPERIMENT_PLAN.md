# Stage 5: IVC review evidence-completion plan

更新日期：2026-08-19

## 1. 目标与范围

本文档只处理 `IVC review.pdf` 结尾归纳的三项实验任务：

1. same-budget baselines 与 component ablations；
2. 复现 2--3 个直接竞争的 StreamVGGT bounded-cache 方法；
3. qualitative/selected-frame 分析，并解释 KITTI 上 K4 优于 Full cache 的现象。

Stage 5 是证据补全，不是新一轮方法搜索。K4、K6、K8 的正式配置继续冻结，
不得根据新增结果修改 DINO 阈值、K、temporal-bin 边界、数据抽样或指标定义。
新增控制组只用于检验已有主张；不通过时应降低论文主张，而不是继续调控制组。

统一执行规则：

- 仍使用 RTX 6000 Ada；集群入口只修改根目录 `run.sh`，不再新增硬件命名的
  Slurm 脚本。
- 使用同一 StreamVGGT checkpoint、$518$ 输入、相同序列与帧顺序。
- 所有方法必须逐序列输出标准指标、FPS、CUDA peak allocated/reserved、状态和
  错误信息；随机方法额外保存随机种子。
- 主分析使用标准指标和 paired sequence statistics；gate/oracle/regret 只作辅助。
- 不把不兼容的软件栈、不同 checkpoint 或不同输入协议合并成一张无条件排名表。

## 2. 现有证据审计

| Review 要求 | 已有证据 | 原始文件 | 当前判断 |
|---|---|---|---|
| 同预算非 DINO 对照 | K6 已在 Sintel/ScanNet/TUM pose 上比较 DINO、Uniform 和 FIFO；静态/动态 reconstruction 也已有 DINO K6 与 Uniform K6 | `stage3_3_pose_results.csv`、`refine_stage3_3b_recon_results.csv`、`stage3_3c_recon_results.csv`、`supplementary/tables/table_s08_pose_policy_ablation.csv` | **部分完成**：能支持 K6，但不能证明主方法 K4 的 DINO 贡献 |
| K4 近期窗口对照 | `fifo_k4`、`old_k4`、K4 和 Full 已在 Bonn `person_tracking2` 110 帧诊断 | `stage3_5a_results.csv`、`stage3_5a_sequence_results.csv`、`stage3_5c1_results.csv` | **部分完成**：只有一条开发序列，不能进入跨域主表 |
| temporal hierarchy 消融 | `standard_dino_k8` 与 `temporal_binned_dino_k8` 同为 K8，在 Bonn 110 帧比较 | `stage3_5c2_results.csv`、`stage3_5c2_sequence_results.csv` | **完成诊断、未完成跨域验证** |
| DINO/anchor/recent component | 旧实验改变过 DINO 历史槽和 recent 槽，但没有一张固定 K4、一次只替换一个 component 的主表；既有 claim audit 也将 K4/K8 的 DINO 因果归因标为 `LIMITED` | Stage 3.5A--C、`stage4b_claim_audit.csv` | **缺失** |
| KV pruning 与 output release 分离 | 已有逐帧内存来源 trace；K8 legacy-retain 与 stream-release 在 Bonn 110 帧预测完全一致；Full+release 和 K4+release 已在三条 TUM 长序列运行 | `stage3_memory_trace_bonn_balloon2.csv`、`stage3_6b_results.csv`、`stage4c_results.csv` | **部分完成**：没有同一序列、同一作业下的 Full/K4 $\times$ retain/release 四格实验 |
| 直接竞争方法 | 没有 XStreamVGGT、OVGGT、FrameVGGT、STAC 等运行记录 | 全工作区检索无结果 | **缺失** |
| retained-frame 可视化 | Stage 4D 曾生成 cache timeline 和 trajectory，manifest 仍在；现有主图还包含 VideoDepth、long scaling 和 pose case | `stage4d_asset_manifest.csv`、`stage4d_case_audit.csv`、`paper_assets/figures/` | **部分完成**：没有 RGB retained-frame grid，也没有 Full/Recent/K4 depth-error 对照 |
| KITTI K4 机制 | 已有 13 条序列的聚合、逐序列和 paired bootstrap | `stage4a_video_depth_results(1).csv`、`stage4b_video_depth_sequence_results.csv`、`stage4b_video_depth_paired_comparison.csv` | **结果完成、机制缺失**：没有逐帧误差、DINO 冗余量和 selection log |

### 2.1 已有数据能够直接写入补充材料的结论

1. **K6 的 DINO 证据不是单向全胜。** 在 pose 上，DINO K6 的 ATE 相对
   Uniform K6 在 Sintel 为 $0.3939$ vs $0.4087$、TUM 为 $0.0279$ vs
   $0.0335$，但 ScanNet 为 $0.1381$ vs $0.1352$。这说明 DINO 作用具有任务/场景
   依赖性，不能写成普遍优势。
2. **K6 在 reconstruction 上的证据较一致。** TUM Dynamics Overall 为 DINO
   K6 $0.0754$、Uniform K6 $0.0838$；静态 reconstruction 的 macro Overall 为
   $0.2962$ vs $0.3018$（均越低越好）。这些结果可保留为跨任务旁证。
3. **FIFO K4 展示了明显的质量取舍。** Bonn `person_tracking2` 上 FIFO K4 将
   rotation RPE 从 K4 的 $41.67^\circ$ 降至 $4.72^\circ$，但 AbsRel 从约
   $0.0452$ 恶化到 $0.1402$。它是有价值的失败对照，但单序列不足以替代 Stage 5A。
4. **temporal hierarchy 已有一次同 K8 诊断。** 分段 K8 相对 standard K8 将
   AbsRel 从 $0.0552$ 降到 $0.0515$，并改善局部 RPE，但两者均未修复全局 ATE。
5. **release 路径已有正确性证据。** Bonn 110 帧上 K8 legacy-retain 与
   stream-release 的 ATE、RPE 和 depth 完全一致；peak allocated 从约
   $9522$ MiB 降到 $8783$ MiB，输入 tensor 从约 $246.5$ MiB 降到 $2.24$ MiB，
   retained outputs 从约 $493.0$ MiB 降到 $0$。
6. **KITTI 改善并非每条序列都成立。** 逐序列 unweighted mean AbsRel 为 Full
   $0.16083$、K4 $0.13076$，paired 95% CI 支持 K4；但 K4 实际只在 13 条中的
   8 条更好、5 条更差。pixel-pooled 主表的 $0.17264\rightarrow0.13337$ 受到
   `drive_0095` 和 `drive_0016` 两个大幅改善序列的明显贡献。因此必须同时展示
   分布和失败案例，不能只展示总体百分比。

另外，现有 `run_k4_ablation.sh` 虽然文件名包含 `ablation`，实际内容只是把
Proposed K4 在 Bonn/Sintel 重复运行三次，没有任何同预算控制组，不能作为本轮
ablation 证据。

## 3. Stage 5A: same-budget baselines and component ablations

### 3.0 实现状态（2026-08-19）

Stage 5A 核心代码已经实现，尚未产生正式集群结果：

- 新增 `anchor_uniform_k4`、`random_reservoir_k4`、
  `dino_diverse_no_anchor_k4` 和 `anchor_dino_diverse_no_recent_k6`；
- Random-4 使用 frame ID 与 seed 生成稳定优先级，种子固定为 `0,1,2`，并将 seed
  写入运行 metadata 和 selection event；
- VideoDepth、pose 和 reconstruction 已统一传递 random seed，双 KV 仍使用同一组
  retained frame indices；
- `run_stage5a.sh` 支持 `video_depth`、`pose`、`reconstruction`、`finalize` 分段运行，
  根目录 `run.sh` 是唯一 Slurm 入口；
- `scripts/summarize_stage5a.py` 严格检查三域 $5/23/13$ 条 VideoDepth、8 条 TUM
  pose、8 条 TUM Dynamics、零失败、相同覆盖和同一 RTX 6000 Ada 软件栈，并输出
  10,000 次 paired bootstrap；
- 全部 CPU 回归测试为 43/43 通过（其中 Stage 5A 新增测试 6 项）；当前 Codex
  工具会话不可见 CUDA，因此真实
  Aggregator smoke 留给可见 GPU 的本地终端执行。

先执行一次最短真实 smoke：

```bash
python scripts/reproduce/smoke_inference.py \
    --repo-root . \
    --weights ckpt/checkpoints.pth \
    --images-dir examples/example_building \
    --method dino_diverse_no_anchor_k4 \
    --max-frames 5
```

正式集群运行：

```bash
sbatch run.sh
```

若集群作业需要拆分，保持输出目录不变，依次提交：

```bash
STREAMVGGT_STAGE5A_PARTS=video_depth sbatch run.sh
STREAMVGGT_STAGE5A_PARTS=pose sbatch run.sh
STREAMVGGT_STAGE5A_PARTS=reconstruction sbatch run.sh
STREAMVGGT_STAGE5A_PARTS=finalize sbatch run.sh
```

调试子集必须显式设置 `STREAMVGGT_STAGE5A_ALLOW_INCOMPLETE=1`；正式结果不得使用该
开关。正式汇总文件为 `stage5a_same_budget_results.csv`、
`stage5a_same_budget_sequence_results.csv`、`stage5a_component_results.csv` 和
`stage5a_paired_statistics.csv`。

### 3.1 先纠正 K4 的槽位语义

当前 K4 `anchor_recent_dino_diverse_k4` 的真实布局是：

```text
1 anchor + 2 DINO-selected historical frames + 1 current frame
```

代码名中的 `1recent` 对应当前帧，并不是额外的“上一历史帧”。当前帧是完成第
$t$ 帧推理所必需的输入，不能做 `remove-current/remove-recent` 消融。论文和图中
应改称 `current`。真正的 historical-recent component 存在于 K6，因此删除
recent 的 component test 放到 K6，而不是构造一个无意义的 K4。

### 3.2 K4 同预算主矩阵

所有 bounded 方法都严格最多保留 4 个整帧 coupled states；Full 只作质量参照。

| 方法 | 在线布局 | 回答的问题 |
|---|---|---|
| Full cache | 全历史 | 原版 StreamVGGT 参照 |
| Recent-4 | 最近 3 个历史帧 + current | 纯 FIFO 是否已经足够 |
| Anchor+Recent-4 | frame 0 + 最近 2 个历史帧 + current | anchor 本身的贡献 |
| Anchor+Uniform-4 | frame 0 + 2 个无 DINO 的 deterministic historical slots + current | 与 Proposed 完全相同槽位数下去掉 DINO |
| Random-4 | 3 个 seeded reservoir historical frames + current | 随机保留能否达到相同效果 |
| DINO-only-4 | 3 个 DINO-diverse historical frames + current，不强制 anchor | 去掉固定 anchor 的影响 |
| Proposed K4 | frame 0 + 2 个 DINO-diverse historical frames + current | 正式方法 |

实现约束：

- `Anchor+Uniform-4` 必须与 Proposed 使用相同 old-candidate range，只把 DINO
  排序替换为 deterministic uniform selection，不能同时改变 recent 数量。
- `Random-4` 使用预注册种子 `0,1,2`，报告三次均值、标准差和全部逐序列结果。
- `DINO-only-4` 必须真正允许 frame 0 被淘汰；当前代码中的 `dino_diverse`
  分支仍会拼接 frame 0，不能直接冒充 no-anchor ablation。
- 所有策略只能从当前 retained states 和新到达帧在线更新，不能重新读取已经删除
  的 KV，也不能利用未来总长度。

### 3.3 数据集与指标

核心 VideoDepth 全覆盖：

- Bonn 5 段、Sintel 23 段、KITTI 13 段；
- AbsRel、RMSE、$\delta_1$、FPS、peak allocated/reserved；
- 每个 bounded 方法必须覆盖与 Full 完全相同的帧。

跨任务 representative subset：

- pose：TUM 的既有 8 条序列，ATE、translation/rotation RPE；
- reconstruction：TUM Dynamics 的既有 8 条序列，Overall、ATE、RPE；
- 选择 TUM/TUM Dynamics 是因为序列长度足以触发 cache policy，NRGBD 的 2--4
  帧和多数短 7-Scenes 样本无法有效区分 K4 策略。

Stage 4A 的一组完整 VideoDepth 四方法推理约需 26 分钟；按现有计时估计，Stage
5A 的新 K4 矩阵约需 1--1.5 GPU 小时，加上 pose/reconstruction 仍属于低成本
补证据实验。

### 3.4 component 补充矩阵

K4 已回答 `no DINO` 与 `no anchor`。其余 component 按真实存在的位置测试：

1. **No historical recent（K6）**：Proposed K6 的两个 recent-history slots 改为
   DINO history，形成 `1 anchor + 4 DINO history + current`，总 K 仍为 6。
   只运行 KITTI VideoDepth、TUM pose 和 TUM Dynamics reconstruction。
2. **No temporal hierarchy（K8）**：直接复用
   `standard_dino_k8` vs `temporal_binned_dino_k8` 的 Bonn 110 帧数据；若版面只
   需要 component existence proof，放 Supplementary，不为追求显著性重跑全域。
3. **No output release（system component）**：见 Stage 5A-M，而不把执行生命周期
   和 selector component 混在一张质量表中。

### 3.5 统计与结论门槛

- 对每个数据集的 primary metric 做 paired sequence bootstrap（10,000 次）；
  Random-4 先对种子取均值，再做逐序列配对。
- 同时报告 effect size、95% CI 和 wins/ties/losses，不使用 post-hoc oracle 选方法。
- 只有 Proposed K4 相对 Anchor+Uniform-4 在至少两个 VideoDepth 域方向一致、
  paired CI 排除零，且在 TUM pose/TUM Dynamics 没有超过 5% 的 primary-metric
  退化，才允许写“DINO selection provides a consistent benefit”。
- 若仅 KITTI 有效，应改写为“DINO-based visual redundancy is particularly useful
  under outdoor domain shift”，不得声称普遍有效。
- 若 Recent-4 或 Anchor+Uniform-4 与 Proposed 无清晰差异，DINO 必须从核心创新
  降为一种低成本实现选择，论文主线转为 end-to-end bounded execution。
- 不论结果如何，都不得据此修改正式 K4。

### 3.6 Stage 5A-M: matched $2\times2$ memory-source experiment

在 `rgbd_dataset_freiburg2_desk` 上，同一作业运行：

| 组 | KV | 输入/输出生命周期 |
|---|---|---|
| A | Full | legacy retain/accumulate |
| B | Full | streaming load + release |
| C | K4 | legacy retain/accumulate |
| D | K4 | streaming load + release |

请求 prefix 为 100/250/500；逐帧记录 GPU allocated/reserved、aggregator/camera
KV、input tensor、retained output、CPU RSS 和 processed frames。OOM 是合法结果，
必须保存发生位置。

正确性 gate：A/B 的共同完成 prefix 预测应一致，C/D 也应一致；至少比较 pose/depth
hash 或 `allclose`、ATE/RPE。图中分别展示 KV pruning 和 release 对斜率/截距的
贡献，不能把二者合并归因为 selector。

已有 Stage 3.6B/4C 数据继续作为跨序列旁证，但不替代这一 matched factorial。

## 4. Stage 5B: direct bounded-StreamVGGT baselines

### 4.1 优先级

目标复现三个方法：

1. **OVGGT**：与本文同样使用 StreamVGGT checkpoint，训练免费、常数成本，且
   Dynamic Anchor Protection 与本文 anchor 最直接相关；官方仓库已标注 ECCV
   2026 accepted：<https://github.com/VAISR/OVGGT>。
2. **FrameVGGT**：整帧 KV unit 与本文 frame-level coupled state 最接近，官方代码
   支持 VideoDepth、pose、reconstruction 和相同 checkpoint：
   <https://github.com/ZhisongXu/FrameVGGT>。
3. **STAC**：CVPR 2026 Highlight，官方代码明确支持 StreamVGGT backbone、
   VideoDepth、pose、reconstruction：<https://github.com/Rainzor/STAC>。

STAC 需要自定义 CUDA/Triton extensions 和较新的 PyTorch，集成风险最高。若在
6000 Ada 上按官方环境进行两次可复现构建仍失败，则用正式发表且代码成熟的
**XStreamVGGT** 替代：<https://github.com/ywh187/XStreamVGGT>。失败日志、commit
和环境必须归档，不能只写“跑不起来”。若四者都顺利，XStreamVGGT 可作为第四个
Supplementary baseline，但主文不必继续扩张。

### 4.2 compatibility smoke

每个外部仓库先完成 10 帧 smoke：

1. 固定官方 commit 和 license；外部源码放在独立目录/环境，不覆盖本项目实现。
2. 使用同一 `checkpoints.pth`，记录 checkpoint SHA、repo commit、Python、Torch、
   CUDA、dtype 和 GPU。
3. 先运行该仓库自己的 Full/StreamVGGT baseline，再运行它的方法。
4. Full baseline 在相同输入上的 depth/pose 必须与本项目 Full 接近：AbsRel 使用
   `max(2%, 0.002)` 容差；ATE/translation RPE 使用 `max(5%, 0.002)` 容差，
   rotation RPE 使用 `max(5%, 0.05 degree)` 容差。超出时先检查 resize、crop、
   scale alignment、pose convention 和 precision。
5. parity 未通过的方法可以作为“protocol incompatible”说明，但不能进入统一排名。

### 4.3 统一核心子集

通过 smoke 的方法只运行下面的 IVC core subset：

- Sintel VideoDepth：23 段；
- KITTI VideoDepth：13 段；
- TUM pose：现有 8 段；
- held-out long sequence：`rgbd_dataset_freiburg2_desk` 的
  100/250/500/1000 帧。

这与 review 建议的 `Sintel + KITTI + TUM long sequence` 一致，同时补一个标准
TUM pose aggregate。暂不把外部方法扩张到全部 10 个 benchmark；只有主表完成且
审稿前版面确有需要时，才考虑追加 reconstruction。

### 4.4 公平比较口径

- 外部方法使用作者推荐的默认配置，不根据 KITTI/TUM 结果调预算。
- token-level、voxel-memory 与 frame-level K 的预算单位不同，因此不强行把外部
  方法标成“K4”。主表报告实测 peak memory、FPS 和质量，并用 Pareto 判断。
- 每个外部方法同时报告“相对其仓库本地 Full baseline”的 memory/FPS 比率，减少
  PyTorch/CUDA 差异对绝对速度的混淆。
- 相同 GPU、checkpoint、输入分辨率、序列、精度和输出指标能统一的必须统一；
  无法统一的 dtype/chunk/custom-kernel 差异放入表注。
- STAC 的 chunk-causal 推理不能写成严格 single-frame causal；表中单列
  `causal granularity`。

### 4.5 完成门槛与结果动作

- 至少两个直接方法通过 parity 并完成核心子集，Stage 5B 才算完成。
- 主表列：AbsRel、$\delta_1$、ATE、RPE、1000-frame completion、peak allocated、
  RSS、FPS、causal granularity、cache unit。
- 只有 K4 在质量--显存--速度上不被外部方法支配，才可写“competitive Pareto
  trade-off”；不得仅凭某一个指标宣称 SOTA。
- 若外部方法全面更优，保留结果并将贡献改为极小 frame budget、DINO selection
  与 end-to-end release 的实证分析，不删除对手或重新调 K。
- 若少于两个方法兼容，论文不得写 SOTA，只能写“comparison was limited by
  incompatible released protocols”，并在 Supplementary 给出完整复现审计。

## 5. Stage 5C: selected-frame qualitative and KITTI mechanism analysis

### 5.1 避免重复推理

Stage 5A 的 KITTI run 同时为 Full、Recent-4 和 Proposed K4 开启：

- per-frame AbsRel/$\delta_1$；
- retained frame IDs 与每次替换事件；
- current-to-retained DINO cosine similarities；
- 输入 RGB 索引；
- Full/Recent-4/K4 的 depth `.npy`，仅保存预注册可视化序列的完整 depth，其他
  序列只保存逐帧 metric，避免空间再次爆炸。

这样 Stage 5C 主要是 CPU 分析和画图，不单独重跑模型。

### 5.2 KITTI 机制统计

对全部 13 条序列报告：

1. per-sequence 与 per-frame $\Delta\mathrm{AbsRel}=\mathrm{AbsRel}_{K4}-
   \mathrm{AbsRel}_{Full}$ 分布；
2. K4/Recent-4 的 retained temporal span、更新频率、最大/平均 DINO similarity；
3. 当前帧相对历史的 redundancy statistic 与 $\Delta\mathrm{AbsRel}$ 的 Spearman
   相关；使用 sequence-cluster bootstrap CI，不能把所有像素当独立样本；
4. K4 的 8/13 胜、5/13 负结果，以及 `drive_0095`/`drive_0016` 对 aggregate
   improvement 的贡献；
5. 若 KITTI raw OXTS 已在现有下载中，则增加 DINO similarity 与相对平移/旋转的
   相关性；若 OXTS 不存在就标记未评估，不为机制图重新下载另一套 KITTI。

措辞门槛：只有相关方向在至少 9/13 序列一致且 cluster-bootstrap 95% CI 排除零，
才允许写“higher visual redundancy is associated with a larger K4 gain”。即使满足，
也只能说 association，不能在没有 attention-weight intervention 的情况下声称
已经证明 attention contamination 的因果机制。

### 5.3 预注册 qualitative cases

为避免从逐帧结果中挑最好看的图片，先冻结：

- success case：`2011_09_26_drive_0023_sync_02`，固定 current frame 100。它是
  现有 110-frame K4-positive drives 中 sequence-level 改善的中位案例，而不是
  最大改善 outlier；
- failure case：`2011_09_26_drive_0013_sync_02`，固定 current frame 100。它是
  现有 110-frame drives 中 K4 退化最大的案例；
- 两例均展示，不允许只保留 success。

主 qualitative figure 布局：

```text
Current RGB
Recent-4 retained RGBs + IDs
K4 retained RGBs + IDs + DINO similarities
Full depth | Recent-4 depth | K4 depth | GT depth
Full error | Recent-4 error | K4 error
```

另做一张机制图：左侧为 13 条 sequence 的 $\Delta$AbsRel 排序，右侧为
redundancy--framewise gain scatter/CI。已有 cache timeline 和 trajectory 继续作为
Supplementary，不用同一信息再做一张主图。

### 5.4 计划产物与论文位置

- Main Table A：K4 same-budget baseline + component ablation；放在主要跨域结果后。
- Main Table B：direct bounded-StreamVGGT comparison；放在长序列结果前或后。
- Main Figure A：selected frames + depth/error qualitative；放在 VideoDepth 定量结果后。
- Main Figure B：KITTI per-sequence gain + redundancy association；紧随 qualitative。
- Supplementary：完整 K6/K8 component、Random seeds、$2\times2$ memory trace、
  external compatibility audit 和全部逐序列表。

建议文件名：

```text
stage5a_same_budget_results.csv
stage5a_same_budget_sequence_results.csv
stage5a_component_results.csv
stage5a_memory_factorial_results.csv
stage5b_external_results.csv
stage5b_external_compatibility.csv
stage5c_kitti_frame_results.csv
stage5c_kitti_mechanism.csv
paper_assets/figures/fig_ivc_selected_frames.pdf
paper_assets/figures/fig_ivc_kitti_mechanism.pdf
```

## 6. 执行顺序与停止规则

1. 实现并单元测试 K4 控制策略，10 帧验证每步 retained IDs、K 上界和双 KV
   同步；不先跑全数据。
2. 完成 Stage 5A VideoDepth，再跑 TUM pose/TUM Dynamics；统计脚本自动输出 CI。
3. 用 Stage 5A 已保存的 KITTI artifacts 完成 Stage 5C，不重复推理。
4. 完成 Stage 5A-M matched memory factorial。
5. 依次做 OVGGT、FrameVGGT、STAC compatibility smoke；至少两个通过后再跑 Stage
   5B 正式核心子集。STAC 构建失败时启用 XStreamVGGT fallback。
6. 所有新增表通过覆盖、hardware、checkpoint、NaN/OOM 审计后，才修改论文主张。

停止规则：

- 不增加 K10/K12/K16，不重新搜索 temporal-bin；
- 不根据新结果更换 KITTI qualitative 序列或 current frame；
- 不删除 Random seed 或外部方法失败结果；
- 不因 Proposed 输给简单 baseline 而发明新 selector；
- 不把 external repo 自报数据与本项目实测数据混为同协议结果。

## 7. Stage 5 完成定义

Stage 5 只有同时满足以下条件才结束：

1. K4 同预算六个 bounded policy（含三次 Random）完成三域 VideoDepth，并完成
   TUM pose/TUM Dynamics representative subset；
2. K6 no-recent 与已有 K8 no-hierarchy 证据被纳入 component 表；
3. Full/K4 $\times$ retain/release 四格同作业结果完整；
4. 至少两个直接竞争方法通过 parity 并完成统一核心子集；不兼容日志只能解释某个
   方法为何被替换，不能代替两个实测外部 baseline；
5. success/failure retained-frame qualitative、KITTI 逐序列分布和机制相关分析
   全部生成；
6. 论文根据结果选择“支持、限定或否定”相应主张，不重新调方法。
