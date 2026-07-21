# StreamVGGT 固定显存帧选择实验计划

更新日期：2026-07-21

## 目标与实验原则

目标是在不改变 StreamVGGT 权重的前提下，用 DINO 驱动的历史帧选择把 KV cache 从随序列线性增长改为固定预算，同时尽量保持深度、位姿和多视图/动态重建质量。CPU offload 不是主方案，只能作为显存来源诊断或工程基线；固定 K 仍必须通过 DINO 选择体现研究贡献。

所有正式对比固定同一模型、输入帧、分辨率和随机种子，并至少包含：

- `full_cache`：质量上界和线性显存基线。
- `stage3_2_k4`：K4，`anchor_recent_dino_diverse_2old_1recent`。
- `uniform_k6`：K6 非 DINO 对照，`anchor_recent_uniform`。
- `old_dino_k6`：K6 旧 DINO，`anchor_recent_dino_diverse`。
- `fifo_k6`：只做少量诊断，不进入每个正式矩阵。

主要资源指标为峰值 allocated/reserved、总推理时间、FPS、平均/末帧延迟；质量指标按任务分别报告，不只看 video depth。

## 已完成结论

### Stage 3.2：Bonn scaling

已形成 full cache、Stage 3.2 K4、uniform K6、old-DINO K6 四条主要基线。K4 的稳定锚点是序列早期的三帧，当前帧槽位随时间变化；它不是每次重新选择四个任意历史帧。该结构节省显存，但超长序列的早期锚点老化风险需要在 Stage 3.4 单独验证。

### Stage 3.3A：Sintel / ScanNet / TUM pose

三套数据均已完成四策略 pose 对比。结果说明固定 cache 的优势不是只存在于深度：它显著降低显存并提高吞吐，但 ScanNet 和 Sintel 上 full cache 仍有明显的位姿质量优势。TUM 上 K4 的 ATE（0.0250）略好于 full cache（0.0269），old-DINO K6 的旋转误差最接近 full cache；因此不能仅凭单一数据集宣布 K4 全面优于 K6。

### Stage 3.3B：静态多视图重建（已完成可用部分，继续诊断 7-Scenes）

首轮完整序列的相对趋势可作为预实验：

- 7-Scenes：full 0.0439；K4 0.0628；uniform K6 0.0562；old-DINO K6 0.0507。
- NRGBD：full 0.0759；K4 0.0714；uniform K6 0.0929；old-DINO K6 0.0844。
- ETH3D：full 0.7618；K4 0.8842；uniform K6 0.7578；old-DINO K6 0.7545。
- old-DINO K6 在三个数据集都优于 uniform K6，支持“DINO 选择本身有效”；但不同场景的最优 K 不同。
- old-DINO K6 相对 full cache 的宏观峰值 allocated 由 9429 MB 降至 8650 MB，FPS 由 10.67 升至 11.84。

修复版已经消除 ETH3D 误枚举和 prefix 未来帧泄漏；ETH3D 13/13、NRGBD 9/9 成功。7-Scenes 在四种策略下均为相同的 12/18 成功，说明剩余 6 段首先是数据/协议问题而不是缓存策略问题。已实施的修复内容：

1. 7-Scenes 和 NRGBD 在采样前剔除缺失或非有限 GT pose 的帧。
2. ETH3D 只枚举具有完整 calibration/images/depth 布局的真实场景。
3. 每个 4/6/8/10 帧 prefix 从原始预测独立执行尺度/位移对齐与 ICP。
4. 重跑 paper full-cache 与 dense 四策略并生成修复版结果表。

剩余工作是用失败 JSON 对 7-Scenes 六段逐项归因；在此之前不把 12/18 宏平均与论文完整 18 段结果混写。Stage 3.4 的 7-Scenes 回环会自动排除有效位姿不足 50 帧的序列，因此不被该诊断阻塞。

ETH3D 与论文绝对数值仍有差距，需将“相同代码内的策略相对比较”和“复现论文绝对值”分开陈述。后者继续检查 THIN_PRISM_FISHEYE 畸变/valid mask 协议，不阻塞 3.3C 的策略对比。

## Stage 3.3C：TUM-dynamics 50 帧动态重建（已完成）

论文补充实验在 TUM-dynamics 每个序列使用 50 帧。为保证因果顺序和确定性，本实验使用 MonST3R `prepare_tum.py` 生成的 `rgb_90` 前 50 帧；通过 `rgb.txt` 与 `depth.txt` 在 0.02 秒内最近邻关联原始深度，不复制深度文件。

数据范围为 Freiburg3 的 sitting/walking 各四种运动，共 8 个序列。正式矩阵为 full cache、Stage 3.2 K4、uniform K6、old-DINO K6；每个序列输出：

- 动态聚合点图的 Acc/Comp/NC（mean 与 median）和 Overall。
- ATE、RPE translation、RPE rotation。
- 10/20/30/40/50 帧独立 prefix 重建曲线。
- 总推理时间、FPS、平均/末帧延迟、峰值 allocated/reserved。
- 固定 cache 的逐帧选择日志。

正式运行默认 `--no-save-artifacts`。只有在定量结果确定后，选 `sitting_static`、`sitting_xyz`、`walking_xyz`、`walking_halfsphere` 四段按需重跑并保存点云，用于定性图。

判定规则：

- 若 old-DINO K6 在 walking 序列稳定优于 uniform K6，说明 DINO 对动态长期上下文有独立贡献。
- 若 K6 明显优于 K4，Stage 3.4 将 K6 作为主预算、K4 作为高压缩方案。
- 若 K4/K6 都在后半段明显退化而 full cache 稳定，则下一优先级是锚点更新机制，不是 CPU offload。
- 若所有策略同向异常，先检查 RGB-depth-pose 关联及动态聚合评价协议，不据此调帧选择。

八段均成功。50 帧总体 Overall 为 full 0.0672、K4 0.0695、old-DINO K6 0.0754、uniform K6 0.0838（越低越好）；K4 在质量、ATE 和资源之间是当前最强压缩方案，old-DINO K6 明显优于同 K 的 uniform K6，继续支持 DINO 选择本身的贡献。full cache 峰值 allocated 为 13.15 GB、FPS 5.68；K4 为 8.55 GB、FPS 12.08。该结果允许进入 Stage 3.4，但还不能排除 K4 固定早期锚点在 100 帧以上老化。

运行顺序：

```bash
python scripts/check_stage3_3c_data.py

# 可选：单序列 6 帧 smoke test
STREAMVGGT_STAGE3_3C_MAX_SCENES=1 \
STREAMVGGT_STAGE3_3C_MAX_FRAMES=6 \
STREAMVGGT_STAGE3_3C_PREFIX_FRAMES="4 6" \
bash run_stage3_3c_recon.sh

# 正式集群任务
sbatch run_stage3_3c_recon_pro6000.sh
```

## Stage 3.4：超长序列、锚点老化与回环（已实现，待集群运行）

不再用短序列推断长期稳定性。正式矩阵仍为 full cache、Stage 3.2 K4、uniform K6、old-DINO K6；第一轮不加入新策略，避免一边诊断失败模式一边改选择器。

### Stage 3.4A：Bonn 110 帧

- 五段序列各做一次 110 帧因果推理，10/20/.../110 帧指标在同一次预测上分别做尺度/位姿对齐，不重复 11 次 GPU 推理。
- 每个 prefix 报告 depth（AbsRel/RMSE/δ1）、pose（ATE/RPE）、KV/descriptor/output/input 显存来源、CUDA allocated/reserved、平均帧延迟和末帧延迟。
- 完整序列报告真实推理峰值、FPS，并保存压缩轨迹与小型 JSON trace；不保存逐帧 `.npy` depth 或点云。

### Stage 3.4B：7-Scenes 100 帧合成回环

- 读取官方 TestSplit，只保留至少 50 个有限 GT pose 的序列；取前 50 个有效帧，再按完全反序拼接为 100 帧。被排除序列必须在日志中列明，不能静默计入失败平均。
- 除全局 ATE/RPE 外，比较成对重访帧的位移误差、旋转误差，以及逐对独立尺度对齐后的预测 depth 一致性。
- 10/20/.../100 帧记录资源曲线；50 帧以后专门观察回到旧视角时是否仍保留对应历史帧。

两部分共同统计 retained frame 年龄、时间覆盖跨度、选择 churn、0 号锚点驻留率、累计不同历史帧数；回环部分额外统计 matching forward frame 的保留率。由这些指标区分“容量不足”“早期锚点老化”和“DINO 没有选中可重定位帧”。

实现文件：

- `src/eval/long_sequence/eval_stage3_4_long.py`：单次推理、prefix 质量/资源、回环与选择统计。
- `scripts/check_stage3_4_data.py`：正式运行前检查 Bonn 五段和 7-Scenes 可用序列。
- `scripts/summarize_stage3_4.py`：输出 `stage3_4_results.csv` 与 `stage3_4_sequence_results.csv`。
- `run_stage3_4.sh` / `run_stage3_4_pro6000.sh`：四策略本地入口与 PRO6000 SLURM 入口。

运行顺序：

```bash
python scripts/check_stage3_4_data.py

# 可选：只做 Bonn 单序列 10 帧 smoke test（四种策略）
STREAMVGGT_STAGE3_4_PARTS=bonn \
STREAMVGGT_STAGE3_4_BONN_SEQUENCES=balloon2 \
STREAMVGGT_STAGE3_4_MAX_FRAMES=10 \
STREAMVGGT_STAGE3_4_BONN_PREFIX_FRAMES="5 10" \
bash run_stage3_4.sh

# 正式两部分 × 四策略
sbatch run_stage3_4_pro6000.sh
```

若希望分成两个任务，可分别设置 `STREAMVGGT_STAGE3_4_PARTS=bonn` 和 `STREAMVGGT_STAGE3_4_PARTS=7scenes_loop` 后提交；结果目录兼容后续合并汇总。

进入条件：3.3B 修复版无系统性数据失败，3.3C 至少 8 个序列全部得到定量结果。

进入条件已满足：3.3C 为 8/8；3.3B 的剩余失败严格集中于相同的 7-Scenes 数据段，3.4B 已通过显式有效位姿筛选隔离该问题。

## Stage 3.5：DINO 选择器升级与最终消融

根据 3.4 的失败模式再决定实现，不提前堆复杂度。候选为分段 DINO coreset：保留一个稳定参考槽、一个近期槽，其余槽位按时间分段后用 DINO 相似度与多样性联合选择；允许稳定锚点在覆盖/新颖度阈值触发时变化，而不是永远固定前三帧。

最终消融至少包含：

- 相同 K 下 DINO vs uniform vs FIFO。
- K4 vs K6，分离“选择算法收益”和“容量收益”。
- 固定锚点 vs 可更新锚点。
- 有/无时间分段与多样性约束。
- Video depth、pose、静态 MV recon、动态 TUM recon、长序列回环五类证据。

最终方案只有在多任务质量下降可接受、显存随序列长度保持有界且 DINO 相对同 K 非 DINO 基线有稳定收益时，才进入论文主结论。
