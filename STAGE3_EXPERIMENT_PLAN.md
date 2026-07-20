# StreamVGGT 固定显存帧选择实验计划

更新日期：2026-07-19

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

### Stage 3.3B：静态多视图重建（待修复后重跑）

首轮完整序列的相对趋势可作为预实验：

- 7-Scenes：full 0.0439；K4 0.0628；uniform K6 0.0562；old-DINO K6 0.0507。
- NRGBD：full 0.0759；K4 0.0714；uniform K6 0.0929；old-DINO K6 0.0844。
- ETH3D：full 0.7618；K4 0.8842；uniform K6 0.7578；old-DINO K6 0.7545。
- old-DINO K6 在三个数据集都优于 uniform K6，支持“DINO 选择本身有效”；但不同场景的最优 K 不同。
- old-DINO K6 相对 full cache 的宏观峰值 allocated 由 9429 MB 降至 8650 MB，FPS 由 10.67 升至 11.84。

首轮结果不能作为最终表格，原因是 7-Scenes 只有 11/18 成功、ETH3D 把 `_archives` 当成了场景，而且 prefix 指标在全序列尺度对齐后再截断，存在未来帧泄漏。修复内容：

1. 7-Scenes 和 NRGBD 在采样前剔除缺失或非有限 GT pose 的帧。
2. ETH3D 只枚举具有完整 calibration/images/depth 布局的真实场景。
3. 每个 4/6/8/10 帧 prefix 从原始预测独立执行尺度/位移对齐与 ICP。
4. 重跑 paper full-cache 与 dense 四策略，覆盖原输出并重新生成 `stage3_3b_recon_results.csv`。

ETH3D 与论文绝对数值仍有差距，需将“相同代码内的策略相对比较”和“复现论文绝对值”分开陈述。后者继续检查 THIN_PRISM_FISHEYE 畸变/valid mask 协议，不阻塞 3.3C 的策略对比。

## Stage 3.3C：TUM-dynamics 50 帧动态重建（当前阶段）

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

## Stage 3.4：超长序列、锚点老化与回环

在 3.3C 后进行，不再用短序列推断长期稳定性。

1. Bonn 五段 110 帧连续序列，记录每 10 帧的 depth/pose/资源指标。
2. 7-Scenes 每序列前 50 帧并拼接反向 50 帧形成回环，比较回到旧视角时的恢复能力。
3. 对比 K4、K6、full cache；必要时加“周期更新稳定锚点”和“DINO 触发锚点替换”。
4. 统计被选历史帧年龄、覆盖跨度、重复选择率和锚点驻留时间，确认失败来自容量不足还是锚点老化。

进入条件：3.3B 修复版无系统性数据失败，3.3C 至少 8 个序列全部得到定量结果。

## Stage 3.5：DINO 选择器升级与最终消融

根据 3.4 的失败模式再决定实现，不提前堆复杂度。候选为分段 DINO coreset：保留一个稳定参考槽、一个近期槽，其余槽位按时间分段后用 DINO 相似度与多样性联合选择；允许稳定锚点在覆盖/新颖度阈值触发时变化，而不是永远固定前三帧。

最终消融至少包含：

- 相同 K 下 DINO vs uniform vs FIFO。
- K4 vs K6，分离“选择算法收益”和“容量收益”。
- 固定锚点 vs 可更新锚点。
- 有/无时间分段与多样性约束。
- Video depth、pose、静态 MV recon、动态 TUM recon、长序列回环五类证据。

最终方案只有在多任务质量下降可接受、显存随序列长度保持有界且 DINO 相对同 K 非 DINO 基线有稳定收益时，才进入论文主结论。
