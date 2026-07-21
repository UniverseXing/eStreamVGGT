# StreamVGGT 固定显存帧选择实验计划

更新日期：2026-07-21

## 目标与实验原则

目标是在不改变 StreamVGGT 权重的前提下，用 DINO 驱动的历史帧选择把 KV cache 从随序列线性增长改为固定预算，同时尽量保持深度、位姿和多视图/动态重建质量。CPU offload 不是主方案，只能作为显存来源诊断或工程基线；固定 K 仍必须通过 DINO 选择体现研究贡献。

所有对比固定同一模型、输入帧、分辨率和随机种子。已有基线及后续使用规则为：

- `full_cache`：质量上界和线性显存基线。
- `stage3_2_k4`：K4，`anchor_recent_dino_diverse_2old_1recent`。
- `old_dino_k6`：K6 旧 DINO，`anchor_recent_dino_diverse`。
- `uniform_k6`：已经完成同 K 非 DINO 消融，保留历史结果，但不进入 Stage 3.5 及后续新增实验。
- FIFO：只做少量因果诊断，不进入完整正式矩阵。

主要资源指标为峰值 allocated/reserved、总推理时间、FPS、平均/末帧延迟；质量指标按任务分别报告，不只看 video depth。

## 已完成结论

### Stage 3.2：Bonn scaling

已形成 full cache、Stage 3.2 K4、uniform K6、old-DINO K6 四条主要基线。Stage 3.2 K4 固定保留 0 号锚点，其余两个 DINO 历史槽可以更新，最后一个槽是当前帧；它不保证保留上一帧。Stage 3.4 已进一步区分“普遍锚点老化”和“动态场景缺少近期连续性”。

### Stage 3.3A：Sintel / ScanNet / TUM pose

三套数据均已完成四策略 pose 对比。结果说明固定 cache 的优势不是只存在于深度：它显著降低显存并提高吞吐，但 ScanNet 和 Sintel 上 full cache 仍有明显的位姿质量优势。TUM 上 K4 的 ATE（0.0250）略好于 full cache（0.0269），old-DINO K6 的旋转误差最接近 full cache；因此不能仅凭单一数据集宣布 K4 全面优于 K6。

### Stage 3.3B：静态多视图重建（已完成，7-Scenes 按 12/18 报告）

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

失败 JSON 已确认六段固定为 `office/seq-06,07,09` 和 `redkitchen/seq-06,12,14`，均在模型推理前因抽样后不足两帧失败。Stage 3.4 在同一 18 段上全部成功，证明 pose 数量足够；实际原因是部分 `depth.proj.png` 未生成。缺失投影深度现已补齐，但按决定不重跑 Stage 3.3B，已有结果明确标记为 12/18 有效序列，不与完整 18 段论文结果混写。

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

# 正式运行
bash run_stage3_3c_recon.sh
```

## Stage 3.4：超长序列、锚点老化与回环（已完成）

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
- `run_stage3_4.sh`：四策略统一运行入口。

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
bash run_stage3_4.sh
```

若希望分成两个任务，可分别设置 `STREAMVGGT_STAGE3_4_PARTS=bonn` 和 `STREAMVGGT_STAGE3_4_PARTS=7scenes_loop` 后提交；结果目录兼容后续合并汇总。

进入条件：3.3B 修复版无系统性数据失败，3.3C 至少 8 个序列全部得到定量结果。

进入条件已满足：3.3C 为 8/8；3.3B 的 7-Scenes 缺失投影深度已完成归因，3.4B 不依赖该投影深度。

### Stage 3.4 结论

- Bonn 110 帧上 K4 的 aggregator KV 固定为 376.5 MiB，峰值 allocated 从 full 的 19.85 GB 降到 8.78 GB，FPS 从 3.64 提升到 13.36；但 retained outputs 仍由 44.8 MiB 增至 493.0 MiB，因此当前只证明 KV 有界，端到端内存还需在选择器定型后做真正流式释放。
- K4 的 Bonn AbsRel 为 0.0755，接近 full 的 0.0747。总体 pose 退化几乎完全来自 `person_tracking2`：其旋转 RPE 为 full 5.44°、K4 41.67°、old-DINO K6 21.58°。去掉该段后，K4 的四段平均 ATE 0.0275，略好于 full 0.0291。
- 当前 K4 不是固定前三帧。0 号锚点始终保留，两个 DINO 历史槽可以变化，但变化较少；它不保证保留上一帧。`person_tracking2` 的失败优先指向近期时序连续性不足，而不是普遍锚点老化。
- 7-Scenes 18/18 回访实验成功。K4 的回访平移、旋转和 depth consistency 分别为 0.00381、0.062°、0.00282，均明显优于 full 的 0.01469、0.245°、0.00982；但该协议是完全相同图像的反序重访，应称为 synthetic revisit consistency，不宣称已经实现传统 SLAM 回环优化。
- old-DINO K6 相对 uniform K6 在 Bonn depth 和 7-Scenes 回访上继续占优，同 K 非 DINO 对照的证据已经充分。后续不再重复运行 uniform K6。

## Stage 3.5A：近期帧连续性诊断（当前阶段）

只在 Bonn `person_tracking2` 上运行 110 帧，不扩张成完整矩阵：

1. `full_cache`。
2. `stage3_2_k4`：0 号锚点 + 2 个 DINO 历史槽 + 当前帧，不保证上一帧。
3. `old_k4`：0 号锚点 + 1 个 DINO 历史槽 + 上一帧 + 当前帧。
4. `fifo_k4`：纯近期帧，只用于诊断时序连续性。
5. `old_dino_k6`：0 号锚点 + 2 个 DINO 历史槽 + 两个近期历史帧 + 当前帧。

不再加入 uniform K6。五种方法复用 Stage 3.4 的单次推理/prefix evaluator，报告每 10 帧 depth、pose、资源与 retained IDs。

```bash
# 可选 10 帧执行检查
STREAMVGGT_STAGE3_5A_MAX_FRAMES=10 \
STREAMVGGT_STAGE3_5A_PREFIX_FRAMES="5 10" \
bash run_stage3_5a.sh

# 正式 person_tracking2 110 帧
bash run_stage3_5a.sh
```

输出为 `stage3_5a_results.csv` 和 `stage3_5a_sequence_results.csv`。

判定规则：

- old K4 与 FIFO K4 都显著修复 pose：近期帧连续性是主因，优先把 old K4 升为新主候选。
- 只有 FIFO K4 修复：DINO 历史帧可能被动态人物干扰，下一步研究静态区域特征或运动抑制。
- K4 都不修复、old-DINO K6 明显更稳：K4 只保留为高压缩/depth 方案，pose 主预算使用 K6。
- 所有固定 cache 都失败：需要增加 camera cache 的独立近期预算，不能只改锚点。

## Stage 3.5B：是否新增 new-DINO K6，以及增量回测 3.3

不在 3.5A 前实现 new K6。当前 old-DINO K6 已保留两个近期历史帧；机械照搬 new K4、把 K6 改成“0 号锚点 + 4 个 DINO 历史槽 + 当前帧”可能放大 `person_tracking2` 的失败。

根据 3.5A 决定：

- 若近期帧关键，候选 new K6 为“0 号锚点 + 3 个 DINO 历史槽 + 上一帧 + 当前帧”，只牺牲一个近期槽增加长期覆盖。
- 若近期帧并不关键，才测试“0 号锚点 + 4 个 DINO 历史槽 + 当前帧”。
- 若 old K4 已形成更好的质量/资源 Pareto，且 old-DINO K6 没有独立缺陷，则不为增加方法数量而新增 K6。

只有 new K6 通过 `person_tracking2` 后，才增量补跑它的 Stage 3.3，不重跑 full/K4/old-DINO K6/uniform K6：

1. Stage 3.3A：Sintel、ScanNet、TUM pose。
2. Stage 3.3B：NRGBD、ETH3D，以及原结果中成功的同一组 12 个 7-Scenes 序列。即使现在 proj 已补齐，也不改成 18 段，以保证和已有 K6 结果同覆盖比较。
3. Stage 3.3C：TUM-dynamics 八段。

所有后续集群入口只新增或修改普通 `run_*.sh`，不再创建硬件命名的 `*_pro6000.sh`。

## Stage 3.5C：选择器定型与最终消融

根据 3.5A/3.5B 再决定是否实现分段 DINO coreset，不提前堆复杂度。候选保留一个稳定参考槽和至少一个近期历史槽，其余槽位按时间分段后用 DINO 相似度与多样性联合选择；稳定锚点只在证据支持时通过覆盖/新颖度阈值更新。

最终消融至少包含：

- 相同 K 下 DINO vs uniform vs FIFO；优先复用已经完成的 uniform 结果，不重复运行。
- K4 vs K6，分离“选择算法收益”和“容量收益”。
- 固定锚点 vs 可更新锚点。
- 有/无时间分段与多样性约束。
- Video depth、pose、静态 MV recon、动态 TUM recon、长序列回环五类证据。

最终方案只有在多任务质量下降可接受、显存随序列长度保持有界且 DINO 相对同 K 非 DINO 基线有稳定收益时，才进入论文主结论。
