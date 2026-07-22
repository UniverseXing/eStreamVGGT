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

## Stage 3.5A：近期帧连续性诊断（已完成）

只在 Bonn `person_tracking2` 上运行 110 帧，不扩张成完整矩阵：

1. `full_cache`。
2. `stage3_2_k4`：0 号锚点 + 2 个 DINO 历史槽 + 当前帧，不保证上一帧。
3. `old_k4`：0 号锚点 + 1 个 DINO 历史槽 + 上一帧 + 当前帧。
4. `fifo_k4`：纯近期帧，只用于诊断时序连续性。
5. `old_dino_k6`：0 号锚点 + 2 个 DINO 历史槽 + 两个近期历史帧 + 当前帧。

不再加入 uniform K6。五种方法复用 Stage 3.4 的单次推理/prefix evaluator，报告每 10 帧 depth、pose、资源与 retained IDs。

```bash
# 可选 10 帧执行检查
env STREAMVGGT_STAGE3_5A_MAX_FRAMES=10 \
    STREAMVGGT_STAGE3_5A_PREFIX_FRAMES="5 10" \
    sbatch run.sh

# 正式 person_tracking2 110 帧
sbatch run.sh
```

输出为 `stage3_5a_results.csv` 和 `stage3_5a_sequence_results.csv`。

判定规则：

- old K4 与 FIFO K4 都显著修复 pose：近期帧连续性是主因，优先把 old K4 升为新主候选。
- 只有 FIFO K4 修复：DINO 历史帧可能被动态人物干扰，下一步研究静态区域特征或运动抑制。
- K4 都不修复、old-DINO K6 明显更稳：K4 只保留为高压缩/depth 方案，pose 主预算使用 K6。
- 所有固定 cache 都失败：需要增加 camera cache 的独立近期预算，不能只改锚点。

### Stage 3.5A 结论

- 近期帧数量与局部旋转稳定性呈明确单调关系：Stage 3.2 K4、old K4、old-DINO K6、FIFO K4 的最终旋转 RPE 分别为 41.67°、31.30°、21.58°、4.72°。old-DINO K6 将明显旋转失稳从约 40 帧推迟到 60 帧以后。
- 但四种固定 cache 的最终 ATE 都约为 0.65，显著差于 full cache 的 0.148；FIFO 修复局部旋转仍不能修复全局漂移。因此结果同时命中“近期连续性关键”和“所有固定 cache 的全局 pose 失败”两部分，不能只靠重新排列同一套保留帧解决。
- FIFO K4 的 AbsRel 为 0.1402，而 Stage 3.2 K4、old-DINO K6 分别为 0.0452、0.0549，说明 DINO 长期帧仍是深度/几何质量的必要组成。
- 当前实现用同一组 frame indices 同时裁剪 aggregator KV 与 camera-head KV。110 帧时 full aggregator KV 为 10353.75 MiB，camera KV 仅为 27.5 MiB；下一步应让 DINO 控制占主要显存的 aggregator cache，并独立寻找 camera cache 的最小有效时序预算。

## Stage 3.5B：双 cache 解耦、新 DINO K6 与 Stage 3.3 回测门槛（当前阶段）

仍只在 Bonn `person_tracking2` 上运行 110 帧，每 10 帧报告 depth、pose、资源以及 aggregator/camera 各自的 retained IDs。Stage 3.5B 自包含地运行三个控制组，避免代码或集群环境变化使 gate 引用不可比的旧结果：

1. `full_cache`。
2. `stage3_2_k4`：aggregator 与 camera 继续耦合，0 号锚点 + 2 个 DINO 历史槽 + 当前帧。
3. `old_dino_k6`：aggregator 与 camera 继续耦合，0 号锚点 + 2 个 DINO 历史槽 + 两个近期历史帧 + 当前帧。

五个新增实验保持 Stage 3.2 DINO K4 aggregator 或固定 K6 总预算，只改变一个因素：

4. `split_k4_camera4`：DINO K4 aggregator；camera 为 0 号锚点 + 最近 3 帧（包含当前帧）。
5. `split_k4_camera8`：DINO K4 aggregator；camera 为 0 号锚点 + 最近 7 帧（包含当前帧）。
6. `split_k4_camera16`：DINO K4 aggregator；camera 为 0 号锚点 + 最近 15 帧（包含当前帧）。
7. `split_k4_camera_full`：DINO K4 aggregator；camera 不裁剪。它只用于确定独立 camera cache 的质量上界，不是可进入后续回测的最终有界方案。
8. `recent_dino_k6`（新 K6）：0 号锚点 + 1 个 DINO 历史槽 + 最近 3 个历史帧 + 当前帧。相对 old-DINO K6，用一个 DINO 长期槽换取一个近期槽，K 和显存预算不变。

不再测试“0 号锚点 + 3/4 个 DINO 历史槽 + 极少近期帧”的 old-heavy new K6，因为 3.5A 已证明减少近期槽会放大旋转失稳。K8/K16 不是预设最终答案，而是总 camera budget 按 4、8、16 倍增的两个有界观测点；分别对应一个锚点加 7/15 个最近帧。

```bash
# 可选短检查；prefix 必须不超过 max frames
env STREAMVGGT_STAGE3_5B_MAX_FRAMES=10 \
    STREAMVGGT_STAGE3_5B_PREFIX_FRAMES="5 10" \
    sbatch run.sh

# 正式 110 帧完整矩阵
sbatch run.sh
```

输出为：

- `stage3_5b_results.csv`：方法及 prefix 汇总，包含独立 camera policy/window。
- `stage3_5b_sequence_results.csv`：最终 aggregator/camera retained IDs 与选择统计。
- `stage3_5b_gate.csv`：是否允许进入 Stage 3.3 增量回测的逐项判定。

### Stage 3.3 增量回测硬门槛

所有阈值以同一次 Stage 3.5B 的 `full_cache` 为参照。候选必须同时满足：

1. 全部序列成功，depth AbsRel 不超过 full 的 1.10 倍。
2. 最终旋转 RPE 不超过 full 的 1.50 倍；所有 prefix 的旋转 RPE 也不超过该上限，且相邻 prefix 的最大正向跳变不超过 5°。
3. 最终 ATE 不超过 full 的 2.0 倍。
4. 峰值 allocated 小于 10240 MiB；aggregator KV 不超过每个 cache frame 100 MiB 的线性上界。
5. camera cache 必须有界；`split_k4_camera_full` 即使质量通过，也只能触发后续 K32/K64 搜索，不能直接进入 Stage 3.3。

`scripts/check_stage3_5b_gate.py` 自动生成 gate 表。默认只报告 PASS/FAIL，不因无人通过而令 SLURM 作业失败；只有人工确认后才使用通过者继续回测。

解释分支：

- camera K4 通过：解耦本身比增加 K 更关键，以最小双 cache 为主方案。
- camera K4 失败而 K8/K16 通过：选择最小通过预算，不继续扩大。
- 只有 camera full 通过：依次补 K32/K64，定位最小有界窗口；未找到前不回测 Stage 3.3。
- `recent_dino_k6` 通过而 split K4 均不通过：新 K6 成为 pose 主预算，K4 保留为高压缩/depth 方案。
- 所有候选失败：转向“长期全局参考 bank + 近期运动 bank”，不浪费算力做完整 Stage 3.3。

### Stage 3.5B 结论

- 五个新增候选均未通过回测门槛。四个 split K4 的 AbsRel 都保持 0.0452，但 ATE 仍约 0.666、旋转 RPE 仍约 41°–42°。
- 即使 camera cache 保留全部 110 帧，K4 aggregator 下的 ATE/旋转也只有 0.6668/41.24°，与耦合 K4 的 0.6666/41.67°几乎相同。camera full 的最终 IDs 已确认是 `[0..109]`，因此不是裁剪实现错误；K32/K64 在同一 K4 aggregator 下不再测试。
- `recent_dino_k6` 在 60 帧时优于 old-DINO K6（旋转 6.27° vs 15.10°），但 110 帧时反而为 28.97° vs 21.58°，且 AbsRel 恶化到 0.0651。用一个长期 DINO 槽交换一个近期槽只推迟失稳，不能解决全局 ATE。
- 主要瓶颈位于 aggregator 上下文：camera head 无法从已经缺失时序/全局信息的 aggregated tokens 中恢复 pose。old-DINO K6 继续作为正式 K6 基线，recent-DINO K6 只保留为诊断结果。

## Stage 3.5C-1：Aggregator/Camera 反向交叉与 K8 容量诊断（已完成）

继续只运行 Bonn `person_tracking2` 110 帧及每 10 帧 prefix。Stage 3.5B 固定 DINO K4 aggregator、扩大 camera 历史没有效果；本轮反过来固定近期 aggregator、扩大 camera 历史，并加入一个标准 DINO K8 容量控制。

四个自包含控制组：

1. `full_cache`。
2. `stage3_2_k4`。
3. `fifo_k4`。
4. `old_dino_k6`。

四个新增组：

5. `fifo_k4_camera16`：aggregator 只保留最近 4 帧；camera 保留 0 号锚点 + 最近 15 帧（包含当前帧）。
6. `fifo_k4_camera_full`：aggregator 只保留最近 4 帧；camera 不裁剪，用作全局 pose 诊断上界。
7. `anchor_recent_k4_camera_full`：aggregator 为 `[0, t-2, t-1, t]`；camera 不裁剪。它与 FIFO K4 的对照用于判断固定早期锚点是改善 ATE，还是会干扰动态场景局部旋转。
8. `standard_dino_k8`：0 号锚点 + 3 个 DINO 历史槽 + 最近 3 个历史帧 + 当前帧。它同时不减少 old-DINO K6 的长期/近期信息，用于区分 K6 容量不足与选择结构错误。

```bash
# 可选 10 帧执行检查
env STREAMVGGT_STAGE3_5C1_MAX_FRAMES=10 \
    STREAMVGGT_STAGE3_5C1_PREFIX_FRAMES="5 10" \
    sbatch run.sh

# 正式矩阵
sbatch run.sh
```

输出为 `stage3_5c1_results.csv`、`stage3_5c1_sequence_results.csv` 和 `stage3_5c1_gate.csv`。gate 沿用 Stage 3.5B 的全部硬门槛，并额外报告 `pose_diagnostic_pass`：

- `pose_diagnostic_pass` 只要求序列成功、ATE、旋转和 prefix 稳定性通过，用于判断 FIFO aggregator + 长期 camera 是否值得发展为独立 pose 分支。
- `eligible_for_stage3_3` 仍额外要求 depth、峰值显存、aggregator/camera 都有界；诊断通过不等于可以回测 Stage 3.3。
- `fifo_k4_camera_full` 和 `anchor_recent_k4_camera_full` 无论质量如何都不能直接进入 Stage 3.3。

判定分支：

- `fifo_k4_camera_full` 的 pose 通过：aggregator 近期上下文与 camera 长期历史可以互补；下一步实现双 aggregator 原型，DINO K4 服务 depth/geometry，FIFO K4 服务 pose，再用 K16/K32 搜索 pose 分支的最小有界 camera budget。
- camera full 通过而 camera K16 不通过：只在 FIFO aggregator 分支补 K32/K64；不再对 DINO K4 aggregator 扩 camera。
- FIFO + camera full 仍不通过 pose：全局 pose 也需要 aggregator 级长期信息，放弃仅靠 camera KV 恢复的路线。
- `standard_dino_k8` 通过完整 gate：优先采用更简单的单 aggregator K8。
- K8 也失败：进入 Stage 3.5C-2，同 K8 比较标准 DINO 与“固定锚点 + long/middle/near 三个分段 landmark + 最近 3 个历史帧 + 当前帧”，分离容量收益与时间覆盖收益。

### Stage 3.5C-1 结论

- 无候选通过完整 gate 或 pose-only gate。FIFO aggregator + camera K16 已将平移/旋转 RPE 恢复到 0.0928/4.88°，与 full 的 0.0949/5.44°相当，但 ATE 仍为 0.648；camera full 的 0.645 没有实质改善。因此不补 camera K32/K64，也暂不实现双 aggregator。
- `anchor_recent_k4_camera_full` 在 60 帧仍为 ATE 0.196、旋转 6.85°，到 70 帧突变为 0.607/14.81°。固定 0 号锚点前期有助于全局参考，远离初始视角后则缺少中期桥接。
- `standard_dino_k8` 的深度为 0.0552、峰值 allocated 为 9.53 GB，但 ATE/旋转仍为 0.654/24.85°，最终 IDs 为 `[0,1,2,3,106,107,108,109]`。增加容量没有阻止 DINO 历史槽冻结在序列开头。
- GT 在 60 帧相对起点约旋转 162°、位移 1.49 m；70 帧位移约 1.99 m且最近十帧转向约 35°，与所有有界 aggregator 的共同失稳点一致。下一步保持 K8 不变，强制形成近/中/长期时间覆盖。

只有通过 Stage 3.5C-1 或后续阶段全部硬门槛的有界候选，才增量补跑它的 Stage 3.3，不重跑 full/K4/old-DINO K6/uniform K6：

1. Stage 3.3A：Sintel、ScanNet、TUM pose。
2. Stage 3.3B：NRGBD、ETH3D，以及原结果中成功的同一组 12 个 7-Scenes 序列。即使现在 proj 已补齐，也不改成 18 段，以保证和已有 K6 结果同覆盖比较。
3. Stage 3.3C：TUM-dynamics 八段。

所有后续集群提交只修改根目录 `run.sh`，由它完成 SLURM 资源申请、Conda 环境激活并调用阶段内部脚本；不再创建硬件命名的 `*_pro6000.sh`。

## Stage 3.5C-2：时间分段 DINO K8（当前阶段）

仍只运行 Bonn `person_tracking2` 110 帧和每 10 帧 prefix，矩阵为：

1. `full_cache`：质量参照。
2. `standard_dino_k8`：0 号锚点 + 3 个无时间约束的 DINO 历史槽 + 最近 3 个历史帧 + 当前帧。
3. `temporal_binned_dino_k8`：同样 K8，但历史槽强制分为 long/middle/near 三个时间 bank；near/middle 保证时序晋级，long bank 再用 DINO 多样性选择全局代表。

`temporal_binned_dino_k8` 的在线有界布局为：

- `anchor`：固定帧 0。
- `recent`：age 0–3，即当前帧和最近 3 个历史帧。
- `near`：age 4–15，保留该时间带最老的候选作为短期桥梁。
- `middle`：age 16–47，保留该时间带最老的候选作为中期桥梁。
- `long`：age ≥48 且排除帧 0，用 DINO 多样性保留 1 个全局 landmark。

每个时间段只在当前仍存在的 KV 候选中在线选择，不保存已经丢弃的 KV，也不事后访问全部图像，因此 aggregator KV 始终不超过 8 帧。near/middle 使用最老候选是结构约束：若这两个 bank 也只按视觉新颖度更新，代表可能反复换成新帧、永远无法老化晋级到 long；DINO 用在 long bank 内，从已晋级的长期候选中选择相对锚点和 recent 更互补的全局代表。在 middle/long 尚未形成的 warm-up 阶段，空余槽临时由最近的未选候选补满；所有 bank 形成后回到严格的 `1 anchor + 3 landmarks + 4 recent`。

```bash
# 可选 10 帧检查；短检查只能验证 recent/near，不能验证 middle/long
env STREAMVGGT_STAGE3_5C2_MAX_FRAMES=10 \
    STREAMVGGT_STAGE3_5C2_PREFIX_FRAMES="5 10" \
    sbatch run.sh

# 正式 110 帧
sbatch run.sh
```

输出为：

- `stage3_5c2_results.csv`。
- `stage3_5c2_sequence_results.csv`：包含 near/middle/long 的占用率、更新次数、unique IDs、最终 bank IDs 和最大时间缺口。
- `stage3_5c2_gate.csv`：质量、资源和 bank 结构的联合 gate。

### Stage 3.5C-2 决策门槛

质量与资源门槛继续以同一次 `full_cache` 为参照，必须全部满足：

1. 全部序列成功；AbsRel ≤ full × 1.10。
2. 最终 ATE ≤ full × 2.0。
3. 最终及所有 prefix 的旋转 RPE ≤ full × 1.50，相邻 prefix 最大正向跳变 ≤5°。
4. 峰值 allocated <10240 MiB；aggregator KV ≤ 每个 cache frame 100 MiB；camera cache 有界。
5. 70 帧附近不能再次出现 ATE/旋转突变。

`temporal_binned_dino_k8` 还必须通过结构门槛，防止质量偶然改善但 bank 实际没有工作：

1. warm-up 后 near、middle、long 各 bank 占用率均 ≥0.95。
2. near 和 middle 在 110 帧内都至少更新 1 次；long 可以稳定保留一个真正长期 landmark。
3. 最终相邻 retained IDs 的最大时间缺口 ≤64 帧。

### 达成门槛后的动作

若 `temporal_binned_dino_k8` 同时通过全部质量、资源和结构门槛：

1. 冻结 K8 bank 边界与选择规则，不再根据 `person_tracking2` 调参。
2. 只增量补跑新方法的 Stage 3.3A：Sintel、ScanNet、TUM pose。
3. 再补 Stage 3.3B：NRGBD、ETH3D、原成功的同一组 12 个 7-Scenes 序列。
4. 再补 Stage 3.3C：TUM-dynamics 八段。
5. 三类任务均保持可接受后，进入 Stage 3.5D，同 K8 增加非 DINO 时间分段对照，验证收益来自 DINO 而不只是时间分桶；已有 full/K4/K6/uniform K6 不重跑。

若 70 帧突变消失但只小幅错过数值门槛，只允许一次预先声明的 bank 边界敏感性实验，再比较 `4/12/40` 与 `4/16/48`；不能连续针对单序列调阈值。若 ATE 仍约为 0.65，或 standard/temporal K8 无实质差异，则停止增加 K、camera window 或帧槽排列，转向显式的局部窗口对齐与全局 pose stitching。

### Stage 3.5C-2 结论

- `temporal_binned_dino_k8` 的结构 gate 全部通过，最终 IDs 为 `[0,33,68,103,106,107,108,109]`，near/middle/long 占用率为 1.00/0.968/1.00，最大时间缺口由 standard K8 的约 100 帧降为 35 帧。因此后续质量失败不是 bank 未工作。
- 相对 standard DINO K8，分段 K8 的 AbsRel 从 0.0552 改善到 0.0515，平移/旋转 RPE 分别改善约 6.2%/9.6%；峰值 allocated 仍为 9.53 GB，FPS 为 full 的约 3.14 倍。DINO 有界 cache 的几何/资源主线仍成立。
- 但 ATE 仍为 0.668，旋转仍为 22.46°，70 帧突变没有消失；完整质量 gate 和 pose-only gate 均失败。这触发预先声明的停止条件：不做 `4/12/40` 边界敏感性、不增加 K10/K12、不继续排列 cache slots。
- 结论边界是“帧选择可改善几何和局部 pose，但不能单独维持该动态长序列的全局轨迹坐标”。下一步冻结 `temporal_binned_dino_k8` 作为 geometry 候选，仅做一次显式有界窗口 pose stitching 可行性实验。

## Stage 3.6A：有界重叠窗口 Pose Stitching（已完成）

目标不是继续改变 DINO 选择器，而是验证全局 ATE 是否来自窗口间隐式坐标/尺度漂移。仍只使用 Bonn `person_tracking2` 110 帧，矩阵为：

1. `full_cache`：全局 pose 质量参照。
2. `temporal_binned_dino_k8`：冻结的 raw streaming pose 参照。
3. `window16_overlap4`：每个局部窗口 16 帧、相邻窗口重叠 4 帧，每次前进 12 帧。
4. `window32_overlap8`：窗口 32、重叠 8，只作为更高上下文的诊断上界。

窗口内部从空 cache 开始并使用 full context；相邻窗口只利用双方预测的重叠相机位姿估计 Sim(3)，不使用 GT：

- 正常情况下用重叠相机中心的 Umeyama Sim(3)。
- 相机中心协方差退化时，用重叠相机朝向的平均相对旋转、重叠位移尺度和中心平移回退。
- 将整个新窗口变换到已有全局坐标，重叠帧保留先前全局结果，只追加非重叠帧。
- 窗口大小固定，显存不随总序列长度增长；必须额外报告 overlap 重算、Sim(3) fallback、重叠残差和真实唯一帧 FPS。

```bash
# 16 帧检查可覆盖一个完整 K16 窗口，但不能验证拼接；至少 28 帧才包含一次 K16/O4 拼接
env STREAMVGGT_STAGE3_6A_MAX_FRAMES=28 \
    STREAMVGGT_STAGE3_6A_PREFIX_FRAMES="10 20 28" \
    sbatch run.sh

# 正式 110 帧
sbatch run.sh
```

输出为：

- `stage3_6a_results.csv`：最终/prefix pose、窗口重算倍率、FPS、显存和最大 overlap 残差。
- `stage3_6a_gate.csv`：K16/K32 是否允许进入 Stage 3.3A。
- 各方法目录中的 `trajectory.npz` 和 `stage3_6a_metrics.json`，用于检查轨迹与逐窗口对齐事件。

### Stage 3.6A 决策门槛

以同次 `full_cache` 为参照，窗口方案必须同时满足：

1. ATE ≤ full × 2.0。
2. 最终及所有 prefix 旋转 RPE ≤ full × 1.50；相邻 prefix 最大正向跳变 ≤5°，70 帧不再突变。
3. 峰值 allocated <10240 MiB。
4. 窗口和 overlap 固定且 `3 ≤ overlap < window`，不得随序列长度增加。
5. 报告而不隐藏重算倍率、唯一帧 FPS、alignment fallback 次数及最大 overlap 平移/旋转残差。

### 达成门槛后的动作

- K16/O4 通过：优先选择最小窗口；geometry/depth 固定使用 temporal-DINO K8，pose 使用 K16/O4 stitching。先只增量跑 Stage 3.3A 的 Sintel、ScanNet、TUM pose，验证跨数据集后再决定是否组合回测 3.3B/C。
- 只有 K32/O8 通过：若其峰值仍低于 10 GiB才允许进入 Stage 3.3A；否则只作为“全局 pose 需要更大上下文”的诊断，不进入最终方案。
- K16/K32 均失败：停止扩展 pose 后端，明确论文结论边界；回到主线实现 GPU retained outputs 的真正流式释放，并验证 500/1000 帧端到端显存有界性，不再继续针对 `person_tracking2` 调参。

### Stage 3.6A 结论

- 两个窗口候选都没有通过预先声明的 gate，因此不进入 Stage 3.3A，也不放宽门槛或继续搜索窗口/overlap。K16/O4 的 ATE 0.3045、旋转 RPE 12.74°，第一次拼接即令 10→20 帧 prefix 旋转增加约 9.70°；它是明确的拼接质量失败。
- K32/O8 将 ATE 改善到 0.1880（full 的 1.27 倍），消除了 raw temporal K8 在 70 帧处的旋转突变，说明“局部重启可阻断全局坐标漂移”的诊断成立。但其旋转 RPE 8.97°超过 8.15°门槛，峰值 allocated 10.69 GiB 也超过 10 GiB，因此只保留为诊断上界。
- K16/K32 的最大 overlap 旋转残差分别为 74.10°/53.60°，且 fallback 次数均为 0。失败不是中心协方差退化，而是窗口间方向预测不一致；仅用重叠相机中心估计 Sim(3) 不足以形成最终 pose 后端。
- 冻结 `temporal_binned_dino_k8` 的 geometry 结论，停止 K、时间 bank、camera window、stitching window 与 pose graph 扩展。项目仍沿“DINO 有界 cache 优化几何推理显存”的主线推进，但最终论文必须明确动态长序列全局 pose 的限制。

## Stage 3.6B：真正流式输出释放与 100/500/1000 帧显存验证（当前阶段）

当前 K8 已将 aggregator KV 和 descriptor 固定，但旧推理路径仍有两个随序列增长的 GPU 来源：evaluator 会预先把全部输入图像搬到 GPU，`StreamVGGT.inference()` 也会在 `all_ress`/`processed_frames` 中保留全部逐帧输出与输入 view。Stage 3.6B 不再改变帧选择，而是移除这两个工程性线性项：

1. 保留旧 `inference()` 默认语义；仅在显式 streaming 模式下接受逐帧 iterable、调用 `output_sink(frame_index, prediction)`，并关闭 GPU output/view retention。
2. 每次只加载并搬运当前图像。sink 将 camera pose 复制为小型 CPU 数组；110 帧等价性实验还逐帧复制 depth 用于原指标，长序列只更新 pose/depth 哈希后立即释放大张量。
3. 继续保留 K8 KV、DINO descriptor 和内存 trace；报告 GPU allocated/reserved、输入/output/view retention、逐帧采样的当前 CPU RSS（另列进程历史峰值）、推理/端到端 FPS 与最终 temporal bank。
4. 不对 500/1000 帧运行 full cache。长序列使用现有 7-Scenes `chess/seq-03` 的原始连续帧，不构造重复序列、不使用 Stage 3.4 的 forward-reverse loop，也不需要下载新数据。

实验矩阵固定为：

1. `bonn_legacy_110`：Bonn `person_tracking2` 110 帧，temporal-DINO K8，旧预加载和 GPU 输出保留路径。
2. `bonn_stream_110`：完全相同数据、权重、K8 和预处理，逐帧输入与 sink 释放路径。
3. `7scenes_stream_100/500/1000`：同一 `chess/seq-03` 原始序列的三个 prefix，只运行 streaming temporal-DINO K8。

正式运行：

```bash
sbatch run.sh
```

只检查代码路径时可以缩短长度并跳过正式 gate：

```bash
env STREAMVGGT_STAGE3_6B_BONN_FRAMES=10 \
    STREAMVGGT_STAGE3_6B_LONG_LENGTHS="10 20" \
    STREAMVGGT_STAGE3_6B_SKIP_GATE=1 \
    sbatch run.sh
```

输出为：

- `stage3_6b_results.csv`：每个模式/长度的数值哈希、质量、吞吐、GPU/CPU 内存及 cache 统计。
- `stage3_6b_gate.csv`：是否允许进入缩小范围的 geometry selector 归因实验。
- 各方法目录的 `stage3_6b_metrics.json`、`memory_trace.json` 和小型 `trajectory.npz`。

### Stage 3.6B 决策门槛

必须同时满足：

1. 110 帧 legacy/stream 的 camera pose 与 depth SHA-256 相同，全部共有 depth/pose 指标绝对差 ≤1e-5。
2. streaming trace 必须明确为逐帧输入和 sink 输出；GPU retained outputs/views 均为 0，最大输入 tensor 不超过 legacy 单帧均值的 1.05 倍。
3. temporal-DINO K8 固定不变，所有 streaming run 的 aggregator KV ≤800 MiB。
4. 110/100/500/1000 帧 streaming 全部成功且峰值 allocated <10240 MiB；1000 帧峰值不得比 500 帧高超过 256 MiB。
5. 单独报告 CPU RSS；1000 帧 RSS peak 不得比 500 帧高超过 256 MiB，防止把大输出从 GPU 隐藏到 CPU。
6. 110 帧 streaming 端到端 FPS ≥同次 legacy K8 的 80%。

### 达成门槛后的动作

- 全部通过：冻结真正流式执行路径，进入 Stage 3.7A 的 geometry-only DINO 归因。使用完全相同的 K8 temporal banks，只改变 bank 内代表选择，对比 DINO 与非 DINO；增量运行已有 video depth、静态 MV recon 和动态 TUM recon，不再声称 raw global pose 已解决。
- 数值等价失败：优先修复 sink/逐帧预处理语义，不能用容差掩盖不同输入或输出。
- GPU 不平台：根据 trace 只修复仍增长的 input/output/view 或 cache 引用；不返回 pose/window 调参。
- GPU 平台但 CPU RSS 失败：将 trace/pose 改为在线汇总或分块写盘后重跑同一矩阵，不改变选择器。
- Stage 3.7A 若不能证明 DINO 相对同 K、同时间 bank 的非 DINO 基线有稳定几何收益，则不能把收益归因于 DINO，需要重新评估论文贡献。

## Stage 3.5D：原完整选择器定型与最终消融（不再触发）

原计划要求 Stage 3.6A 形成通过 gate 的 geometry+pose 组合并完成 Stage 3.3A；该前提没有满足，因此不再执行面向“geometry+pose 全部通过”的完整 Stage 3.5D。DINO 与非 DINO 的必要归因被缩小为 Stage 3.7A，只支持 geometry/memory 范围的结论。候选保留固定 K8 和相同时间 bank，稳定锚点是否更新不再根据 `person_tracking2` 调整。

原完整消融清单至少包含：

- 相同 K 下 DINO vs uniform vs FIFO；优先复用已经完成的 uniform 结果，不重复运行。
- K4 vs K6，分离“选择算法收益”和“容量收益”。
- 固定锚点 vs 可更新锚点。
- 有/无时间分段与多样性约束。
- Video depth、pose、静态 MV recon、动态 TUM recon、长序列回环五类证据。

由于 geometry+pose 前提已经失败，上述完整主结论不再成立。后续只有在 Stage 3.6B 证明端到端 GPU 有界、Stage 3.7A 又证明 DINO 相对同 K 非 DINO 基线有稳定几何收益时，才能形成明确限定在 geometry/memory 范围内的论文结论；全局 pose 失败作为限制单独报告。
