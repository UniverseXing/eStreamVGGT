# StreamVGGT 固定显存帧选择实验计划

更新日期：2026-07-25

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

## Stage 3.5B：双 cache 解耦、新 DINO K6 与 Stage 3.3 回测门槛（已完成）

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

## Stage 3.5C-2：时间分段 DINO K8（已完成）

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
5. 原计划在三类任务可接受后进入同 K8 非 DINO 归因；该步骤已由 Stage 3.7 的最新决策取消，改为只做四组跨任务回测，并限制最终因果措辞。

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

## Stage 3.6B：真正流式输出释放与 100/500/1000 帧显存验证（已完成）

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

### Stage 3.6B 结论

- 五个 run 全部通过 gate。110 帧 legacy 与 stream 的 camera/depth SHA-256 完全一致，全部公共指标最大绝对差为 0，说明逐帧输入和 sink 释放没有改变推理数值。
- 100/500/1000 帧的峰值 allocated 都为 8782.88 MiB，500→1000 帧增量为 0；aggregator KV 最大 753 MiB，证明端到端 GPU 显存已经有界，而不只是 cache 本身有界。
- 最大输入 tensor 从 legacy 110 帧的 246.49 MiB 降为单帧 2.24 MiB；streaming retained outputs/views 都为 0。1000 相对 500 帧的 CPU RSS peak 只增加 21.04 MiB，没有把线性 GPU 输出转移到 CPU。
- 110 帧 streaming/legacy 端到端 FPS 比为 1.399，超过 0.80 门槛。至此冻结 streaming release 路径和 `temporal_binned_dino_k8`，不再进行显存来源归因或 selector 内部归因。

## Stage 3.7：四组 Stage 3.3 增量回测（已完成）

完整 DINO 归因对当前决策价值有限，因此取消原先暂定的“相同 temporal bank 内 DINO/非 DINO”实验；Stage 3.7 的 A/B/C 现重新定义为 pose、静态重建、动态重建回测。最终对照固定为四组：

1. `full_cache`：质量上界，复用已有 Stage 3.3。
2. `stage3_2_k4`：最小高压缩 DINO 方案，复用已有 Stage 3.3。
3. `old_dino_k6`：既有 K6 中综合表现最好，复用已有 Stage 3.3。
4. `temporal_binned_dino_k8`：冻结的新 K8，仅补跑这一组。

不纳入 `uniform_k6`、FIFO 和 split-camera 诊断组。`recent_dino_k6` 在 Bonn 的 AbsRel/旋转 RPE 为 0.0651/28.97°，弱于 old-DINO K6 的 0.0549/21.58°；`standard_dino_k8` 的 AbsRel/旋转 RPE 为 0.0552/24.85°，也被 temporal K8 的 0.0515/22.46°支配。因此新增计算只给 temporal K8，避免重跑已完成或已淘汰的方法。

本阶段分三部分顺序执行：

### Stage 3.7A：Pose 回测

- 数据集：Sintel 14 段、ScanNet 6 段、TUM 8 段。
- 保持原 Stage 3.3 pose evaluator、输入、分辨率和汇总语义不变，保证与旧三组直接比较。
- 该实验用于检查跨数据集灾难性退化，不把 Bonn 110 帧 raw global pose 的问题宣称为已经解决。

### Stage 3.7B：静态多视图重建回测

- NRGBD 9 段、ETH3D 13 段。
- 7-Scenes 精确使用旧 3.3B 成功的 12 段：`chess/seq-03`、`chess/seq-05`、`fire/seq-03`、`fire/seq-04`、`heads/seq-01`、`office/seq-02`、`pumpkin/seq-01`、`pumpkin/seq-07`、`redkitchen/seq-03`、`redkitchen/seq-04`、`stairs/seq-01`、`stairs/seq-04`。
- 使用 dense protocol 和 4/6/8/10 prefix；即使其余 6 段投影深度已经修好，也不改变覆盖，避免与旧结果混入数据修复收益。

### Stage 3.7C：动态重建回测

- TUM-dynamics 原 8 段，每段前 50 帧，prefix 为 10/20/30/40/50。
- 继续采用既有 `first_50`、RGB-depth-pose 关联和指标实现。

这些静态序列只有约 3–20 帧，TUM 也只有 50 帧、刚触及 long bank。它们能回答跨数据集兼容性和重建质量，不能单独证明超长序列中的 DINO bank 行为；100/500/1000 帧有界性由 Stage 3.6B 单独支撑。

正式运行只提交根目录入口：

```bash
sbatch run.sh
```

若任务中断，可只重跑指定部分；pose 还可设置 `STREAMVGGT_STAGE3_7_POSE_RESUME=1` 合并已有序列，重建部分则按所选 part 重新汇总：

```bash
env STREAMVGGT_STAGE3_7_PARTS="static dynamic" sbatch run.sh
```

输出为：

- `stage3_7_pose_results.csv`、`stage3_7b_recon_results.csv`、`stage3_7c_recon_results.csv`：新 K8 的分任务原始汇总。
- `stage3_7_comparison.csv`：旧三组与新 K8 归一后的四组最终对照。
- `stage3_7_gate.csv`：覆盖、显存、几何质量和 pose 灾难保护门槛。

### Stage 3.7 决策门槛

1. 覆盖必须精确为 pose 14/6/8、静态 12/9/13、动态 8 段，且新 K8 无失败；7-Scenes 不允许悄悄扩为 18 段。
2. 新 K8 在三类任务中的峰值 allocated 均 <10240 MiB。
3. 每个静态数据集和动态 TUM 的 `mean_overall` 分别不得超过同数据集 K4/old-K6 中较优者的 1.10 倍。
4. 每个静态数据集和动态 TUM 的 `mean_nc` 分别不得低于 K4/old-K6 中较优者 0.03。
5. Pose 只设灾难保护：每个数据集的 ATE 和旋转 RPE 均不得超过 K4/old-K6 较优值的 2 倍。该项失败不反向否定已通过的 geometry，只把结论标记为 pose limitation。

### 达成门槛后的动作

- 覆盖、显存、静态和动态 geometry 全部通过，且 pose 保护也通过：判为 `PASS_ALL_BACKTESTS`，冻结 temporal-DINO K8 为最终主候选，开始整理最终表格、选择日志可视化和论文结果描述。
- Geometry 全部通过但 pose 保护失败：判为 `PASS_GEOMETRY_WITH_POSE_LIMITATION`。仍可形成“有界显存的 DINO-guided geometry/reconstruction”结论，但必须单列全局 pose 限制，不再开发新 pose cache 或 stitching。
- 任一 geometry 指标失败：判为 `FAIL_GEOMETRY`。按失败数据集定位是静态/动态泛化问题，最终主方案回退到该数据集上较优的 K4 或 old-K6；不再新增 K、bank 或显存归因实验。
- 任一覆盖不完整：判为 `FAIL_INCOMPLETE`，只补缺失序列，不能根据不完整均值做方法决策。

因为取消了同 K、同 temporal bank 的非 DINO 对照，最终措辞必须是“集成的 DINO-guided temporal K8 达到何种质量/显存折中”，不能声称时间分段 K8 的全部增益都由 DINO 单独造成。既有 old-DINO K6 对 uniform K6 的结果仍可作为较早预算下的独立 DINO 证据。

### Stage 3.7 结论

- temporal K8 完整覆盖全部 pose、静态和动态序列，峰值 allocated 9384 MiB，全部 NC 和 pose 灾难保护通过；但 NRGBD Overall 0.09064 超过 0.07858 门槛，动态 TUM Overall 0.08069 超过 0.07649 门槛，因此正式决定为 `FAIL_GEOMETRY`，不能作为跨任务统一主方案。
- 逐序列共同覆盖为 12 个 7-Scenes、9 个 NRGBD、13 个 ETH3D 和 8 个 TUM，共 42 段。动态 TUM 上 K4 赢 6/8，temporal K8 相对 K4 在 7/8 段退化；NRGBD 的 K8 则为 4 胜、4 负、1 平，均值失败主要由 `grey_white_room`、`kitchen` 等重尾场景造成。
- 以每段最优 bounded 方法为 oracle，old K6 的平均/最大 regret 为 13.3%/60.1%，低于 K4 的 21.9%/207.8% 和 temporal K8 的 15.9%/106.5%。因此冻结角色为：old K6 是稳健默认，K4 是 compact/VideoDepth/动态配置，temporal K8 是 long-sequence/pose 扩展，full cache 只作参考。
- Stage 1 已完成 Bonn 和 Sintel VideoDepth 跨数据集表。新 K4 相对 full 的 AbsRel 在 Bonn 仅差 0.78%，在 Sintel 改善 2.57%；同 K6 下 DINO 相对 uniform 的 AbsRel 分别改善 8.73%/2.54%。这些质量结论和 Stage 2.5/2.6 的重复稳定性仍作为历史证据保留，但 Stage 1 的性能数据来自 A6000，而 Stage 3 后续主体使用 RTX 6000 Ada，显存、FPS 和耗时不能直接并入同一硬件主表。因此 Stage 4A 在 6000 Ada 上重跑统一 VideoDepth 矩阵。

## Stage 4A：6000 Ada 统一 VideoDepth 矩阵（已完成）

目标是在不改 selector、不重新选择 K 的前提下，在同一 RTX 6000 Ada、相同代码和评价协议下形成 `3 数据集 × 4 配置 = 12` 组可直接比较的 VideoDepth 主表，同时增加未参与开发的户外 KITTI 域。KITTI 严格遵循 MonST3R 协议：13 条 KITTI depth-validation drive，左相机每条最多取前 110 个带 GT depth 的帧；评价继续使用现有 scale alignment、AbsRel、SqRel、RMSE、Log RMSE 和 δ1/2/3。

阶段编号随本阶段插入统一顺延：原 Stage 4A/4B/4C 分别改名为 Stage 4B/4C/4D；可选的方法增强保持在 Stage 4E，不与当前结果补全混在一起。

正式矩阵固定为：

1. 数据集：Bonn 5 段、Sintel 23 段、KITTI 13 条 drive。
2. 配置：`full_cache`、`stage3_2_k4`、`old_dino_k6`、`temporal_binned_dino_k8`。
3. 共运行 12 组；不新增 uniform K6。Stage 1 A6000 表只作历史复现对照，不再注入 Stage 4A 汇总或性能结论；早期 standard K8 不能冒充 temporal K8。
4. 每个 runtime JSON 必须记录 GPU 型号、PyTorch/CUDA 版本和 Slurm Job ID；最终 gate 要求 12 组 GPU 名完全一致且包含 `6000 Ada`。

下载和准备在登录节点执行，不占 GPU：

```bash
bash scripts/download_stage4a_kitti.sh
python scripts/prepare_stage4a_kitti.py --root data/eval/kitti --mode hardlink
python scripts/check_stage4a_kitti.py --root data/eval/kitti
```

下载脚本使用 KITTI 官方 S3 上的 annotated depth 和 MonST3R 指定的 13 个 raw drive，支持 `wget -c` 断点续传。准备脚本默认使用 hardlink，不复制第二份图像/深度数据；源压缩包保留在 `data/eval/kitti/downloads`，确认正式结果后再人工决定是否清理。

正式运行：

```bash
sbatch run.sh
```

可选 4 帧 smoke；必须使用独立输出目录并跳过正式汇总，防止 smoke 与正式结果重名：

```bash
env STREAMVGGT_STAGE4A_METHODS="temporal_binned_dino_k8" \
    STREAMVGGT_STAGE4A_DATASETS="kitti" \
    STREAMVGGT_STAGE4A_MAX_FRAMES=4 \
    STREAMVGGT_STAGE4A_SKIP_FINALIZE=1 \
    STREAMVGGT_STAGE4A_RESULTS_ROOT="$PWD/eval_results/video_depth_stage4a_smoke" \
    sbatch run.sh
```

输出为：

- `stage4a_video_depth_results.csv`：12 组同卡 VideoDepth 质量、显存、速度和运行环境主表。
- `stage4a_gate.csv`：K4、old K6、temporal K8 在三数据集上的逐项门槛。
- `eval_results/video_depth/*stage4a_*`：逐 drive runtime/memory、depth prediction 和 scale-aligned 结果。

### Stage 4A 决策门槛

1. 12 组 GPU 名必须完全一致且包含 `6000 Ada`；任何 A6000 或缺失 GPU provenance 的旧结果都不得混入。
2. 四组必须在 Bonn/Sintel/KITTI 分别完整覆盖 5/23/13 段、无 OOM，并在同一数据集处理完全相同的总帧数。
3. 每个 bounded 方法在每个数据集都要求 AbsRel ≤同次同卡 full ×1.15，δ1 ≥同次 full−0.03；三个数据集分别判断，不能用平均值掩盖单域失败。
4. 每个 bounded 方法在每个数据集的 peak allocated 和 reserved 都必须严格低于同次同卡 full。legacy VideoDepth evaluator 不追加 reserved <10 GiB 的绝对门槛；真正流式的 10 GiB/长度平台结论仍由 Stage 3.6B 支撑。
5. temporal K8 的主方案晋级另要求：三个数据集的 AbsRel 均 ≤同数据集 K4/old-K6 较优者 ×1.10，δ1 均 ≥二者较优值−0.03。基础门槛通过但该项失败时记为 `PASS_SPECIALIST_ONLY`，继续保留 long-sequence/pose specialist 身份，不提升为统一默认。
6. KITTI 是冻结后的 held-out 域。任何失败只进入结果和限制，不允许据此调整 K、DINO 相似度、temporal bank 边界或数据抽样。

达成门槛后：K4、old K6 中通过三域基础门槛者进入 Stage 4B 的最终统计/Pareto 表，分别竞争 compact 和 robust-default 角色；temporal K8 只有额外竞争性门槛也通过才竞争统一主方案，否则按 specialist 报告。任一方法单域失败仍保留完整结果，但不得宣称跨域通过。

### Stage 4A 结论

- 12 组均在同一 RTX 6000 Ada、PyTorch 2.3.1/CUDA 12.1 和同一 Slurm 作业中完成；Bonn/KITTI/Sintel 精确覆盖 5/13/23 段、每种方法共 2883 帧且无 OOM。
- K4 和 old K6 通过全部三域基础门槛。K4 在 Bonn/KITTI/Sintel 的 AbsRel 分别为 0.07545/0.13337/0.31605，三个数据集都优于 old K6 和 temporal K8；KITTI 相对 full 的 AbsRel 改善 22.75%，δ1 提升 10.68 个百分点。
- K4 三域合计推理时间 336.80 秒、加权 FPS 8.56，而 full 为 545.03 秒/5.29 FPS；最大 allocated/reserved 从 full 的 21641/44804 MiB 降为 10566/11558 MiB。K4 冻结为 VideoDepth/compact 主方案，old K6 继续作为跨 reconstruction 的 robust default。
- temporal K8 在 KITTI 的 AbsRel/δ1 为 0.19230/0.67373，δ1 低于 0.69143 基础门槛；相对 K4 的 KITTI AbsRel 退化 44.19%。正式决定为 `FAIL`，不能作为统一主方案，只保留 long-sequence/pose specialist 和失败案例，不再调整 K、temporal bin 或 DINO 阈值。

## Stage 4B：最终统计定型（已完成）

当前先完成 Stage 4B-VD，不运行模型推理。利用 Stage 4A 已保存的逐帧 `.npy` prediction 重新运行纯 depth evaluation，把 evaluator 内部已经按序列计算的指标写入 `result_scale_sequences.json`，再和每个目录现有的 `runtime_memory_rank0.json` 按序列严格配对。

正式统计口径：

1. 固定 41 条序列 ×4 方法，共 164 条记录；四方法在同一序列必须帧数完全相同，任何缺失、OOM、运行时/质量序列名不一致都直接失败。
2. 每个数据集/方法输出逐序列 mean、median、sample std 和 10000 次 paired bootstrap 95% CI，bootstrap seed 固定为 0。该统计使用“序列等权”；Stage 4A 的 `result_scale.json` 继续保留为官方有效像素加权总指标，二者不能混称。
3. 对四方法的所有两两组合，在 AbsRel、RMSE、δ1、inference time 和 peak allocated 上输出 paired advantage CI 与 win/tie/loss。只有 CI 完全位于零的一侧才能写“显著更优”；否则写“aggregate 更好但 paired CI 未排除零”。
4. AbsRel 和 δ1 分别计算逐数据集及全部 41 段的 normalized regret、oracle wins；Pareto 使用 AbsRel、peak allocated 和 total inference time 三维共同判断。
5. Stage 1 A6000 结果只单列为历史复现，不混算硬件指标；不新增 uniform、K、bank 或 attribution 推理。

集群运行：

```bash
sbatch run.sh
```

该作业依次对现有 12 个 Stage 4A 输出目录执行纯 `eval_depth.py --align scale`，不会调用 StreamVGGT 模型。输出为：

- `stage4b_video_depth_sequence_results.csv`：164 条质量—运行时配对记录。
- `stage4b_video_depth_statistics.csv`：逐数据集/方法/指标的描述统计和 bootstrap CI。
- `stage4b_video_depth_paired_comparison.csv`：两两 paired CI 与 win/tie/loss。
- `stage4b_video_depth_regret.csv`：逐域和全域 normalized regret。
- `stage4b_pareto.csv`：质量—allocated—时间 Pareto。

Stage 4B-VD 完整生成后，不根据显著性重新调 selector：K4 继续作为 VideoDepth 主方案，old K6 继续作为跨任务 robust default，K8 继续作为 specialist。若 paired CI 不支持“优于”，论文措辞降级为“匹配/保持”；只有输出覆盖不完整才补跑纯评价。随后统一 pose、静态/动态 reconstruction 和 long-sequence 已有结果，完成跨任务最终主表，再进入 Stage 4C。

### Stage 4B-VD 结论

- 41 条序列 ×4 方法共 164 条记录完整。K4 是 Bonn、KITTI、Sintel 三域唯一同时位于 AbsRel—allocated—时间 Pareto 前沿的 bounded 方法。
- K4 相对 full 的 KITTI AbsRel 显著更好；Bonn 和 Sintel 的 paired CI 未排除零，因此只能写“保持/匹配”，不能写显著提升。Bonn 的 full 在 δ1 上仍有约 0.33 个百分点的显著优势，需要保留为细粒度限制。
- 在 bounded-only 逐序列 oracle 下，K4 的 AbsRel 平均 regret 为 2.41%，31/41 段获胜；old K6 为 10.52%，K8 为 16.73%。K4 的最坏 regret 仍达到 59.86%，主要失败场景包括 Sintel `mountain_1` 和 `temple_2`，因此必须继续做冻结后的真实长序列验证，而不能宣称所有序列占优。

### Stage 4B-X：跨任务角色与 claim audit

本部分不运行模型，也不修改 selector。脚本直接汇总 Stage 4B-VD、Stage 3.3/3.7 pose、Stage 3.7 逐序列 reconstruction archive、Stage 3.6B 长序列 gate 和 Stage 4A 候选资格：

```bash
python scripts/summarize_stage4b_cross_task.py
```

不需要 GPU 或 Slurm，因此根目录 `run.sh` 保持不变。输出为：

- `stage4b_cross_task_summary.csv`：40 个核心任务—数据集—方法单元，加 1 条 K8 长序列平台证据。
- `stage4b_cross_task_regret.csv`：VideoDepth、pose、静态/动态 reconstruction 的 bounded-only 逐单元、数据集、任务和跨任务 regret。
- `stage4b_method_roles.csv`：冻结后的四方法角色及其资格、胜场、regret 和资源证据。
- `stage4b_claim_audit.csv`：论文中允许/禁止使用的措辞，以及进入 Stage 4C 的 gate。

证据粒度必须明确区分：VideoDepth 和 reconstruction 使用共同覆盖的逐序列结果；当前 pose 本地只保存完整数据集汇总，因此 pose 行标记为 `dataset_aggregate`，不得伪称逐序列显著性。跨任务核心覆盖固定为：

1. VideoDepth：Bonn、KITTI、Sintel。
2. Pose：ScanNet、Sintel、TUM。
3. 静态 reconstruction：7-Scenes、NRGBD、ETH3D。
4. 动态 reconstruction：TUM。
5. 共 10 个 benchmark ×4 方法 = 40 个核心汇总单元；缺少任一单元即停止角色冻结。

角色决策采用“门槛优先”，不能由事后平均值反向改写：

1. 先应用 Stage 3.7/Stage 4A 的预注册资格；失败者不能竞争统一主方案。
2. 在 Stage 4A 合格的 bounded 方法中，以 10 个 benchmark 主指标的 bounded-only oracle 胜场决定 primary，宏平均 regret 只作平局处理。
3. 任务级 Pareto、逐序列 regret 和最坏 regret 用于描述适用范围和备选角色。
4. 不同任务的 AbsRel、ATE 和 Overall 即使都归一化，也仍是异质指标。跨任务宏平均只能作风险摘要，不能单独作为方法选择 gate。

### Stage 4B-X 决策门槛与结论

进入 Stage 4C 必须同时满足：

1. 40/40 个核心单元覆盖完整，VideoDepth/reconstruction 的方法间序列集合一致。
2. K4 在全部 10 个 benchmark 上 allocated 均低于 full，且 Stage 4B-VD 三域 Pareto 和 paired claim 审计通过。
3. Stage 3.6B 1000 帧有界 streaming gate 保持 PASS。
4. 方法角色不推翻既有 gate：Stage 4A 不合格者只能保留 specialist 身份。

实际结果全部满足：

- K4 和 old K6 是 Stage 4A 唯二合格的 bounded 候选。K4 获得 7/10 个 benchmark 主指标 oracle 胜场，old K6 为 1/10，因此冻结 K4 为 `primary_bounded_deployment`。
- old K6 在静态 reconstruction 的任务级平均 regret 为 13.40%，低于 K4 的 26.51%，继续冻结为 `robust_bounded_alternative`；这不代表它是 VideoDepth Pareto 方法。
- temporal K8 获得 2/10 个 benchmark 胜场且 pose 平均 regret 最低，但已经失败 Stage 3.7 geometry 与 Stage 4A KITTI gate，只能冻结为 `long_sequence_pose_specialist`。
- K8 的朴素跨任务宏平均 regret 为 13.71%，低于 K4 的 15.21% 和 old K6 的 20.18%。该结果作为异质任务汇总的局限如实报告，不能据此把已失败预注册门槛的 K8 提升为统一默认。
- claim audit 的覆盖、K4 VideoDepth default、K4 cross-task primary、old K6 robust alternative 和 Stage 4C readiness 均为 PASS；K8 specialist 为 `PASS_LIMITED`，DINO 因缺少 K4/K8 同 K 非 DINO 因果对照保持 `LIMITED`。

达成门槛后的动作是只携带 K4、old K6 和 temporal K8 进入 Stage 4C；full cache 作为质量/资源参考运行到成功上限。不得重新搜索 K、temporal bin、DINO 阈值或按 Stage 4C 结果调参。若 Stage 4C 暴露失败，只报告失败域并收缩 claim。

## Stage 4C：冻结后的未见长序列验证（已完成）

本阶段不再复用参与过 selector 开发或前序回测的 Bonn、7-Scenes test、TUM dynamics 序列，固定使用三条此前未进入本项目实验矩阵的 TUM RGB-D 原始真实长序列：

1. `rgbd_dataset_freiburg1_room`：办公室大范围闭环轨迹。
2. `rgbd_dataset_freiburg2_desk`：多桌面、包含多次闭环。
3. `rgbd_dataset_freiburg3_long_office_household`：办公室/家庭环境长闭环。

每条序列按 TUM 官方 RGB timestamp 与 mocap ground truth 最近邻关联，最大时间差固定 0.02 秒；使用前 100/250/500/1000 个有效关联帧。该阶段评价真实长序列 pose 和系统资源，不重复 Stage 4A/B 已完成的 VideoDepth 像素指标。

下载与检查在登录节点运行：

```bash
bash scripts/download_stage4c_tum.sh
python scripts/check_stage4c_data.py --root data/eval/stage4c_tum
```

下载脚本只从官方 archive 解出本阶段需要的 `rgb/`、`rgb.txt` 和 `groundtruth.txt`，不解压未使用的 depth。空间紧张时可设置 `STREAMVGGT_STAGE4C_DELETE_ARCHIVES=1`，让每个 archive 解出后立即删除；否则保留压缩包以支持复用。

正式矩阵为：

1. `stage3_2_k4`、`old_dino_k6`、`temporal_binned_dino_k8` 必须完整运行三序列 ×四长度，共 36 个 bounded run。
2. `full_cache` 使用相同逐帧输入和 sink 输出，只让 cache 本身保持无界；每条序列从 100 帧递增运行，首次 OOM/失败写入 JSON 后停止更长 prefix，并报告最大成功长度。
3. 每个 prefix 独立进程运行，保证 500 与 1000 帧的 CUDA peak 和 CPU RSS 可直接比较。不得用一次 1000 帧运行的最终 peak 冒充四个 prefix。
4. cache 配置完全冻结：K4=`anchor_recent_dino_diverse_2old_1recent/4`，old K6=`anchor_recent_dino_diverse/6`，temporal K8=`temporal_binned_dino_k8/8`。

正式提交只使用根目录入口，且保留既有 Conda 激活方式：

```bash
sbatch run.sh
```

本地 smoke 可以只运行单序列、10/20 帧和 K4，并写入独立目录；它不通过 `sbatch`，也不产生正式 gate：

```bash
env STREAMVGGT_STAGE4C_METHODS="stage3_2_k4" \
    STREAMVGGT_STAGE4C_SEQUENCES="rgbd_dataset_freiburg1_room" \
    STREAMVGGT_STAGE4C_LENGTHS="10 20" \
    STREAMVGGT_STAGE4C_RESULTS_ROOT="$PWD/eval_results/stage4c_smoke" \
    STREAMVGGT_STAGE4C_SKIP_GATE=1 \
    bash run_stage4c.sh
```

正式输出为：

- `stage4c_results.csv`：逐方法、序列和长度的 pose、吞吐、CUDA/RSS、streaming 语义、cache trace 和运行环境。
- `stage4c_gate.csv`：三个 bounded 角色的逐项决定，以及 full cache 每条序列的最大成功/首次失败长度。
- `eval_results/stage4c_tum_long/<method>/<sequence>/<frames>/`：`stage4c_metrics.json`、小型 trajectory 和 memory trace。

### Stage 4C 决策门槛

每个 bounded 方法分别判断，必须同时满足：

1. 12/12 个方法内 run 完整成功，processed frames 与请求长度严格相同；所有成功结果来自同一 GPU 型号且必须包含 `6000 Ada`。
2. 输入必须为 `streaming`、输出必须为 `sink`，retained outputs/views 都为 0；cache window/policy 必须与冻结配置完全一致。
3. 三序列所有 prefix 的 peak allocated <12288 MiB。
4. 每条序列 1000 相对 500 帧的 peak allocated 增长 ≤256 MiB；CPU RSS peak 增长也 ≤256 MiB。
5. 全部 pose evaluation 成功。对每个序列/prefix，以三个 bounded 方法中较优值为 oracle，ATE 和旋转 RPE 都不得超过 oracle 的 2 倍；这是灾难保护而非新的方法选择规则。
6. full cache 至少在每条序列成功完成 100 帧；其 OOM/失败是资源参考，不作为 bounded gate 失败，也不得删去失败记录。

### 达成门槛后的动作

- K4 PASS：确认最终 `primary_bounded_deployment` 具备未见真实长序列的端到端有界性，进入 Stage 4D 论文资产与最终表格。
- old K6 PASS：保留 `robust_bounded_alternative`；失败则把该角色收缩到 Stage 3.3/3.7 已验证的重建范围，不用新参数补救。
- temporal K8 PASS：保留 `long_sequence_pose_specialist`；失败则撤销 specialist claim，但不反向影响 K4 的主方案结论。
- 任一方法只因 pose 2×保护失败：其内存有界结论仍可单独报告，质量 claim 标记为 limitation。
- 任何结果都不得触发新的 K、bank、DINO threshold、序列抽样或按 held-out 数据调参；失败只用于收缩适用范围。

### Stage 4C 结论

- 三个 bounded 方法全部完成 3 序列 ×4 长度，共 36/36 个 run；均保持 streaming input、sink output 和冻结后的 cache 配置。full cache 三条序列都只成功到 100 帧，并在请求 250 帧、实际处理到约 195 帧时 OOM。
- K4、old K6、temporal K8 的最大 peak allocated 分别为 8026.32、8406.07、8782.88 MiB；每个方法从 500 到 1000 帧的 GPU 增量都为 0 MiB，CPU RSS 最大增量分别仅 0.68、1.66、4.35 MiB。因此三组的系统有界性通过，且明显区别于 full cache。
- 三组的正式候选 gate 都只因 pose catastrophe 失败，而不是覆盖、显存或流式语义失败。K4/old K6 在 `freiburg1_room/250` 的 rotation RPE 分别达到 bounded oracle 的 13.36/10.91 倍；temporal K8 在 `freiburg3_long_office_household/500` 的 ATE 达到 oracle 的 2.34 倍。
- 12 个序列—长度单元中，K4 获得 8 个 ATE 最优；temporal K8 则获得 11 个 translation RPE 最优和 11 个 rotation RPE 最优。该结果说明 K4 更擅长全局轨迹位置，K8 更擅长局部运动，但直接用 K8 pose 并不自动保留 K4 的 ATE。
- 因此保留 K4 的 `primary_bounded_deployment` 系统角色，但必须将真实超长序列 pose 写成明确限制；temporal K8 仍只作为 pose specialist。为判断两种优势是否能在不重新训练、不调 selector 的情况下组合，先插入 Stage 4E-A 的小型离线筛查；Stage 4D 暂不扩展新实验。

## Stage 4D：定性结果、失败分析与论文资产（待 Stage 4E-A 决策）

固定选择成功与失败场景，生成点云、depth error、retained-frame 时间线和 memory/quality 曲线。重点包括 K8 成功的 7-Scenes/ETH3D、失败的 `grey_white_room`、`walking_halfsphere`、`sitting_rpy`，以及 K8 逆转成功的 `walking_rpy`、`staircase`。最终同步生成主表、消融表、运行稳定性表和可复现实验清单。

## Stage 4E-A：K4/K8 离线可组合性筛查（当前阶段）

本阶段只回答一个窄问题：能否让 K4 保留较好的全局位置/geometry，同时使用 temporal K8 较好的局部姿态信息。它不是新的模型主实验，也不声称已经实现在线双分支；直接复用 Stage 4C 保存的 K4/K8 trajectory，不重新运行网络、不使用 GPU、不调整 K、DINO 或 temporal bank。

实验固定使用 Stage 4C 的三条 TUM 长序列和 250/500/1000 三个 prefix，共 9 个序列—长度单元。100 帧只用于 full-cache 吞吐参考，不放进融合质量门槛。先重新评价保存的 K4/K8 trajectory，并要求与 `stage4c_results.csv` 的 ATE/RPE 数值误差不超过 `1e-5`，以排除拿错轨迹或 evaluator 漂移。

固定比较两个无需训练、无需 ground truth 的候选：

1. `direct_k4_geometry_k8_pose`：geometry/depth 角色仍归 K4，但最终相机 trajectory 直接采用 K8。它的 pose 指标应精确复现 K8，是最简单的上下界和一致性检查；Stage 4E-A 尚不重新生成 K4 点云，因此不能用这一行声称完成了跨 head 的 3D 一致性验证。
2. `component_k4_translation_k8_rotation`：保留 K4 camera centers/translation，使用 K8 rotation；只用两条预测轨迹第一帧的相对旋转把 K8 orientation 放入 K4 初始 gauge，全程不看 ground truth、不学习融合权重。它用于直接测试“K4 全局位置 + K8 局部旋转”是否真的可组合。

这里不尝试只给 K4 camera head 扩大 cache。Stage 3.5B 已显示 K4 coupled 与“K4 aggregator + full camera cache”的 ATE 分别为 0.6666/0.6668、rotation RPE 为 41.67°/41.24°，说明主要上下文瓶颈不在 camera KV。若后续在线实现，必须明确维护两套 aggregator state；不能把 camera-only cache 冒充 K8 pose 分支。

集群正式运行只提交根目录入口：

```bash
sbatch run.sh
```

该作业是 CPU 作业，读取服务器上既有的：

```text
eval_results/stage4c_tum_long/<method>/<sequence>/<frames>/trajectory.npz
```

输出为：

- `stage4e_a_results.csv`：两候选的均值、最坏 ratio 和资源代理摘要。
- `stage4e_a_sequence_results.csv`：两候选 ×9 单元的 K4/K8 基线、融合后 ATE/RPE，以及资源代理量。
- `stage4e_a_gate.csv`：逐候选的质量、资源代理和 Stage 4E-B 资格。
- `eval_results/stage4e_a_pose_fusion/`：融合后 trajectory，供失败定位和后续实现核对。

若只想在服务器上检查一个单元，可在已激活 `StreamVGGT` 环境后直接运行，不能把该结果作为正式 gate：

```bash
env STREAMVGGT_STAGE4E_A_SEQUENCES="rgbd_dataset_freiburg1_room" \
    STREAMVGGT_STAGE4E_A_LENGTHS="250" \
    bash run_stage4e_a.sh
```

### Stage 4E-A 决策门槛

每个候选独立判定，必须同时满足：

1. 9/9 个单元成功，K4/K8 重算指标与 Stage 4C 的最大绝对误差 ≤`1e-5`。
2. 每个单元的融合 ATE ≤同单元 K4 ATE ×1.10；不能用跨序列平均掩盖一个长序列崩溃。
3. 每个单元的 translation RPE 和 rotation RPE 分别 ≤同单元 K8 对应指标 ×1.10。
4. 两次 Stage 4C 推理顺序执行的 peak proxy，以及 `K8 peak + K4 aggregator KV` 的一阶在线 peak projection 都必须 <12288 MiB。
5. 每个单元按 `K4 inference_sec + K8 inference_sec` 得到的 dual-FPS proxy，都必须高于三条 full-cache 100 帧 run 的平均 FPS。

第 4、5 项只用于决定是否值得实现。它们来自两次既有运行的代理计算，不是在线双分支的真实显存/吞吐结果，也不能进入论文系统性能主表。

### 达成门槛后的动作

- `component_k4_translation_k8_rotation` 通过：进入 Stage 4E-B，只实现这一种冻结的双 aggregator 在线分支，并在相同三序列上实测显存/吞吐/pose；随后补最小 reconstruction consistency 回测。Stage 4E-B 必须重新满足 <12288 MiB、长度平台和正式质量门槛，才能进入最终方法。
- 只有 `direct_k4_geometry_k8_pose` 通过：Stage 4E-B 走更简单的 K4 geometry + K8 pose 输出，但必须优先验证相机—深度/点云坐标一致性；若 reconstruction 退化则立即停止。
- 两者都通过：优先实现 component 版本，因为它保留 K4 的全局位置；direct 仅作消融。
- 两者任一质量门槛失败，不得修改融合权重、按序列路由或用 held-out 结果调参；两个候选都失败则停止 Stage 4E 方法增强，直接进入 Stage 4D，以 K4 主方案、old K6 robust alternative、K8 pose specialist 的现有结论投稿。

若 Stage 4E-B 在线实测通过，Stage 4E-C 才考虑不含数据集名称的在线路由和 leave-one-dataset-out 验证，并沿用此前预注册的增强门槛：相对 old K6 平均 normalized regret 至少下降 20%、最大 regret 不超过 60.1%。未达到就不把路由写成最终方法。
