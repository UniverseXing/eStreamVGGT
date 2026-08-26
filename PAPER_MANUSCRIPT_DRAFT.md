# 论文正文草稿

> 本文件只存放可直接翻译并进入 LaTeX 稿件的正文。写作解释、取舍理由、
> 禁止性表述和扩展材料继续保存在 `PAPER_WRITING_GUIDE.md`。

## 5.1 Abstract

面向持续视频流的前馈三维感知需要同步恢复相机运动与场景几何，但
StreamVGGT 保留全部历史 key--value（KV）状态及稠密输出时，内存会随序列
长度持续增长，因而难以在固定资源上长期运行。本文提出一种无需重新训练的
DINO-guided bounded-cache 方法：利用 DINOv2 帧描述符、稳定锚点、近期上下文
和分层时间槽位构造 K4/K6/K8 历史策略，以同一帧索引耦合裁剪 aggregator 与
camera-head KV，并通过逐帧输入和 streaming release 及时释放稠密输出。我们在
Bonn、Sintel 和 KITTI 上使用 AbsRel、$\delta_1$、FPS 与峰值显存评价
VideoDepth，在 Sintel、ScanNet、TUM、7-Scenes、NRGBD、ETH3D 和 TUM
Dynamics 上使用 ATE、RPE 与 reconstruction Overall 评价跨任务表现，并在三条
未见 TUM RGB-D 序列上测试至 1000 帧。相较具有相同四状态预算的
Anchor+Recent-4，K4 在 Bonn、Sintel 和 KITTI 上分别将 AbsRel 降低
21.6\%、6.3\% 和 28.2\%，三个逐序列 paired-bootstrap 95\% 置信区间均排除
零。在 110 帧显存分解中，K4 相较 Full cache 将 peak allocated/reserved
memory 分别降低 57.9\%/80.3\%，并获得 $3.24\times$ 吞吐率；KV pruning
贡献其中 10.80 GiB 的 peak-allocated 节省，output release 额外节省
0.48 GiB。此外，Full cache 在三条长序列中均于约 195 个已处理帧后 OOM，
而 K4/K6/K8 均完成 1000 帧，并在 500 至 1000 帧之间保持零 GPU-peak
增量。这些结果说明，DINO 历史选择带来的改善并非仅来自小窗口，耦合裁剪与
streaming release 则共同解决了在线模型无法在固定内存下持续运行的问题。

## 5.2 Introduction

面向开放环境的自主系统必须把持续到达的二维视频转化为可定位、可测量和可交互
的三维世界表征\cite{zhuo2026streamvggt}；然而真实视频通常没有预先给定的
终点，边缘设备也无法无限保存历史观测和中间状态。从图像序列中恢复相机运动
与三维场景结构，是连接二维视觉观测和真实三维世界的基础问题
\cite{wang2025vggt,zhuo2026streamvggt}。这项能力支撑自动
驾驶、增强现实和具身智能等应用：相机运动确定每个观测的拍摄位置与朝向，使
跨时间图像能够在统一坐标系中对齐，并为定位、轨迹估计和回环关联提供基础；
场景三维结构则把二维像素恢复为空间中的深度、表面与遮挡关系，使导航规划、
碰撞规避、虚实注册和机器人交互能够利用真实空间约束。上述系统接收的通常不是
预先收集好的短图像集合，而是持续到达的视频流。实际部署因此不仅要求准确的
深度、相机和三维几何预测，还要求模型能够在每个新帧到达后低延迟地更新结果，
并在未知序列长度下保持可控的计算和内存开销。

传统 Structure-from-Motion（SfM）和 Multi-View Stereo（MVS）通过特征匹配、
几何验证、三角化和全局优化获得可靠重建，但其离线优化流程难以满足逐帧更新
需求；近期工作仍将多阶段处理和测试时几何优化视为其主要效率瓶颈
\cite{wang2024dust3r,wang2025vggt,murai2025mast3rslam}。NeRF 和 3D Gaussian
Splatting 等场景表征通常也需要针对每个场景进行迭代优化
\cite{kerbl2023gaussians}。近年的
学习式方法开始直接预测跨视图一致的点图或相机参数：DUSt3R 和 MASt3R 从图像
对建立稠密几何关系\cite{wang2024dust3r,leroy2024mast3r}，Fast3R 与 VGGT
进一步通过多视图全局交互实现前馈式联合预测\cite{yang2025fast3r,wang2025vggt}。
然而，全局交互模型在新帧到达时需要重新处理完整序列。StreamVGGT 通过时间
因果注意力和历史 key--value（KV）cache 将这一过程改为增量推理，使每个当前帧
能够复用过去状态\cite{zhuo2026streamvggt}。这一设计消除了重复计算，却默认
保留全部历史 KV；随着视频增长，常驻历史状态及逐帧输出仍会持续累积。因此，
“能够在线更新”尚不等同于“能够在固定资源下持续运行”。原论文还在 200 帧
序列上评估了 50/100 帧 window 和 K-nearest recent cache，证明粗粒度裁剪能够
限制推理开销\cite{zhuo2026streamvggt}。2026 年出现的直接相关工作进一步从
token pruning/quantization、自适应 rolling cache、帧级一致性单元、时空压缩及
query--key 检索等角度构造有界历史
\cite{su2026xstreamvggt,yuan2026infinitevggt,lu2026ovggt,xu2026framevggt,
wang2026stac,zou2026retrievevggt,liu2026streamcachevggt}。本文研究其中一个不同
且更紧凑的设计点：在仅 4--8 帧预算内联合保留 DINO 语义差异与多尺度时间
角色，以统一帧索引同步裁剪 StreamVGGT 的双 KV 分支，并通过输出释放约束完整
执行路径，而不只压缩 attention cache。

本文研究的问题是在不重新训练或修改 StreamVGGT 预测 heads 的情况下，如何在
固定历史预算内保留最有价值的帧状态。简单的最近帧窗口虽然有界，却可能删除
建立全局参照所需的早期视图；长期固定首帧又可能与当前场景逐渐失配。为此，
我们提出 DINO-guided bounded cache。DINOv2 的自监督特征同时包含语义区分和
空间结构信息，可为历史视图的视觉差异提供稳定描述\cite{oquab2023dinov2}。
K4 和 K6 在固定预算内组合稳定 anchor、DINO-diverse history 与 recent
context，其中 K6 为近期连续帧分配更多槽位；K8 则进一步引入 near、middle 和
long temporal banks。帧选择同时作用于 aggregator 和 camera head 的 KV 状态，
以保持两个分支的历史一致性；逐帧输入和 streaming-release 则及时释放已消费的
稠密输出。由此，历史状态开销由序列长度相关的 $\Theta(T)$ 转化为预算相关的
$\mathcal{O}(K)$。

实验结果验证了该设计的资源可扩展性和任务边界。在相同四状态预算下，K4 相较
Anchor+Recent-4 在 Bonn、Sintel 和 KITTI 上分别将 AbsRel 降低 21.6\%、
6.3\% 和 28.2\%，且三域 paired-bootstrap 95\% 置信区间均排除零。在三条
held-out TUM RGB-D 长序列上，full cache 在请求 250 帧时均于约 195 个已处理
帧后发生 OOM，而 K4、K6 和 K8 均完成 1000 帧推理，并在 500 至 1000 帧之间
保持不变的 GPU peak。独立的四格显存实验进一步表明，KV pruning 是主要节省
来源，streaming release 则移除随输出数量增长的附加项。与此同时，较大的 K
并不构成普遍优势：K6 在部分静态重建任务上更稳健，K8 可改善长序列局部位姿，
却不保证更低的全局轨迹误差。这些结果说明，执行可扩展性与长期几何一致性是
相关但不同的目标。

本文的主要贡献如下：

1. 我们提出无需重新训练的 DINO-guided bounded-cache family。该方法复用
   StreamVGGT 已计算的 DINOv2 patch embeddings 构造帧级描述符，以候选帧对
   近期参考的最大余弦相似度衡量视觉冗余，再将固定 anchor、DINO-selected
   history、recent context 和分层时间槽位组合为 K4/K6/K8。相较于仅保留最近
   帧的 FIFO/window 策略，该设计在 4--8 帧的紧凑预算内同时保护长期参照、
   局部连续性和非冗余视图，且不增加图像重编码或模型训练。
2. 我们提出面向 StreamVGGT 双缓存结构的端到端有界执行机制。选择器使用同一组
   帧索引同步裁剪 aggregator 与 camera-head KV、描述符和帧 ID，避免两个预测
   分支引用不一致的历史；逐帧加载与 output-sink release 进一步阻止输入图像和
   深度、点图、置信度等稠密输出累积。该组合解决了“KV 已裁剪但完整程序内存仍
   随序列增长”的问题，将历史状态开销由 $\Theta(T)$ 限制为
   $\mathcal{O}(K)$。
3. 我们建立统一的 gate-first 跨任务与长序列评价协议，以 VideoDepth 质量、
   camera pose、静态/动态 reconstruction、速度和 GPU/CPU 内存共同检验方法，
   并在策略冻结后加入未见的 1000 帧真实序列。该评价不仅验证有界执行能力，
   还明确识别出 K4 的默认紧凑角色、K6 的静态重建备选角色和 K8 的局部位姿
   specialist 角色，揭示局部 RPE 改善并不必然转化为更低全局 ATE 的适用边界。

## 5.3 Related Work

### 5.3.1 Feed-forward 3D reconstruction

多视图重建的几何基础是利用相机投影把同一三维点在不同图像中的观测联系起来。
记第 $j$ 个三维点为 $\mathbf{X}_j$，其在第 $i$ 个相机中的像素观测为
$\mathbf{x}_{ij}$，内参与位姿分别为 $\mathbf{K}_i$ 和
$(\mathbf{R}_i,\mathbf{t}_i)$，则针孔投影和典型 bundle adjustment 可概括为

\begin{equation}
\widetilde{\mathbf{x}}_{ij}
\sim
\mathbf{K}_i
\left[\mathbf{R}_i\mid\mathbf{t}_i\right]
\widetilde{\mathbf{X}}_j,
\qquad
\min_{\{\mathbf{R}_i,\mathbf{t}_i,\mathbf{X}_j\}}
\sum_{(i,j)\in\mathcal{M}}
\rho\!\left(
\left\|\mathbf{x}_{ij}-
\pi(\mathbf{K}_i,\mathbf{R}_i,\mathbf{t}_i,\mathbf{X}_j)
\right\|_2^2
\right),
\label{eq:rw_multiview_geometry}
\end{equation}

其中 $\mathcal{M}$ 是跨视图匹配集合，$\pi(\cdot)$ 是投影函数，$\rho$ 是稳健
损失。传统 SfM 先估计对应关系和相机运动，再通过上述目标联合优化相机与稀疏
结构；MVS 随后利用已知相机将几何扩展为稠密表面。该范式具有明确的几何约束，
但匹配、验证和全局优化通常需要反复访问完整图像集合。

学习式前馈重建将上述对应、融合和回归过程吸收到参数化模型
$\hat{\mathcal{Y}}=F_{\theta}(\{I_i\})$ 中。DUSt3R 从未标定图像对直接回归共同
坐标系中的 point maps，MASt3R 进一步增强匹配与尺度相关能力
\cite{wang2024dust3r,leroy2024mast3r}。Fast3R 使用并行视图融合和
memory-efficient attention，VGGT 则通过多视图全局交互联合预测相机、深度、
point maps 和 tracking features\cite{yang2025fast3r,wang2025vggt}。
StreamVGGT 进一步把全局交互改造成时间因果计算，使这些输出可随视频逐帧更新
\cite{zhuo2026streamvggt}。本文以该前馈映射及其冻结的几何预测 heads 为基础，
不替换投影模型或重新学习三维表示；创新集中在支撑该映射的历史状态如何被选择
和执行。

### 5.3.2 Streaming and memory-efficient 3D perception

流式模型通过持久状态使当前帧复用过去信息。对于当前 query
$\mathbf{q}_t$ 和历史帧索引集合 $\mathcal{R}_{t-1}$，标准 cache attention
可写为

\begin{equation}
\mathbf{h}_t
=
\operatorname{softmax}\!\left(
\frac{\mathbf{q}_t
\mathbf{K}_{\mathcal{R}_{t-1}}^{\mathsf{T}}}{\sqrt{d}}
\right)
\mathbf{V}_{\mathcal{R}_{t-1}},
\label{eq:rw_cached_attention}
\end{equation}

其中 $\mathbf{K}_{\mathcal{R}_{t-1}}$ 与
$\mathbf{V}_{\mathcal{R}_{t-1}}$ 是所保留帧的 key 和 value，$d$ 是特征维度。
Full cache 令 $\mathcal{R}_{t-1}=\{0,\ldots,t-1\}$，能够访问全部上下文，但
状态数量及单步 attention 范围随 $t$ 增长。固定最近窗口则令
$\mathcal{R}_{t-1}=\{\max(0,t-K),\ldots,t-1\}$，可把状态限制在 $K$ 帧，
代价是所有早期参考都会按年龄被删除。因此，有界流式推理不仅是高效实现
attention，还包含一个决定 $\mathcal{R}_{t-1}$ 的信息保留问题。

不同三维流式模型采用了不同状态形式：Spann3R 使用可寻址 spatial memory，
CUT3R 通过 recurrent transformer 读写学习式 scene state，Point3R 引入
geometry-aligned spatial pointer memory
\cite{wang2024spann3r,wang2025cut3r,wu2025point3r}。StreamVGGT 不显式维护
优化式地图，而是缓存 aggregator 和 camera head 的因果 KV。其原论文进一步
使用 windowed streaming 和 K-nearest-frames caching：前者独立重建固定长度
片段后用相机外参对齐点云，后者使当前帧只访问最近 $K$ 帧
\cite{zhuo2026streamvggt}。FlashAttention-2 可以降低 attention kernel 的
访存与计算开销，却不改变必须常驻的历史帧数量\cite{dao2023flashattention2}。
2026 年的工作开始直接压缩 StreamVGGT/VGGT 类模型的长期状态。XStreamVGGT
结合 token-importance pruning 与 KV quantization；InfiniteVGGT 使用与
attention 实现解耦的自适应 rolling pruning；OVGGT 则以 FFN residual 评分并
动态保护坐标 anchor
\cite{su2026xstreamvggt,yuan2026infinitevggt,lu2026ovggt}。FrameVGGT 将每帧
新增 KV 作为一致的 memory unit，并维护互补帧单元与稀疏 anchor；STAC 结合
时间 token 评分、voxel-aligned 长期空间缓存和 chunk-causal 处理；
StreamCacheVGGT 使用跨层评分与保留、合并、删除相结合的 hybrid compression
\cite{xu2026framevggt,wang2026stac,liu2026streamcachevggt}。RetrieveVGGT 则
利用首个全局 attention layer 的 query--key 相似度、segment sampling 与
pose-aware spatial memory 构造固定预算上下文\cite{zou2026retrievevggt}。

这些方法说明，有界 cache、anchor 保护或帧级历史组织本身已不能作为本文的独有
贡献。本文选择的是不同的紧凑工作点：以已有 DINO tokens 形成帧级冗余信号，
在 4--8 帧预算内将固定 anchor、近期连续性和时间分层组合为可解释角色；同一
帧集合还同步约束 aggregator/camera KV，并配合稠密输出 release。因而本文的
比较重点是这种语义--时间帧选择与完整执行生命周期的组合，而非声称首次提出
有界 StreamVGGT cache。

### 5.3.3 Keyframes and semantic frame descriptors

关键帧选择的目标是在有限状态中保留足以约束相机和场景的观测。传统 SfM/SLAM
通常先检测局部兴趣点并用描述符建立跨图像对应，再通过本质矩阵、重投影误差或
视差检验剔除错误匹配。LightGlue 以自适应深度的学习式 matcher 提高局部特征
匹配的精度与效率\cite{lindenberger2023lightglue}；MASt3R-SLAM 则进一步结合
point-map matching、camera tracking、图构建、回环和全局优化，在实时系统中
联合恢复位姿与稠密几何\cite{murai2025mast3rslam}。这类方法可依据共视关系、
基线、跟踪质量或相机运动维护关键帧，其选择与式
\eqref{eq:rw_multiview_geometry} 中的显式几何约束紧密耦合。

DINOv2 则通过自监督学习产生兼具语义区分和空间结构的 patch tokens
\cite{oquab2023dinov2}。若第 $i$ 帧的 patch 表征为
$\{\mathbf{p}_{i,n}\}_{n=1}^{N}$，通用的全局描述可由池化和归一化构造：

\begin{equation}
\mathbf{d}_i
=
\frac{\frac{1}{N}\sum_{n=1}^{N}\mathbf{p}_{i,n}}
{\left\|\frac{1}{N}\sum_{n=1}^{N}\mathbf{p}_{i,n}\right\|_2},
\qquad
s(i,j)=\mathbf{d}_i^{\mathsf{T}}\mathbf{d}_j.
\label{eq:rw_frame_descriptor}
\end{equation}

$s(i,j)$ 提供帧级外观相似度，但它本身不等于特征对应、相机基线或几何约束。
本文正是以这一表征能力为基础进行改进：复用 StreamVGGT 已产生的 DINO tokens，
将相似度解释为 cache 中的视觉冗余，而不是建立新的 SLAM map；再把该语义信号
与不可丢失的 anchor、近期连续性和时间分层结合，决定式
\eqref{eq:rw_cached_attention} 下一时刻能够访问哪些整帧 KV 状态。

### 5.3.4 Long-sequence 3D evaluation

长输入能力在离线和流式模型中具有不同含义。离线模型能够一次处理 $T$ 帧，
并不意味着当 $T$ 未知且输入持续到达时，其常驻状态 $M(T)$ 具有固定上界。
因此，长序列评价至少应同时检查：是否完成目标帧数、峰值 GPU/CPU 内存是否随
$T$ 形成平台、吞吐率是否可接受，以及几何误差是否随时间恶化。对于对齐后的
参考轨迹位置 $\mathbf{p}_i$ 和预测位置 $\hat{\mathbf{p}}_i$，ATE 反映全局
累计误差，而相邻位姿增量的 RPE 反映局部运动误差，可概括为

\begin{equation}
\operatorname{ATE}
=
\min_{\mathbf{S}\in\operatorname{Sim}(3)}
\sqrt{\frac{1}{T}\sum_{i=1}^{T}
\left\|\mathbf{p}_i-\mathbf{S}(\hat{\mathbf{p}}_i)\right\|_2^2},
\qquad
\mathbf{E}^{\mathrm{rel}}_i
=
(\mathbf{P}_i^{-1}\mathbf{P}_{i+1})^{-1}
(\hat{\mathbf{P}}_i^{-1}\hat{\mathbf{P}}_{i+1}),
\label{eq:rw_pose_metrics}
\end{equation}

其中 RPE 分别从 $\mathbf{E}^{\mathrm{rel}}_i$ 的平移和旋转部分汇总。较低 RPE
并不保证较低 ATE，因为小的单步误差仍可沿序列累积。

Fast3R 展示了单次前馈处理千帧以上图像的能力，但其目标不是持续输入下的常驻
状态管理\cite{yang2025fast3r}。StreamVGGT 报告了逐帧延迟与显存增长、200 帧
pruning、长序列 reconstruction consistency 和模拟 loop closure 的 pose 评价
\cite{zhuo2026streamvggt}，其 pruning 主要采用 50/100 帧窗口。本文在此基础上
将同一组 4--8 帧策略冻结后用于 VideoDepth、camera pose、静态与动态
reconstruction，并在未见真实序列上联合检查 1000 帧完成率、GPU peak、CPU
RSS、ATE 和 RPE。这一设计同时验证“是否能够持续执行”和“保留下来的历史是否
仍支持三维预测”，而不在 checkpoint 与协议不一致时声称优于其他架构。

## 5.4 Method

第 5.3 节给出了本文方法的两个直接理论基础：StreamVGGT 通过式
\eqref{eq:rw_cached_attention} 的因果 KV attention 复用历史，而 DINOv2
描述符通过式 \eqref{eq:rw_frame_descriptor} 提供帧级视觉相似度。现有 full
cache 将所有历史索引加入 $\mathcal{R}_{t-1}$，最近窗口则仅按时间删除状态；
二者分别面临无界内存和长期参照丢失的问题。本文不改变式
\eqref{eq:rw_multiview_geometry} 所代表的相机—场景关系，也不重新训练
StreamVGGT backbone 或 prediction heads，而是对“哪些历史状态可被后续查询”
以及“这些状态在系统中存活多久”进行改进。

与近期依据内部 attention/FFN 重要性进行 token 压缩、合并或检索的方法不同，
本文复用基础模型已生成的 DINO 表征，在整帧层面显式组合语义差异和时间角色；
同时把历史选择从单一 attention cache 扩展到双 KV 分支与稠密输出生命周期。
因此，本文的理论改进并非重新定义式 \eqref{eq:rw_cached_attention}，而是为其中
$\mathcal{R}_{t-1}$ 引入一个可解释、固定容量且跨分支一致的构造规则。

具体而言，本文包含三项相互依赖的优化。第一，将时间截断替换为语义—时间联合
选择：DINO 相似度衡量视觉冗余，anchor、recent 和 temporal banks 分别约束
长期参考、局部连续性与跨尺度时间覆盖，由此得到 K4/K6/K8。第二，把一个帧的
aggregator KV、camera-head KV、描述符和 frame ID 视为不可分割的 coupled
retained state，并用同一索引集合裁剪，避免两个预测分支获得不一致历史。第三，
在有界 cache 之外采用逐帧加载和 streaming release，使已经消费的输入与稠密
预测不再随序列累积。前两项解决固定预算下的信息保留与状态一致性，第三项将
理论上的 cache 上界扩展为完整执行路径的内存上界。后续各小节依次给出问题
定义、DINO 描述符、三种预算策略、输出释放和复杂度分析。

![Method overview](paper_assets/figure1/fig_method_overview.svg)

**图 1. 方法总览。当前帧同时进入冻结的 StreamVGGT 推理分支和 DINO 描述符
分支；bounded selector 根据 anchor、recent、视觉差异与分层时间角色选择统一的
帧索引，并据此同步保留 aggregator/camera-head KV states。当前表示查询保留的
历史状态后产生相机与稠密三维输出，已消费输出经 prediction sink 写出并释放。**

## 5.5 Experimental Setup

### 5.5.1 Tasks and datasets

我们从预测质量、跨任务泛化和长序列可扩展性三个方面评价有界 cache。实验
覆盖 VideoDepth、相机位姿和多视图重建，并额外使用三条长 TUM RGB-D 序列
测试 100--1000 帧推理。数据集与用途汇总如下：

| Task | Datasets | Evaluation focus |
|---|---|---|
| VideoDepth | Bonn, Sintel, KITTI | Depth quality and cross-domain generalization |
| Pose | Sintel, ScanNet, TUM | Global and local camera motion |
| Static reconstruction | 7-Scenes, NRGBD, ETH3D | Multi-view geometric consistency |
| Dynamic reconstruction | TUM Dynamics | Dynamic-scene geometry |
| Long-sequence streaming | Three TUM RGB-D raw sequences | 100/250/500/1000-frame scalability |

KITTI 和三条 TUM RGB-D raw 序列均在 cache policy 冻结后加入：前者作为户外
VideoDepth 测试域，后者用于 held-out 长序列验证。其余数据集遵循各自既有的
预处理、有效深度区域和帧采样协议；完整序列清单及采样间隔放入补充材料。

### 5.5.2 Compared methods and implementation

我们比较 full cache 与三个冻结的有界配置 K4、K6 和 K8。四种方法共享同一
StreamVGGT checkpoint、预测 heads 和 $518\times518$ 输入分辨率，不进行
额外训练。Full cache 保留全部历史 KV；K4、K6 和 K8 分别最多保留 4、6 和
8 帧状态，具体选择规则见第 5.4 节。为隔离 DINO selection 的作用，
VideoDepth 还加入两个四状态非 DINO 对照：Recent-4 保留最近三个历史状态与
current，Anchor+Recent-4 保留 frame 0、最近两个历史状态与 current。二者与
K4 使用相同窗口、checkpoint、输入和评估顺序；Uniform-4 与三个随机种子的
Random-4 完整结果放入补充材料。所有 bounded runs 均采用耦合的
aggregator/camera cache；held-out 长序列实验进一步启用逐帧输入与
output-sink release，以避免稠密输出随序列累积。

主文中的资源比较统一在单张 NVIDIA RTX 6000 Ada GPU 上完成，并报告
inference FPS、CUDA peak allocated/reserved memory；长序列实验还记录 CPU
RSS。早期 A6000 测量不与该同卡主表混合。除明确标注的 OOM 外，同一数据集
上的所有方法使用相同输入、采样和评价代码。

为分解显存来源，我们还在 Bonn `person_tracking2` 的相同 110 帧输入上构造
$2\times2$ factorial：Full/K4 KV 与 accumulated/streaming-release outputs。
四组均逐帧加载输入；固定输出生命周期比较 Full 与 K4 得到 KV-pruning
贡献，固定 KV 策略比较 accumulated 与 release 得到 output-release 贡献。
同一 KV 策略的两种生命周期必须产生相同的 pose 和 depth prediction hashes。

### 5.5.3 Metrics and analysis protocol

VideoDepth 报告 AbsRel、SqRel、RMSE、log RMSE 及
$\delta_1$/$\delta_2$/$\delta_3$。Pose 使用 Sim(3) 对齐后的 ATE、translation
RPE 和 rotation RPE；ATE 衡量全局轨迹误差，RPE 衡量局部相对运动。重建
报告 accuracy、completeness、normal consistency (NC) 及
$\mathrm{Overall}=(\mathrm{Accuracy}+\mathrm{Completeness})/2$。资源指标包括
FPS、GPU peak memory 和 CPU RSS；长序列内存平台期通过比较 500 与 1000 帧
峰值进行检验。

VideoDepth 的方法比较以相同序列上的配对差值为单位，并使用 10,000 次配对
bootstrap（seed 0）计算 95\% confidence interval。序列等权统计与官方
valid-pixel-weighted aggregate 分开报告；当置信区间跨越 0 时，我们只称其
为保持或无清晰差异，不宣称显著改善。方法角色按照实验前确定的覆盖、质量、
内存和灾难退化门槛判定，而不是在观察最终平均结果后重新选择。由此，K4、K6
和 K8 分别作为 primary compact configuration、reconstruction-oriented
alternative 和 pose-oriented specialist 进行报告。

## 5.6 Results

### 5.6.1 同预算帧选择与 VideoDepth 折中

表 1 首先隔离了 DINO selection，而不是把 K4 只与无界 Full cache 比较。
Recent-4、Anchor+Recent-4 和 K4 均最多保留四个状态；其中
Anchor+Recent-4 与 K4 具有相同的 anchor、两个可替换历史槽和 current，唯一
差异是两个历史槽由近期帧还是 DINO 低冗余帧填充。

**表 1. 单张 RTX 6000 Ada 上的同预算 VideoDepth AbsRel（越低越好）。四个
AbsRel aggregate 均按有效像素数加权；最后一列为逐序列等权的
$d=\mathrm{AbsRel}_{A+R4}-\mathrm{AbsRel}_{K4}$ 均值及 95\% paired-bootstrap CI，
正值表示 K4 更好。Full cache 是原版 StreamVGGT。**

| Dataset | Full cache | Recent-4 | Anchor+Recent-4 | K4 | K4 advantage over Anchor+Recent-4 (95\% CI) |
|---|---:|---:|---:|---:|---:|
| Bonn | 0.0746 | 0.1210 | 0.0962 | 0.0755 | 0.0207 [0.0115, 0.0294] |
| Sintel | 0.3232 | 0.4000 | 0.3374 | 0.3161 | 0.0191 [0.0030, 0.0358] |
| KITTI | 0.1726 | 0.1316 | 0.1857 | 0.1334 | 0.0476 [0.0341, 0.0603] |

相较关键的 Anchor+Recent-4 对照，K4 在 Bonn、Sintel 和 KITTI 上分别将
valid-pixel-weighted AbsRel 降低 21.6\%、6.3\% 和 28.2\%；逐序列结果为
5/0、18/5 和 13/0 个 wins/losses，三个置信区间均排除零。因此，K4 的改善
不能仅用“小窗口”或“保留首帧”解释，而支持 DINO 历史选择在同预算下具有
可测收益。Uniform-4 上的三域优势同样清晰；相对三种子 Random-4，优势在 Bonn
与 Sintel 清晰，而 KITTI 无清晰差异，故本文不宣称 K4 对所有 selector 普遍
最优。完整 Uniform-4、Random-4、逐序列结果和 bootstrap 数据见补充材料。

Full cache 是原版 StreamVGGT 的无界参照。在 Bonn 上，K4 的 AbsRel 与其接近
（0.0755 vs. 0.0746），但 peak allocated memory 降低 51.2\%，吞吐率提高至
$2.48\times$；在 Sintel 上，K4 将 AbsRel 从 0.3232 降至 0.3161，同时降低
23.3\% 显存并获得 $1.24\times$ 加速；在 KITTI 上则将 AbsRel 从 0.1726
降至 0.1334、$\delta_1$ 从 0.7214 提升至 0.8282，同时降低 35.7\% 显存并
获得 $1.51\times$ 加速。Recent-4 在 KITTI 的加权 AbsRel 略低于 K4，但其
逐序列差异置信区间跨零；这一结果也限制了“任意数据域均最优”的表述。

### 5.6.2 跨任务表现与缓存角色分化

我们进一步在三个 pose、三个 static reconstruction、一个 dynamic
reconstruction 和三个 VideoDepth benchmarks 上比较冻结的 bounded
configurations。对于每个 task--dataset 单元，以 primary metric 最优的
bounded method 记为 oracle winner。K4 在 10 个单元中赢得 7 个，K6 和 K8
分别赢得 1 个和 2 个，表明不存在统一支配所有任务的 cache budget。Full
cache 仅作为无界 reference，不参与 bounded oracle 计数。

**表 2. Pose 与 reconstruction 结果（均为越低越好）。Pose 单元格依次报告
ATE / rotation RPE（度），reconstruction 单元格报告 Overall；每项粗体分别表示
该指标的最佳 bounded configuration。**

| Task | Dataset | Metric | Full | K4 | K6 | K8 |
|---|---|---|---:|---:|---:|---:|
| Pose | ScanNet | ATE / RPE$_{\mathrm{rot}}$ ($^\circ$) | 0.0346 / 0.384 | 0.1301 / 2.181 | 0.1381 / 2.173 | **0.0626** / **0.453** |
| Pose | Sintel | ATE / RPE$_{\mathrm{rot}}$ ($^\circ$) | 0.2326 / 0.711 | **0.3831** / 1.375 | 0.3939 / 1.054 | 0.3991 / **1.010** |
| Pose | TUM | ATE / RPE$_{\mathrm{rot}}$ ($^\circ$) | 0.0269 / 0.318 | **0.0250** / 0.355 | 0.0279 / **0.332** | 0.0317 / 0.333 |
| Static recon. | 7-Scenes | Overall | 0.0434 | 0.0607 | 0.0496 | **0.0478** |
| Static recon. | ETH3D | Overall | 0.7618 | 0.8842 | **0.7545** | 0.7639 |
| Static recon. | NRGBD | Overall | 0.0759 | **0.0714** | 0.0844 | 0.0906 |
| Dynamic recon. | TUM Dynamics | Overall | 0.0672 | **0.0695** | 0.0754 | 0.0807 |

角色差异在具体任务上十分明显。K8 在 ScanNet 上将 bounded ATE 从 K4 的
0.1301 降至 0.0626，并将 rotation RPE 从 $2.181^\circ$ 降至
$0.453^\circ$；但 K4 在 Sintel 和 TUM 上具有更低的 ATE，而这两个数据集
的最低 bounded rotation RPE 分别由 K8 和 K6 取得。Static
reconstruction 的最优方法同样随数据域改变：K8、K6 和 K4 分别在
7-Scenes、ETH3D 和 NRGBD 上最佳；K4 还在 TUM Dynamics 上取得最低
bounded Overall。这些结果说明，较大的 temporal budget 并不自动改善动态
几何，也不存在跨任务统一最优的窗口。

尽管 K8 的 post-hoc macro regret 为 0.137，低于 K4 的 0.152 和 K6 的
0.202，它在 KITTI VideoDepth 和部分 reconstruction gates 上失败，因此
不能据此取代 K4。按照预先设定的
gate-first protocol，K4 被确定为默认紧凑方案；K6 保留为 static
reconstruction regret 和 tail risk 更低的备选；K8 则用于强调局部 pose 的
场景。Macro regret 仅作为异构任务的风险摘要，而不构成统一最优性的证据。
完整的 translation RPE、NC、逐序列结果和 normalized regret 放入补充材料。

### 5.6.3 1000 帧长序列可扩展性

我们在三条 held-out TUM RGB-D raw 序列上测试 100、250、500 和 1000 帧
前缀。Full cache 在三条序列的 100 帧测试中均成功，但请求 250 帧时均在处理
约 195 帧后发生 OOM。相比之下，K4、K6 和 K8 完成了全部 36 个 bounded
runs，包括所有 1000 帧测试。三者在 1000 帧时的 peak allocated memory 分别
为 8026、8406 和 8783 MiB，且从 500 帧增加到 1000 帧时均未产生额外 GPU
peak；同期 CPU RSS 的最大增量仅为 4.35 MiB。结果表明，有界 cache 与逐帧
输出释放共同将历史状态和输出带来的 GPU/CPU 内存增长限制在常数范围内，使
StreamVGGT 能够处理 full cache 无法完成的长序列。位姿精度随序列长度的变化
将在第 5.6.5 节分析。

![Long-sequence resource scaling](paper_assets/figures/fig_long_sequence_scaling.png)

**图 2. 三条 held-out TUM RGB-D 序列上的长序列资源扩展。（a）三条序列中
每种方法的最大 peak allocated GPU memory；（b）平均 inference FPS。
Full cache 的红色叉号表示请求 250 帧时在实际处理约 195 帧后 OOM，而不是
一条成功的 250 帧测量。**

### 5.6.4 显存来源分解

图 3 在相同 110 帧输入和逐帧加载条件下进一步拆分 KV pruning 与 output
release。固定 streaming release 后，K4 将 Full cache 的 peak allocated
memory 从 19084.6 MiB 降至 8026.3 MiB，节省 57.9\%；peak reserved memory
从 44594 MiB 降至 8782 MiB，节省 80.3\%，吞吐率由 3.44 提升至 11.15 FPS
（$3.24\times$）。固定 accumulated outputs 时得到相近的 56.5\%
peak-allocated 降幅，说明该结果不依赖输出释放方式。

![Memory decomposition](paper_assets/figures/fig_stage5b_memory_decomposition.png)

**图 3. Bonn `person_tracking2` 110 帧上的显存分解。（a）Full/K4 与
accumulated/streaming-release outputs 的逐帧 CUDA allocated memory；（b）
streaming-release 条件下 KV pruning 的独立节省，以及 K4 条件下 output
release 的独立节省。**

KV pruning 在 streaming-release 口径下独立节省 11058.3 MiB（10.80 GiB）
peak allocated memory，是主要贡献；K4 条件下 output release 额外节省
492.0 MiB（0.48 GiB，5.8\%）。该差值与 accumulated 路径实际保留的
493.0 MiB 稠密输出一致。四个单元均完成 110 帧且输入模式均为 streaming；
同一 KV 策略的 accumulated/release pose 与 depth hashes 完全一致。因此，这一
factorial 只归因执行生命周期带来的资源差异，不把 Full 与 K4 的预测差异误写为
output-release 效果。图 2 展示完整系统能否扩展至 1000 帧，图 3 则解释显存
节省主要来自何处，两者提供互补证据。

### 5.6.5 全局轨迹与局部位姿的差异

有界内存并不保证位姿误差随序列长度保持稳定。在 1000 帧处，K4、K6 和 K8
的平均 ATE 分别为 0.776、0.808 和 0.820，而平均 rotation RPE 分别为
$9.88^\circ$、$8.08^\circ$ 和 $4.57^\circ$：K4 保留了较低的全局轨迹误差，
K8 则具有更准确的局部旋转。图 4 给出了三个代表性案例的完整 bounded 结果。

![Representative long-sequence pose errors](paper_assets/figures/fig_pose_case_comparison.png)

**图 4. 三个代表性长序列前缀上的位姿误差。（a）ATE；（b）translation RPE；
（c）rotation RPE。所有指标均为越低越好，（b）和（c）采用对数纵轴，黑色空心
轮廓标出每个案例的最佳 bounded configuration。F1-R、F2-D 和 F3-LO 分别表示
TUM 的 Freiburg1 room、Freiburg2 desk 和 Freiburg3 long office household，
斜线后的数字为 prefix 帧数。**

K8 在 F1-R/250 上同时改善全局与局部位姿；但在其余两个前缀
上，最低 ATE 由 K4 取得，最低 RPE 则由 K8 取得。最明显的
F3-LO/500 案例中，K8 的两个 RPE 均较低，但其
ATE 是 K4 的 $2.34\times$。因此，较低的单步相对运动误差不足以避免长期
累积漂移；K8 是局部位姿 specialist，而非 K4 的通用替代方案。

### 5.6.6 K4/K8 朴素输出融合的失败

基于 K4 的全局轨迹优势与 K8 的局部旋转优势，我们在九个长序列前缀上评估
了两种 post-hoc 融合。第一种保留 K4 的 geometry 输出，但直接采用 K8 的
camera poses；它本质上继承了 K8 的位姿结果，平均 ATE 为 0.424，高于 K4 的
0.406，最坏案例达到 K4 的 $2.34\times$。第二种进一步组合 K4 的 camera
translation 与 K8 的 rotation。该方案保持了 K4 的平均 ATE（0.406）和 K8
的平均 rotation RPE（$2.162^\circ$），但平均 translation RPE 为 0.0933，
明显高于 K8 的 0.0455，且最坏比例达到 $3.77\times$。尽管离线双分支资源
代理估算约为 9159 MiB 和 4.99 FPS（并非在线双分支实测），两种方案仍均未
通过质量门槛。这一负结果说明，相机中心与旋转在 SE(3) 轨迹中相互耦合，
不能通过独立挑选两个 configuration 的输出得到稳定改进。

## 5.7 Conclusion

本文针对 StreamVGGT 虽能逐帧更新、却因历史 KV 和稠密输出累积而无法在固定
资源上持续运行的问题，提出了 DINO-guided bounded cache、耦合状态裁剪和
streaming-release 执行路径。实验结果分别为 Introduction 中的三项贡献提供了
如下证据。

第一，实验支持了语义—时间联合历史选择在紧凑预算下的有效性。相较具有相同
anchor、两个历史槽和 current 的 Anchor+Recent-4，K4 在 Bonn、Sintel 和
KITTI 上分别将 AbsRel 降低 21.6\%、6.3\% 和 28.2\%，三个逐序列
paired-bootstrap 95\% 置信区间均排除零；相对 Uniform-4 也在三域取得清晰
优势。尤其在 KITTI 上，K4 将原版 StreamVGGT Full cache 的 AbsRel 从
0.1726 降至 0.1334，将 $\delta_1$ 从 0.7214 提升至 0.8282，同时把峰值显存
从 12.43 GiB 降至 8.00 GiB、速度从 5.99 FPS 提升至 9.06 FPS。这些结果说明，
复用已有 DINO 表征并将视觉低冗余与 anchor/recent 角色结合，能够在无需重新
训练或重编码历史图像的条件下带来超出简单小窗口的可测收益。

第二，长序列结果验证了耦合 KV 裁剪与输入输出释放确实形成端到端内存上界。
Full cache 在三条未见 TUM RGB-D 序列的 100 帧运行中成功，但请求 250 帧时均
在约 195 个已处理帧后 OOM。相比之下，K4、K6 和 K8 完成了全部 36 个 bounded
runs，包括所有 1000 帧序列；其 1000 帧 peak allocated memory 分别为
8026、8406 和 8783 MiB，而且三个配置从 500 帧增加到 1000 帧时 GPU peak
增量均为 0 MiB，CPU RSS 最大增量仅为 4.35 MiB。该平台现象与
$\mathcal{O}(K)$ 历史状态分析一致，说明只裁剪单个 attention cache 并不足够，
同步维护 aggregator/camera 状态并释放已消费的稠密输出才解决了完整程序随
序列增长的问题。110 帧四格分解进一步给出直接机制证据：固定 streaming
release 后，K4 相较 Full cache 将 peak allocated/reserved memory 分别降低
57.9\%/80.3\%，并获得 $3.24\times$ 吞吐率；其中 KV pruning 独立节省
10.80 GiB peak allocated memory，而 output release 再节省 0.48 GiB。由此，
KV pruning 是主要显存贡献，output release 则消除随稠密输出数量增长的附加项。

第三，统一的 gate-first 评价明确了固定预算方法的适用范围，而不是只报告一个
有利任务。K4 被支持为默认紧凑配置，K6 在部分静态 reconstruction 数据上构成
更稳健的备选，K8 则体现出局部 pose 优势。在 1000 帧处，K4、K6 和 K8 的平均
ATE 分别为 0.776、0.808 和 0.820，而平均 rotation RPE 分别为
$9.88^\circ$、$8.08^\circ$ 和 $4.57^\circ$。因此，K8 更低的局部旋转误差并未
转化为最低的全局轨迹误差；K4/K8 的 post-hoc 输出融合也未通过质量门槛。这些
证据验证了本文评价设计的必要性，并揭示出“执行内存有界”和“长期几何一致”是
两个相关但不能相互替代的目标。

本研究仍有两项主要限制。其一，不存在适用于全部任务的统一 cache budget；
不同任务对长期参照、近期运动和视角多样性的需求不同。后续可在保持严格内存
上界的前提下，研究由任务、置信度或场景变化驱动的动态预算与在线路由。其二，
有界执行不能阻止局部位姿误差沿时间累积为全局漂移。未来可将帧选择与 pose
graph、回环约束或相对运动一致性联合建模，使保留状态同时满足视觉代表性和
长期几何约束。

综上，本文用结构化的 4--8 帧历史选择和完整的张量生命周期管理，回应了流式
前馈三维模型“能够在线更新但不能固定内存长期运行”的关键缺口。结果既证明了
该方法在严格资源上界下维持深度、位姿和重建能力的可行性，也给出了不同 cache
预算的任务角色与失败边界，为进一步研究同时具有执行可扩展性和长期几何一致性
的流式三维基础模型提供了可复现基线。
