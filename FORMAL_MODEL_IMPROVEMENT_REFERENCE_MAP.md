# 正式模型改进思路、文献与实现对应说明

> 整理日期：2026-08-10  
> 当前正式模型：Phase 3G `GenerativeInnovationFIRController`  
> 原正式模型：Phase 1 P1-E2 `TimeDomainANC`

本文整理当前正式模型相对原正式模型的改进动机、参考文献、公开实现、本项目中的具体改造和实验依据。文中的“参考”不等于逐行复现：有些工作提供了控制思想，有些提供了训练方法或数据构造方法，最终结构还受到本赛题离线推理、参数量、实时性和官方 v6 评分规则的约束。

## 1. 模型版本与结论

### 1.1 原正式模型：Phase 1 P1-E2

原正式模型是一个 42,764 参数的因果时域 TCN `TimeDomainANC`。Phase 1 没有改变网络结构，而是把训练和选模目标对齐到官方协议：完整处理 `0.5 s` 初始化和六个 `0.5 s` 计分窗口，使用官方三分之一倍频程主指标和高频反弹定义训练与验证。

- 模型实现：[model.py](legacy_models/phase0_phase1/model.py)
- 可微 v6 指标：[v6_metrics.py](v6_metrics.py)
- 训练入口：[train_phase1.py](legacy_models/phase0_phase1/train_phase1.py)
- 固定官方验证：[phase1_validation.py](phase1_validation.py)
- 最优 checkpoint：[best_official_composite.pt](runs/phase1_suite_seed2026/P1-E2/checkpoints/best_official_composite.pt)

十路径结果为 `S=10.9790 dB`、`R=1.7392 dB`、`C=7.1635`，最差路径为 `0.9733 dB`。它能够取得较好的平均分，但网络没有显式表示次级路径，面对未建模路径时缺少可解释的识别和控制核泛化机制。

### 1.2 当前正式模型：Phase 3G

当前模型由三部分组成：

1. 解析创新误差路由器从闭环观测中判断当前路径与八个已知路径模板的相容程度；
2. 八个冻结的 2048-tap FIR 专家提供稳定的路径特定控制基底；
3. 一个离线训练、部署时冻结的轻量 GRU 生成 16 维残差激活，使有效 FIR 能够离开专家凸包。

有效控制核为：

\[
W_t=\sum_{i=1}^{8}\alpha_{t,i}W_i+z_tB,
\]

其中 `W_i` 是冻结专家，`alpha` 是解析创新路由权重，`B` 是 `16 × 2048` 的离线学习残差字典，`z_t` 是 GRU 每 240 点更新一次的有界激活。模型共有 41,552 个可训练参数，部署后全部冻结。

- 核心模型：[phase3g_model.py](phase3g_model.py)
- 正式 checkpoint：[best_phase3g_selection.pt](runs/phase3g_suite_seed2027/P3G-E2/checkpoints/best_phase3g_selection.pt)
- 正式模型清单：[phase3g_formal_model.json](artifacts/phase3g_formal_model.json)
- 提交导出：[export_phase3g_submission.py](export_phase3g_submission.py)
- 提交运行时：[phase3g_submission_runtime.py](phase3g_submission_runtime.py)

十路径结果为 `S=17.9585 dB`、`R=1.6870 dB`、`C=12.0648`，最差路径为 `3.9775 dB`。相对 P1-E2，主指标提高 `6.9795 dB`，反弹降低 `0.0522 dB`，综合分提高 `4.9013`，最差路径提高 `3.0043 dB`。

## 2. 改进思路与资料对应总表

| 原模型问题 | 当前采用的改进 | 主要资料依据 | 本项目的关键改造 | 代码与实验落点 |
|---|---|---|---|---|
| 训练目标与官方六窗口、三分之一倍频程、高频反弹不完全一致 | 可微复现官方 v6 指标并据此训练、验证、选模 | Participant Kit v6 是评分真值；Deep ANC 提供端到端监督式 ANC 背景 | 六窗独立去直流和 STFT，官方频带聚合，硬最大反弹，固定 manifest | [v6_metrics.py](v6_metrics.py)、[train_phase1.py](legacy_models/phase0_phase1/train_phase1.py)、[PHASE1_V6.md](PHASE1_V6.md) |
| 单个 TCN 隐式承担全部路径变化，路径行为难解释 | 建立路径特定的 FIR 专家集合 | Nam & Elliott 的多模型 ANC；GFANC 的预训练固定滤波器组合 | 八个 2048-tap 专家分别针对路径 1～8；专家在正式推理中冻结 | [phase3_model.py](phase3_model.py)、[train_phase3.py](legacy_models/phase3/train_phase3.py)、[PHASE3_FEEDBACK_FIR.md](PHASE3_FEEDBACK_FIR.md) |
| 10 维统计加 GRU 的路径分类准确率低 | 直接用候选模型的创新误差进行解析路由 | 多模型 ANC；Hu 等人的无附加探测次级路径辨识分析 | 对每个候选计算 `e+S_i*y` 所恢复的扰动与 `P_i*x` 模板之间的归一化谱误差；EWMA 和软权重替代硬分类 | [phase3r_templates.py](phase3r_templates.py)、[phase3r_model.py](phase3r_model.py)、[PHASE3R_INNOVATION_ROUTING.md](PHASE3R_INNOVATION_ROUTING.md) |
| 创新路由已可靠，但离散专家的凸组合对未知路径插值和外推不足 | 在专家混合上生成低秩 FIR 残差 | GFANC；Unsupervised-GFANC；E2E-CFG | 不直接生成完整 2048 维控制核，而用 16 维字典残差降低参数量和块更新成本；条件输入改为闭环创新证据 | [phase3g_model.py](phase3g_model.py)、[phase3g_closed_loop.py](phase3g_closed_loop.py) |
| 只有八条测量路径，连续路径覆盖不足 | 用 DTW 对齐后的插值、受限外推和普通增强构造连续路径训练分布 | Holzmüller & Sontacchi 的次级路径 DTW 插值 | DTW 只用于离线训练数据构造；加入物理边界、确定性回退、端点遮蔽和路径切换 | [phase3g_data.py](phase3g_data.py)、[train_phase3g.py](train_phase3g.py) |
| 神经控制器容易只记住可见专家 | 候选遮蔽与严格 LOPO | 本项目的泛化验证设计，不声称来自某一篇论文 | 训练时按 `40%/30%/30%` 保留全部、遮蔽一个端点、遮蔽两个端点；LOPO 物理删除留出路径的全部信息 | [phase3g_data.py](phase3g_data.py)、[phase3g_lopo.py](phase3g_lopo.py) |
| 赛题禁止运行期训练或参数更新 | 所有专家、模板、GRU 和字典离线训练并冻结 | Participant Kit 推理参数约束；fixed-filter ANC 思路 | 运行期只改变缓存、隐状态、`alpha` 和 `z`，不调用优化器或反向传播；推理前后校验 `state_dict` SHA-256 | [phase3g_validation.py](phase3g_validation.py)、[export_phase3g_submission.py](export_phase3g_submission.py) |

## 3. 各项改进的详细对应关系

### 3.1 从“时域损失”转向“官方评分对齐”

#### 资料依据

- [Deep ANC: A Deep Learning Approach to Active Noise Control](https://doi.org/10.1016/j.neunet.2021.03.037) 将 ANC 表述为监督学习问题，并以深度网络近似控制器，为原始 TCN 路线提供了端到端学习背景。
- 本赛题真正的度量依据是本地 [Participant Kit README](DEEPANC_PARTICIPANT_KIT/README.md) 及 [public_demo_scoring.py](DEEPANC_PARTICIPANT_KIT/public_demo_scoring.py)，而不是通用宽带 MSE。

#### 本项目实现

Phase 1 保留 `TimeDomainANC` 结构和参数量，重做了训练时序、损失和 checkpoint 选择：丢弃初始化段，按六个窗口分别计算三分之一倍频程功率，主损失覆盖 50 Hz～5 kHz，高频反弹覆盖 1～8 kHz。该阶段解决的是“优化什么”，尚未解决“如何显式适应未知次级路径”。

#### 对应代码

- [v6_metrics.py](v6_metrics.py)：官方窗口、STFT、频带和反弹的可微实现；
- [phase1_validation.py](phase1_validation.py)：直接调用官方 scorer 的固定验证；
- [train_phase1.py](legacy_models/phase0_phase1/train_phase1.py)：复合损失与官方指标选模。

### 3.2 从单模型转向多 FIR 专家

#### 资料依据

- H. D. Nam 和 S. J. Elliott 的 [Adaptive active attenuation of noise using multiple model approaches](https://doi.org/10.1006/mssp.1995.0042) 使用多个先验次级路径模型应对时变对象，核心思想是让多个候选模型竞争或协同解释当前系统。
- [Deep Generative Fixed-filter Active Noise Control](https://arxiv.org/abs/2303.05788) 提出通过预训练固定控制滤波器/子滤波器生成适合当前噪声的控制器；作者提供了 [GFANC 官方实现](https://github.com/Luo-Zhengding/GFANC-Generative-fixed-filter-active-noise-control)。

#### 本项目实现

Phase 3 将控制器线性部分显式化为八个路径专家。与 Nam 和 Elliott 不同，本项目不在运行期自适应更新模型；与 GFANC 不同，专家按次级路径组织，而不是按主要噪声的子带控制器组织。专家集合的价值主要有两点：给已测路径提供 oracle 上限，并为后续未知路径生成提供稳定基底。

#### 实验判断

P3-E1 oracle 专家证明 2048-tap FIR 容量足以覆盖已测路径。但只在专家之间选取或混合，无法保证未知路径性能；因此 oracle 成绩只是容量诊断，不能成为正式升级依据。

### 3.3 从学习式路径分类转向创新误差路由

#### 资料依据

- 多模型控制的基本原则是：能够最好解释当前观测的候选模型应获得更高权重。
- M. Hu、J. Wang、J. Xue 和 J. Lu 的 [A New Insight into the Secondary Path Modeling Problem in Active Noise Control](https://arxiv.org/abs/1811.03755) 从系统辨识角度讨论了不依赖持续附加探测噪声的次级路径建模条件。

#### 本项目实现

本项目没有复现在线次级路径参数辨识，而是把上述思想改造成冻结模板上的一致性检验。对候选路径 `i`：

\[
\hat D_i(f)=\operatorname{STFT}(e+S_i*y),
\]

\[
J_i=\frac{\sum_{50\text{Hz}:8\text{kHz}}|\hat D_i-P_iX|^2}
{\sum_{50\text{Hz}:8\text{kHz}}|\hat D_i|^2+\epsilon}.
\]

正确候选的 `S_i*y` 能够从误差信号中恢复出与主路径模板 `P_iX` 更一致的扰动，因此 `J_i` 应更小。`log J` 经 EWMA、温度 softmax 和权重平滑得到 `alpha`。这样做把原先难学的“路径标签分类”改成有明确物理含义的“候选模型解释误差”。

#### 对应代码

- [phase3r_templates.py](phase3r_templates.py)：生成 `P_i`、`S_i` 和带 SHA-256 的模板；
- [phase3r_model.py](phase3r_model.py)：逐采样创新路由；
- [phase3g_model.py](phase3g_model.py)：在当前正式模型中保留 P3R-E1c 的解析路由参数。

#### 经验结论

创新路由基本解决了路径识别，但严格 LOPO 仍出现明显退化。这表明后续瓶颈不是“认不出最相近路径”，而是“相近路径专家的线性/凸组合不足以表示新的最优控制器”。

### 3.4 用低秩残差生成跳出专家凸包

#### 资料依据

- GFANC 的关键启发是：固定控制滤波器不必只做单一选择，可以由神经网络生成适合当前条件的控制滤波器。
- [Unsupervised learning based end-to-end delayless generative fixed-filter active noise control](https://arxiv.org/abs/2402.09460) 将协处理器和实时控制器放入可微 ANC 闭环，并直接用累计误差信号训练；其 [官方实现](https://github.com/Luo-Zhengding/Unsupervised-GFANC) 可作为端到端 GFANC 的代码参考。
- [Transformer-based End-to-End Control Filter Generation for Active Noise Control](https://arxiv.org/abs/2605.00494) 进一步强调直接生成控制滤波器、摆脱有限子滤波器组合表示的方向。

#### 本项目实现

当前模型没有照搬 GFANC 的噪声 CNN，也没有采用 E2E-CFG 的 Transformer。考虑赛题需要逐采样实时运行，完整生成 2048 个系数成本过高，因此采用低秩残差：

```text
innovation features (52)
        -> GRUCell(52, 32)
        -> Linear(32, 16)
        -> smoothed latent z
        -> z @ residual_dictionary[16, 2048]
```

52 维特征由中心化 `log J`、创新 proposal、解析 `alpha`、置信度统计、候选 mask 和上一时刻 `z` 构成。GRU 每 240 点更新一次，块内 FIR 不变。最终 `alpha @ experts` 提供稳定基底，`z @ B` 提供凸包外的路径特定修正。

这部分与文献的对应关系是“采用生成式固定滤波器和可微闭环训练思想”，而下列设计属于本项目针对赛题的具体实现：

- 用闭环创新而不是噪声类别作为生成条件；
- 只生成 16 维残差激活，而不是完整控制 FIR；
- 解析路由与神经残差并行，避免生成器独自承担路径识别；
- `0.98*tanh` 软限幅，保证公开接口输出有界；
- 41,552 参数，比原 TCN 的 42,764 参数略少。

#### 对应代码

- [phase3g_model.py](phase3g_model.py)：网络、残差字典、逐采样状态和软限幅；
- [phase3g_closed_loop.py](phase3g_closed_loop.py)：严格使用 `e[t-1]` 的可微闭环、声学损失和分块 BPTT；
- [train_phase3g.py](train_phase3g.py)：E1 暖启动和 E2 连续路径训练。

### 3.5 用 DTW 和受限合成建立连续路径训练分布

#### 资料依据

F. Holzmüller 和 A. Sontacchi 的 [Dynamic Time Warping for Secondary Path Interpolation in Local Active Noise Control](https://doi.org/10.1109/OJSP.2026.3689448) 研究了在插值前用 DTW 对齐次级路径脉冲响应，以减少传播延迟差异对直接插值的破坏；[开放资料页](https://phaidra.kug.ac.at/detail/o%3A137250) 提供论文记录。

#### 本项目实现

DTW 不参与正式推理，也不直接在运行期插值 FIR 专家。它只用于生成离线训练样本：

- 50% 原始测量路径；
- 30% DTW 对齐后的路径对插值；
- 10% 有界外推；
- 10% 增益、延迟和尾部增强；
- 25% 样本在绝对时间 2.0 秒切换路径。

插值/外推会检查峰值延迟和频带能量边界，超限时确定性回退到全局延迟对齐。对应 disturbance 使用同一参考噪声的对齐 `EXPECTED_NOISE` 同权组合，避免只改变次级路径却保留不一致目标。

#### 对应代码

- [phase3g_data.py](phase3g_data.py)：DTW、连续路径合成、边界检查和候选遮蔽；
- [phase3g_data.py](phase3g_data.py) 中的 `build_phase3g_manifest()`：固定随机种子、来源和 SHA-256；[train_phase3g.py](train_phase3g.py) 在每次实验目录中固化该 manifest。

### 3.6 用候选遮蔽、严格 LOPO 和封存路径验证泛化

这一部分主要是本项目的实验设计，不对应某篇论文的直接算法。

训练时随机遮蔽生成路径的一个或两个端点，迫使模型在没有精确专家时利用创新证据和残差字典。严格 LOPO 中，每折物理删除留出路径的 disturbance、次级路径、oracle 专家、创新模板、字典初始化来源和插值/外推端点，防止“表面留出、实际泄漏”。路径 9/10 在开发、结构选择、LOPO 和 seed 选择阶段保持封存，只在全部门槛通过后做一次最终评估。

- 候选遮蔽：[phase3g_data.py](phase3g_data.py)
- 严格 LOPO：[phase3g_lopo.py](phase3g_lopo.py)
- 三种子和封存路径最终评估：[phase3g_final_evaluation.py](phase3g_final_evaluation.py)

严格八折 LOPO 的主指标增益为：

| 留出路径 | 相对 P1-E2 增益（dB） |
|---:|---:|
| 1 | +1.1602 |
| 2 | +2.6055 |
| 3 | +1.1473 |
| 4 | -0.1347 |
| 5 | +0.2235 |
| 6 | +2.5421 |
| 7 | +0.6751 |
| 8 | -2.3413 |

中位增益为 `+0.9112 dB`，`6/8` 折不退化，达到预设门槛。路径 8 仍是明确的残余风险，说明当前方法改善了总体泛化，但并未解决所有外推情形。

### 3.7 冻结权重与赛题合规性

赛题要求部署模型离线推理，实际运行期间不得在线训练或更新参数。因此当前模型把“自适应”限定为状态自适应：

- 允许改变：FIR 环形缓存、频谱历史、EWMA、`alpha`、GRU hidden、16 维 `z`；
- 不允许改变：八个专家、路径模板、GRU 权重、输出头、残差字典及任何 `state_dict` 内容；
- 不包含：运行期优化器、`.backward()`、在线 FxLMS、Meta-AF、持续探测噪声或录音间状态继承。

[phase3g_validation.py](phase3g_validation.py) 对推理前后 `state_dict` 计算 SHA-256，[export_phase3g_submission.py](export_phase3g_submission.py) 只导出模型配置和权重。这里的“生成 FIR”是冻结网络的输出激活，与运行期学习或更新模型参数不同。

## 4. 从原模型到当前模型的技术演化

```text
P1-E2: 因果 TCN
  └─ 优点：端到端、官方 v6 对齐、平均成绩好
  └─ 问题：次级路径隐式建模，最差路径弱

P3-E1: 路径特定 FIR oracle 专家
  └─ 结论：FIR 容量足够，专家化方向可行

P3-E2/E3: 10 维块统计 + GRU 路由
  └─ 失败原因：反馈统计不能可靠辨识次级路径

P3R: 多模型创新误差解析路由
  └─ 改善：路径识别和切换恢复基本解决
  └─ 剩余问题：未知路径控制器不在专家凸包内

P3G: 创新条件 GRU + 低秩残差字典
  └─ 改善：保留可靠解析路由，同时生成凸包外 FIR 修正
  └─ 结果：开发、压力集、严格 LOPO、三种子和十路径门槛通过
```

这条演化链中的关键判断来自本地消融实验：先分离“控制器容量”“路径识别”和“未知路径表示能力”三个问题，再只针对已经证实的瓶颈增加结构。当前正式模型不是简单把多篇论文模块堆叠起来，而是保留了每一阶段经实验验证有效的最小部分。

## 5. 公开资料与本地实现索引

### 5.1 论文

1. H. Zhang and D. Wang, [Deep ANC: A Deep Learning Approach to Active Noise Control](https://doi.org/10.1016/j.neunet.2021.03.037), *Neural Networks*, 2021.
2. H. D. Nam and S. J. Elliott, [Adaptive active attenuation of noise using multiple model approaches](https://doi.org/10.1006/mssp.1995.0042), *Mechanical Systems and Signal Processing*, 1995.
3. M. Hu, J. Wang, J. Xue, and J. Lu, [A New Insight into the Secondary Path Modeling Problem in Active Noise Control](https://arxiv.org/abs/1811.03755), arXiv:1811.03755, 2018.
4. Z. Luo et al., [Deep Generative Fixed-filter Active Noise Control](https://arxiv.org/abs/2303.05788), ICASSP 2023, DOI: [10.1109/ICASSP49357.2023.10095205](https://doi.org/10.1109/ICASSP49357.2023.10095205).
5. Z. Luo et al., [Unsupervised learning based end-to-end delayless generative fixed-filter active noise control](https://arxiv.org/abs/2402.09460), ICASSP 2024, DOI: [10.1109/ICASSP48485.2024.10448277](https://doi.org/10.1109/ICASSP48485.2024.10448277).
6. Z. Yang et al., [Transformer-based End-to-End Control Filter Generation for Active Noise Control](https://arxiv.org/abs/2605.00494), arXiv:2605.00494, 2026.
7. F. Holzmüller and A. Sontacchi, [Dynamic Time Warping for Secondary Path Interpolation in Local Active Noise Control](https://doi.org/10.1109/OJSP.2026.3689448), *IEEE Open Journal of Signal Processing*, 2026.

### 5.2 公开实现

- [GFANC 官方代码](https://github.com/Luo-Zhengding/GFANC-Generative-fixed-filter-active-noise-control)：固定子滤波器组合和 CNN 生成权重的参考实现；
- [Unsupervised-GFANC 官方代码](https://github.com/Luo-Zhengding/Unsupervised-GFANC)：端到端可微 GFANC 训练参考实现。

公开实现用于理解论文结构和训练思路。本项目没有复制其模型或预训练权重，正式模型训练数据、路径模板、网络结构和导出运行时均来自本仓库。

### 5.3 本地资料

- [MODEL_IMPROVEMENT_LOG.md](MODEL_IMPROVEMENT_LOG.md)：各阶段实验、停止规则和正式升级记录；
- [PHASE1_V6.md](PHASE1_V6.md)：官方 v6 指标对齐；
- [PHASE3_FEEDBACK_FIR.md](PHASE3_FEEDBACK_FIR.md)：FIR 专家及失败的学习式反馈路由；
- [PHASE3R_INNOVATION_ROUTING.md](PHASE3R_INNOVATION_ROUTING.md)：创新误差路由修复；
- [PHASE3G_GENERATIVE_FIR.md](PHASE3G_GENERATIVE_FIR.md)：当前正式生成式 FIR 模型；
- [phase3g_formal_model.json](artifacts/phase3g_formal_model.json)：正式 checkpoint 与最终指标；
- [phase3g_final_evaluation.json](runs/phase3g_final_evaluation.json)：三种子和十路径最终评估；
- [lopo_summary.json](runs/phase3g_suite_seed2026_rerun1/P3G-LOPO-rerun1/lopo_summary.json)：严格八折 LOPO 汇总。

## 6. 结果与边界

| 指标 | P1-E2 | Phase 3G 正式模型 | 变化 |
|---|---:|---:|---:|
| 十路径主指标 `S` | 10.9790 dB | 17.9585 dB | +6.9795 dB |
| 平均窗口反弹 `R` | 1.7392 dB | 1.6870 dB | -0.0522 dB |
| 综合分 `C` | 7.1635 | 12.0648 | +4.9013 |
| 最差路径主指标 | 0.9733 dB | 3.9775 dB | +3.0043 dB |
| 参数量 | 42,764 | 41,552 | -1,212 |
| 平均 MAC/sample | 42,048 | 约 12,392 | 约 -70.5% |

正式模型还通过了以下检查：

- 严格 LOPO 中位提升 `+0.9112 dB`，`6/8` 折不退化；
- seeds `2026/2027/2028` 均通过开发和合成压力集门槛；
- 路径 9/10 平均相对 P1-E2 提高 `2.9915 dB`；
- Participant Kit 提交检查参数量 41,552，CPU RTF `0.695`；
- 公开完整闭环 demo RTF `0.959`；
- 38 项自动测试通过，推理前后 `state_dict` 不变。

这些结果支持 Phase 3G 取代 P1-E2 成为当前正式模型，但不意味着问题完全解决。最明显的风险是 LOPO 路径 8 仍退化 `2.3413 dB`，说明训练路径覆盖、低秩字典表达能力和创新模板的域外可靠性仍有提升空间。后续若继续改进，应优先针对这一失败折做因果诊断，而不是在路径 9/10 上继续调参。
