# DEEPANC 模型改进记录

> 本文件是持续更新的实验与改进日志。Phase 0、Phase 1 与 Phase 3G 已完成，Phase 2、Phase 3、Phase 3R 已按
> 停止规则判定未通过。Phase 4R 的运行时加固已通过；最差路径诊断已完成，但首个修正实验 E10-A 未通过。
> 当前正式模型仍为经过 E09-A 等价加固的 Phase 3G v3。

## 1. 当前状态摘要

| 项目 | 当前状态 |
|---|---|
| 当前阶段 | **Phase 4R：运行时加固通过；E10-A 最差路径修正未通过** |
| 当前模型 | `GenerativeInnovationFIRController`，seed 2027，E2 第 15 轮 |
| 参数量 | 41,552，约 0.0416 M |
| 理论计算量 | 平均约 12,392 MAC/sample；块边界峰值事件 2,484,720 MAC |
| 训练/推理时序 | 0.5 s 初始化 + 3.0 s 六窗口计分，共 3.5 s；240 点更新一次 FIR 激活 |
| 训练 batch | micro-batch 8，梯度累积 1，有效 batch 8 |
| 三个正式种子 | 2026 / 2027 / 2028，全部通过开发门槛 |
| 十路径 v6 主指标 | **17.9585 dB**，P1-E2 为 10.9790 dB |
| 十路径平均窗口反弹 | **1.6870 dB**，P1-E2 为 1.7392 dB |
| 十路径综合分 | **12.0648**，P1-E2 为 7.1635 |
| 十路径最差主指标 | **3.9775 dB**，P1-E2 为 0.9733 dB |
| 路径 9 / 10 主指标 | 15.2938 / **3.9775 dB** |
| Participant Kit CPU RTF | v3 十次完整闭环 P50/P95/最大 0.6760/0.7000/0.7082 |
| 当前 checkpoint | `runs/phase3g_suite_seed2027/P3G-E2/checkpoints/best_phase3g_selection.pt` |
| 正式提交包 | `phase3g_submission_final_seed2027_v3` |

Phase 2 开发实验虽将路径 1～8 平均主指标最高提高到 12.7328 dB，但最差路径最高
仅为 0.9884 dB；路径增强/最坏组修正的最好结果也只有 0.9822 dB，未达到继续
LOPO 与最终路径验证所需的 1.5 dB 停止阈值。因此路径 9/10 未被访问，正式模型不变。

Phase 3 的公共 FIR 达到 12.7009 dB，oracle 专家达到 16.5658 dB 且最差路径
4.1513 dB，证明 FIR 专家容量足够；但 hidden 24/32 的反馈门控主指标分别只有
8.2564/4.4530 dB，路由准确率仅 11.91%/21.51%。按规则停止在门控阶段，未运行
联合微调、LOPO、三种子复训或路径 9/10 最终评估。

Phase 3R 用显式多模型创新误差替代 GRU 猜测路径。三个固定无探测候选均通过开发门槛，
选中的 P3R-E1c 达到 `S=16.5705`、`R=0.1899`、路由准确率 `100%`、最慢切换恢复
`85 ms` 和 CPU RTF `0.704`。但严格移除留出路径专家、`P_i`、`S_i` 的八折 LOPO
中位增益为 `-1.1187 dB`，仅 3/8 折不退化，因此停止在 LOPO，路径 9/10 仍未访问。

Phase 3G 用冻结权重的 41,552 参数 GRU 生成器在 P3R 解析专家混合上生成 16 维残差 FIR 激活。
seed 2026 正式 E2 开发达到 `S=19.5192/R=0.3395`，48 条连续路径压力集中位增益
`+2.7751 dB`、91.67% 不退化。严格八折 LOPO 中位增益 `+0.9112 dB`，6/8 折不退化，
8/8 折物理隔离通过，因此没有触发 E3。seeds 2026/2027/2028 均通过全部开发门槛。
最终十路径评估按开发 D 选中 seed 2027：十路径 `S=17.9585/R=1.6870/C=12.0648`，
最差路径和路径 10 均为 `3.9775 dB`，路径 9/10 平均相对 P1-E2 提升 `2.9915 dB`。
Participant Kit 正式提交检查 RTF 为 0.695，完整公开闭环为 0.959；Phase 3G 正式升级通过。

Phase 4R E09-A 删除了提交包装器逐采样重复创建的 `torch.inference_mode()`，保留模型
240 点块边界的局部推理模式。v3 与 v2 的 168,000 点控制输出和闭环残差逐点完全一致，
权重和 `state_dict` 哈希不变；十次完整公开闭环 RTF P50/P95/最大值降为
`0.6760/0.7000/0.7082`，因此运行时加固通过并停止在 E09-A。

冻结 v3 后完成 Path 8 因果诊断，根因按机械规则选为 `training_coverage_or_dictionary`。
随后运行 E10-A：仅把合成样本的确定性单最近邻端点改为折内三邻居均匀采样。Path 8
主增益改善 `0.5643 dB`，但仍只有 `-1.7770 dB`，反弹为 `10.5620 dB`；八折中位
增益降至 `0.6638 dB`，仅 5/8 不退化，Path 7/1/6 均未改善。E10-A 未通过预注册
门槛，未运行三种子、未修改正式 v3、ZIP 或正式指标。

Phase 0 的操作说明保留在 [PHASE0_STRONG_BASELINE.md](PHASE0_STRONG_BASELINE.md)，
Phase 1 的实现与运行说明见 [PHASE1_V6.md](PHASE1_V6.md)，完整后续路线见
[MODEL_IMPROVEMENT_ROADMAP.md](MODEL_IMPROVEMENT_ROADMAP.md)。

---

## 2. Phase 0：建立可信的强基线

### 2.1 阶段目标

Phase 0 不修改 `TimeDomainANC` 网络结构，也不改变时域 MSE 训练目标。该阶段只修复数据、采样、验证、可复现性、日志与模型保存流程，为后续结构和损失函数消融建立可信对照。

因此，Phase 0 不改变正式推理时的参数量和单次推理复杂度；增加的是训练样本数、优化器更新次数和训练总计算量。

### 2.2 已完成的代码改进

#### 数据加载与完整性

- WAV 扩展名改为大小写不敏感扫描；
- 完整识别当前数据集的 8 类噪声；
- 兼容标准命名 `KTV_scene_01.wav`；
- 兼容特殊命名 `餐厅.WAV_scene_01.wav`；
- Dataset 初始化时检查原始噪声、期望噪声、采样率和路径索引；
- `sh.npy` 从 `(1967, 10)` 转换为训练使用的 `(10, 1967)`；
- 训练路径通过循环索引实现近似均衡采样；
- 噪声类型和音频起点继续随机采样。

#### 虚拟 epoch

旧代码中：

```text
Dataset 长度 = 8 条训练路径
batch size = 8
每个 epoch = 1 次 optimizer.step()
60 epoch ≈ 60 次更新
```

Phase 0 默认配置：

```text
samples_per_epoch = 512
batch_size = 8
每个 epoch = 64 次 optimizer.step()
60 epoch = 3840 次更新
```

#### 测试片段拼接修复

旧测试代码计划把两个 0.5 秒片段拼成 1 秒，但底层读取函数每次实际读取 1 秒，最终返回了 2 秒测试样本。

Phase 0 已修正为：

```text
前半段噪声 0.5 秒 + 后半段噪声 0.5 秒 = 1.0 秒
```

该修复会改变测试结果，因此 Phase 0 前后的数值只能作为趋势参考，不能视为完全相同测试条件下的严格消融。

#### 可复现性

已固定：

- Python `random`；
- NumPy；
- PyTorch CPU；
- PyTorch CUDA；
- CuDNN deterministic；
- CuBLAS workspace configuration；
- DataLoader generator 与 worker seed。

#### 验证、日志与 checkpoint

训练脚本现在会：

- 默认每 5 轮执行一次验证；
- 分别输出已见路径和未见路径结果；
- 记录每条路径的 NR；
- 记录平均、最好和最差路径 NR；
- 写入逐轮 `history.jsonl`；
- 写入训练参数 `config.json`；
- 写入实际数据划分 `data_split.json`；
- 写入最终结果 `summary.json`；
- 每轮覆盖保存 `latest.pt`；
- 保存平均 NR 最优的 `best_mean_nr.pt`；
- 保存最差路径 NR 最优的 `best_robust_nr.pt`；
- 训练结束后自动加载平均 NR 最优模型生成图表。

### 2.3 修改文件

- [dataset.py](dataset.py)：数据解析、配对验证、虚拟 epoch 和测试拼接；
- [train.py](legacy_models/phase0_phase1/train.py)：配置、随机种子、日志、周期验证、checkpoint 和结果图；
- [PHASE0_STRONG_BASELINE.md](PHASE0_STRONG_BASELINE.md)：Phase 0 运行说明。

---

## 3. Phase 0 正式 60 轮实验

### 3.1 实验环境

```text
Conda environment : pytorch2.5.1_py310
Python            : 3.10.16
PyTorch           : 2.5.1
CUDA              : 12.4
GPU               : NVIDIA GeForce RTX 3050 Ti Laptop GPU
GPU VRAM          : 4 GiB
```

### 3.2 实验配置

```text
epochs              = 60
samples_per_epoch    = 512
batch_size           = 8
optimizer steps/epoch= 64
total optimizer steps= 3840
segment_duration     = 1.0 s
learning_rate        = 0.001
validation_interval  = 5
seed                 = 2026
```

运行命令：

```powershell
conda run --no-capture-output -n pytorch2.5.1_py310 python -u train.py `
  --epochs 60 `
  --samples-per-epoch 512 `
  --batch-size 8 `
  --validation-interval 5 `
  --output-dir runs/phase0_60ep_seed2026 `
  --seed 2026
```

实际训练时间：

```text
2612.69 秒，约 43 分 33 秒
```

### 3.3 数据划分

```text
训练噪声：KTV、公交、厨房、地铁、步行街、火车
测试噪声：车载、餐厅
训练路径：1～8
未见路径：9～10
```

测试片段由“车载 0.5 秒 + 餐厅 0.5 秒”拼接而成。

### 3.4 周期验证结果

当前 Phase 0 仍使用旧版整段能量 NR，而不是正式 v6 六窗口 1/3 倍频带指标。

| Epoch | 已见路径平均 NR | 未见路径平均 NR | 十路径综合平均 NR | 十路径最差 NR |
|---:|---:|---:|---:|---:|
| 1 | 4.414 | 2.971 | 4.126 | 0.476 |
| 5 | 8.088 | 5.932 | 7.657 | 0.588 |
| 10 | 6.424 | 5.634 | 6.266 | 0.824 |
| 15 | 10.335 | 8.072 | 9.883 | 0.715 |
| 20 | 4.766 | 2.845 | 4.382 | 0.406 |
| 25 | 9.612 | 7.570 | 9.203 | 0.787 |
| 30 | 10.173 | 7.294 | 9.597 | 0.754 |
| **35** | **11.104** | **7.676** | **10.419** | 0.794 |
| 40 | 8.116 | 5.413 | 7.575 | 0.739 |
| **45** | 10.126 | 7.669 | 9.635 | **0.876** |
| 50 | 10.404 | 7.149 | 9.753 | 0.799 |
| 55 | 11.001 | 7.450 | 10.291 | 0.795 |
| 60 | 9.173 | 5.356 | 8.410 | 0.690 |

结果表明验证性能并不随 epoch 单调增加。第 60 轮明显弱于第 35 轮，证明周期验证和最佳 checkpoint 保存是必要的。

### 3.5 平均 NR 最优模型

平均 NR 最优模型出现在第 35 轮：

```text
已见路径平均 NR：11.104 dB
未见路径平均 NR：7.676 dB
十路径综合平均：10.419 dB
十路径最差值：0.794 dB
```

各路径结果：

```text
已见路径 1～8：
14.55, 13.95, 14.21, 13.82, 12.95, 6.61, 0.79, 11.94 dB

未见路径 9～10：
13.82, 1.53 dB
```

模型文件：

- [best_mean_nr.pt](runs/phase0_60ep_seed2026/checkpoints/best_mean_nr.pt)

### 3.6 最差路径最优模型

稳健性最优模型出现在第 45 轮：

```text
已见路径平均 NR：10.126 dB
未见路径平均 NR：7.669 dB
十路径综合平均：9.635 dB
十路径最差值：0.876 dB
```

模型文件：

- [best_robust_nr.pt](runs/phase0_60ep_seed2026/checkpoints/best_robust_nr.pt)

### 3.7 与旧 60 轮运行的参考对比

| 测试条件 | 旧 60 轮结果 | Phase 0 平均最优 |
|---|---:|---:|
| 未见噪声 + 已见路径 | 5.17 dB | 11.10 dB |
| 未见噪声 + 未见路径 | 3.48 dB | 7.68 dB |

该对比表明增加有效训练样本和参数更新次数具有明显收益，但不是严格消融，原因包括：

- 测试拼接长度从错误的 2 秒修正为 1 秒；
- 新版本完整纳入 `餐厅.WAV`；
- 测试噪声组合发生变化；
- 新版本固定随机种子并使用最佳 checkpoint。

---

## 4. Phase 1：训练目标对齐官方 v6 评分

### 4.1 阶段目标与边界

Phase 1 保持 `TimeDomainANC` 网络结构、通道数、层数和参数量不变，只修改：

- 训练片段由 1.0 秒改为正式协议要求的 3.5 秒；
- 训练目标由整段时域 MSE 改为 v6 评分对齐复合损失；
- 验证由旧版整段能量 NR 改为官方六窗口 1/3 倍频带指标；
- checkpoint 由平均 NR 选择改为官方比例内部综合分选择；
- 增加固定三场景、每路径、每窗口的可追溯验证结果。

本阶段明确不包含：

- 逐采样提交接口和显式卷积缓存；
- `e[t-1]` 反馈分支；
- 次级路径增强；
- CVaR/最差路径损失；
- 网络结构和复杂度优化。

因此 Phase 1 是严格的“评分对齐”阶段，而不是流式提交或路径鲁棒阶段。

### 4.2 官方协议与路线图修正

Participant Kit 1.5.0 / 协议 v6 的单场景时序为：

```text
0.5 秒初始化 + 3.0 秒正式计分
3.0 秒计分段 = 6 × 0.5 秒连续、不重叠窗口
```

每个窗口独立完成：

```text
去直流
Hann window
n_fft = 8192
hop_length = 2048
center = False
尾部补 576 点以纳入窗口末端
FFT 功率汇总为标准 1/3 倍频带
```

根据官方实现和显存实测，对原路线图草案做出以下修正：

1. **反弹训练直接使用官方硬最大值**。每个窗口在 1～8 kHz 频带中取最坏
   `change_db` 后执行 `ReLU`，不再使用路线图建议的平滑最大值。主频带项已经为
   所有频带提供梯度，硬最大值可以把额外梯度准确施加到最危险频带。
2. **时域 MSE 改为尺度无关能量比**。原始 MSE 约为 `1e-4` 量级，与 dB 损失
   直接相加会产生严重量纲不匹配，因此改为计分段 `mean(e²)/mean(d²)`。
3. **控制正则改为安全护栏**。Phase 0 控制峰值实测仅约 0.074，没有必要持续
   压缩控制 RMS；仅当 `|y| > 1.0` 时处罚超限部分。
4. **整段处理 3.5 秒**。当前模型没有显式状态缓存，因此 Phase 1 直接整段因果
   推理，避免分块重置左上下文。显存实测 micro-batch 2 的复合损失反向峰值约
   1.46 GiB，可以在 4 GiB 显卡上稳定运行。
5. **暂缓 CVaR**。3.5 秒长片段下可用 micro-batch 较小，当前阶段加入 CVaR
   会混淆评分损失本身的收益，留到路径鲁棒阶段单独验证。

### 4.3 可微损失实现

新增 [v6_metrics.py](v6_metrics.py)，输入信号长度固定为 168,000 点：

```text
change_db = 10 log10(P_error / P_target)

L_primary = mean(change_db[50 Hz:5 kHz]) / 10
L_rebound = mean_window(ReLU(max_band(change_db[1 kHz:8 kHz]))) / 10
L_time    = mean(e_score²) / (mean(d_score²) + eps)

v         = ReLU(abs(y_full) - 1.0)
L_guard   = mean(v²) + max(v²)
```

其中：

- `P_target` 从计算图分离；
- 所有频带功率使用 `clamp_min(1e-20)`；
- 初始化段参与模型与次级路径连续传播，但不参与声学损失；
- 安全护栏覆盖完整 3.5 秒，包括初始化段。

两个正式训练目标为：

```text
P1-E1:
L = 1.0 L_primary + 0.1 L_time + 1.0 L_guard

P1-E2:
L = 0.7 L_primary + 0.3 L_rebound
  + 0.1 L_time + 1.0 L_guard
```

### 4.4 固定三场景验证集

新增 [phase1_data.py](phase1_data.py) 和
[phase1_validation.py](phase1_validation.py)。验证 manifest 固定为：

| 场景 | 内容 | 切换点 |
|---|---|---:|
| `vehicle_continuous` | 车载连续 3.5 秒 | 无 |
| `restaurant_continuous` | 餐厅连续 3.5 秒 | 无 |
| `vehicle_to_restaurant` | 车载切换到餐厅 | 绝对时间 2.0 秒 |

统一从源音频第 20 秒开始读取。每个场景遍历全部 10 条路径，因此一次正式验证
包含 30 个场景-路径组合、180 个独立计分窗口。

验证不使用可微近似值选模，而是直接调用：

```text
DEEPANC_PARTICIPANT_KIT/public_demo_scoring.py
score_windowed_signals()
```

内部模型选择分数定义为：

```text
C = 0.7 × S - 0.3 × R
```

其中 `S` 是 50 Hz～5 kHz 主指标，`R` 是六窗口平均的 1～8 kHz 最坏频带
反弹。公开 demo 没有用于训练或 checkpoint 选择。

### 4.5 代码、checkpoint 与测试

新增文件：

- [v6_metrics.py](v6_metrics.py)：可微 v6 频带功率、主指标和反弹损失；
- [phase1_data.py](phase1_data.py)：固定三场景验证数据；
- [phase1_validation.py](phase1_validation.py)：官方 scorer 调用和指标聚合；
- [train_phase1.py](legacy_models/phase0_phase1/train_phase1.py)：Phase 1 微调、早停、日志和 checkpoint；
- [run_phase1_experiments.py](legacy_models/phase0_phase1/run_phase1_experiments.py)：E0/E1/E2/条件 E3 编排；
- [PHASE1_V6.md](PHASE1_V6.md)：运行说明；
- `tests/test_v6_metrics.py`、`tests/test_phase1_data.py`：回归测试。

每次训练保存：

```text
latest.pt
best_official_composite.pt
best_primary.pt
best_rebound.pt
```

共 7 项单元测试通过，覆盖：

- 可微实现与官方 scorer 的逐窗口误差小于 `1e-4 dB`；
- 0.5 倍、2 倍幅度的解析结果；
- 六窗口 dB 等权平均；
- 初始化段排除；
- 反向梯度有限且非零；
- 输出护栏触发条件；
- 固定场景形状和训练随机采样复现。

### 4.6 正式实验配置

P1-E1 与 P1-E2 均从 Phase 0 第 35 轮 `best_mean_nr.pt` 独立初始化，只加载
模型参数并重新创建优化器，保证二者消融公平。

```text
epochs                  = 20
samples_per_epoch        = 256
segment_duration         = 3.5 s
micro_batch_size         = 2
gradient_accumulation    = 4
effective_batch_size     = 8
optimizer steps/epoch    = 32
maximum optimizer steps  = 640
optimizer                = Adam + AMSGrad
learning_rate            = 0.0001
gradient_clip            = 1.0
validation_interval      = 1 epoch
early_stop_patience      = 5
early_stop_min_delta     = 0.05
seed                     = 2026
AMP                      = disabled
device                   = CUDA
```

GPU 训练必须通过 Conda 启动，以正确加载 CUDA/NVRTC 动态库：

```powershell
conda run --no-capture-output -n pytorch2.5.1_py310 python -u -m legacy_models.phase0_phase1.run_phase1_experiments `
  --output-root runs/phase1_suite_seed2026 `
  --device cuda
```

### 4.7 P1-E0：正式 v6 基线

Phase 0 第 35 轮模型在固定三场景 v6 验证集上的结果：

```text
主指标 S             = 6.7681 dB
平均窗口反弹 R       = 11.6514 dB
内部综合分 C         = 1.2423
首窗口主指标         = 6.8700 dB
最差窗口主指标       = 0.6215 dB
最差路径主指标       = 0.6890 dB
控制输出峰值         = 0.0665
```

这与此前旧拼接场景得到的 `S≈6.75 dB、R≈11.43 dB` 接近，说明新固定验证集
没有产生异常的指标偏移。

### 4.8 P1-E1：仅主频带对齐

关键训练轮次：

| Epoch | 主指标 S | 反弹 R | 综合分 C |
|---:|---:|---:|---:|
| 1 | 8.7972 | 10.5123 | 3.0044 |
| 5 | 10.0354 | 10.0704 | 4.0037 |
| 10 | 10.6453 | 10.2226 | 4.3849 |
| 15 | 10.8735 | 9.8678 | 4.6511 |
| 19 | 10.9330 | 9.5173 | 4.7979 |
| **20** | **10.9839** | **9.4301** | **4.8597** |

训练时间：1997.84 秒，约 33.30 分钟。

相对 P1-E0：

```text
主指标提高：4.2158 dB
反弹降低：19.06%
综合分提高：3.6174
```

E1 证明 1/3 倍频带主损失与官方主指标方向一致，也会间接改善反弹。但反弹降幅
为 19.1%，略低于预设 20% 硬门槛，因此 E1 被标记为未通过。

### 4.9 P1-E2：70/30 主指标与反弹复合目标

关键训练轮次：

| Epoch | 主指标 S | 反弹 R | 综合分 C |
|---:|---:|---:|---:|
| 1 | 8.6378 | 8.9670 | 3.3563 |
| 5 | 9.8702 | 4.6206 | 5.5230 |
| 10 | 10.2860 | 2.0517 | 6.5847 |
| 15 | 10.8175 | 1.7374 | 7.0510 |
| **19** | **10.9790** | **1.7392** | **7.1635** |
| 20 | 10.8972 | 1.6624 | 7.1293 |

训练时间：1890.61 秒，约 31.51 分钟。综合分最佳模型出现在第 19 轮，而不是
反弹最低的第 20 轮，说明分别保存综合、主指标和反弹 checkpoint 是必要的。

相对 P1-E0：

```text
主指标提高：4.2109 dB
反弹降低：85.07%
综合分提高：5.9213
控制输出峰值：0.0663
```

验收结果：

| 验收项 | 要求 | 实际 | 结果 |
|---|---:|---:|---|
| 综合分提升 | ≥ 0.5 | +5.9213 | 通过 |
| 主指标下降 | 不超过 0.25 dB | **提高 4.2109 dB** | 通过 |
| 反弹降低 | ≥ 20% | 85.07% | 通过 |
| 控制峰值 | ≤ 1.0 | 0.0663 | 通过 |
| 数值稳定性 | 无 NaN/Inf | 无异常 | 通过 |

E2 全部通过，因此没有触发预设的 P1-E3 权重修正实验。

### 4.10 分场景与分路径结果

三个固定场景均同时提高主指标并显著降低反弹：

| 场景 | P1-E0 S | P1-E2 S | P1-E0 R | P1-E2 R |
|---|---:|---:|---:|---:|
| 餐厅连续 | 6.7972 | 10.9642 | 12.0774 | 1.8736 |
| 车载连续 | 6.7356 | 11.0546 | 11.3522 | 1.6128 |
| 车载→餐厅 | 6.7715 | 10.9182 | 11.5246 | 1.7313 |

十路径结果：

| 路径 | P1-E0 S | P1-E2 S | P1-E0 R | P1-E2 R |
|---:|---:|---:|---:|---:|
| 1 | 10.190 | 14.669 | 12.545 | 0.000 |
| 2 | 8.225 | 10.467 | 9.546 | 2.190 |
| 3 | 7.067 | 14.253 | 18.494 | 0.539 |
| 4 | 8.748 | 15.875 | 14.774 | 2.457 |
| 5 | 10.298 | 15.603 | 15.826 | 3.125 |
| 6 | 3.633 | 5.114 | 8.994 | 1.984 |
| 7 | 0.689 | 0.973 | 1.684 | 0.293 |
| 8 | 9.561 | 19.547 | 14.621 | 4.347 |
| 9 | 8.255 | 11.365 | 14.085 | 1.718 |
| 10 | 1.015 | 1.923 | 5.944 | 0.738 |

所有路径的主指标均有提高，所有路径的反弹均有降低。路径 7 和路径 10 虽有改善，
但主指标仍分别只有 0.973 dB 和 1.923 dB，表明评分对齐损失无法单独解决路径
泛化问题。

### 4.11 Phase 1 结论

Phase 1 已确认：

- 可微 1/3 倍频带损失能够显著提升官方主指标；
- 只优化主频带不足以稳定压低每窗口最坏频带反弹；
- 直接加入官方硬反弹目标后，反弹从 11.651 dB 降至 1.739 dB；
- 70/30 复合目标没有以牺牲主指标为代价，主指标仍提高 4.211 dB；
- 三个场景和十条路径的改善方向一致；
- `TimeDomainANC` 结构、参数量和理论推理复杂度均未改变；
- P1-E2 第 19 轮确定为当前最佳 Phase 1 checkpoint。

当前最佳模型：

- [best_official_composite.pt](runs/phase1_suite_seed2026/P1-E2/checkpoints/best_official_composite.pt)

下一阶段应转向路径鲁棒性，而不是继续单纯增加 Phase 1 训练轮数。

---

## 5. Phase 2：路径鲁棒训练

### 5.1 目标、现实修正与数据隔离

Phase 2 保持 `TimeDomainANC` 和 42,764 参数不变，从 P1-E2 第 19 轮独立初始化。
同一参考片段对 K 条路径分别加载与真实基础路径匹配的 disturbance，再计算
`mean_path + beta * top-25%`。增强限制为 ±1.5 dB、±2 个整数采样和
-35～-30 dB 平滑尾部扰动，避免不真实的强 tap 白噪声、任意插值和陷波移动。

训练沿用前六条噪声，固定 vehicle/restaurant 两条噪声只用于 validation。路径 1～8
用于训练和 development；路径 9/10 不参与梯度、逐轮选模、E3 判断，并在本轮实验中
始终没有被访问。固定 synthetic stress 只由路径 1～8 的确定性局部增强构成。

### 5.2 正式配置与实验

```text
checkpoint             = P1-E2 epoch 19
epochs                 = 10
samples/epoch          = 256
micro-batch            = 2
gradient accumulation  = 4
effective batch        = 8
learning rate          = 1e-4
seed                    = 2026
P2-E0                   = K1, beta 0, no augmentation
P2-E1                   = K4 real paths, beta 0.25
P2-E2                   = K4 augmented paths, beta 0.25
P2-E3                   = K4 augmented paths, beta 0.50 (deterministic trigger)
```

P2-E2 最差路径提升不足 0.5 dB，因此按规则只执行一次 E3，提高 `beta` 到 0.50。
新增的 7 项 Phase 2 测试与原有 7 项回归测试共 14 项全部通过；修正训练噪声隔离后，
一个 batch 的 GPU 烟雾训练也完成了前向、反向、development/stress 验证和 checkpoint 链路。

### 5.3 正式结果

开发集只含路径 1～8，不能与 Phase 1 的十路径总体均值直接比较。P1-E2 在该开发集上的
正式对照为 `S=12.0627`、`R=1.8670`、`C=7.8838`、`robust=10.1661`、
最差路径 `0.9733 dB`。

| 实验 | 最佳轮 | 开发 S | 开发 R | robust | 最差路径 | stress 最差 |
|---|---:|---:|---:|---:|---:|---:|
| P2-E0 继续单路径训练 | 8 | 12.4375 | 1.7382 | 10.4165 | 0.9674 | 1.1639 |
| P2-E1 K4 真实路径 | 10 | **12.7328** | 1.9570 | **10.6433** | **0.9884** | **1.1901** |
| P2-E2 保守增强 | 9 | 12.5255 | **1.7033** | 10.5893 | 0.9725 | 1.1711 |
| P2-E3 beta=0.50 | 9 | 12.5148 | 1.7752 | 10.6100 | 0.9822 | 1.1831 |

P2-E1 的开发 robust 最高，但路径 7 仅从 0.9733 提高到 0.9884 dB，提升只有
0.0151 dB。针对路径鲁棒性的 E2/E3 也均低于 1.0 dB，说明均值和 P25 改善不能
替代绝对最差路径改善。

### 5.4 结论与停止决定

Phase 2 **未通过**。E2/E3 最好最差路径为 0.9822 dB，低于预设的 1.5 dB
停止阈值，更低于最终 2.0 dB 硬门槛。按计划不运行 LOPO、不运行三种子 P2-E4、
不访问路径 9/10，也不把开发 robust 最高的 E1 当作正式升级。正式 checkpoint 保留：

```text
runs/phase1_suite_seed2026/P1-E2/checkpoints/best_official_composite.pt
```

结果支持此前的物理判断：纯前馈 `x -> y` 对未知次级路径只能学习折中，不能从残差中
辨识当前路径。下一阶段应转入 Phase 3，引入 `e[t-1]` 反馈和显式流式状态，而不是继续
扩大增强范围或反复调整 beta。

### 5.5 产物

- 正式套件：`runs/phase2_suite_seed2026/`
- 套件结论：`runs/phase2_suite_seed2026/suite_summary.json`
- E0/E1/E2/E3 各自配置、manifest、history、summary 和 checkpoint 均保存在对应目录；
- 实现与运行说明：`PHASE2_PATH_ROBUSTNESS.md`。

## 6. Batch size 基准与最终决定

### 6.1 Phase 0 的 4 GiB RTX 3050 Ti 实测

使用与训练相同的 48,000 点前向、次级路径卷积和反向传播进行预热后测试：

| Batch size | 峰值预留显存 | 单步中位耗时 | GPU 吞吐 |
|---:|---:|---:|---:|
| **8** | 1.66 GiB | 0.209 s | 38.28 samples/s |
| 12 | 2.48 GiB | 0.289 s | 41.59 samples/s |
| 16 | 3.35 GiB | 29.438 s | 0.54 samples/s |

结论：

- batch 12 的纯 GPU 吞吐仅比 batch 8 高约 8.6%；
- 实际训练还包含 WAV 随机读取，加速幅度可能更低；
- batch 12 会把每轮更新次数从 64 降为 43；
- batch 16 虽未 OOM，但触发严重显存压力或卷积算法降速；
- 改变 batch 会改变优化轨迹，无法与现有 60 轮结果严格对比。

### 6.2 Phase 0 决定与 Phase 1 调整

> Phase 0 及其他 1 秒片段的严格消融继续保持 `batch_size = 8`。

Phase 1 改为 3.5 秒片段，并增加 STFT 复合损失，因此显存与样本时长条件已经改变。
实测采用：

```text
micro_batch_size      = 2
gradient_accumulation = 4
effective_batch_size  = 8
```

这样保留了有效 batch 8，同时把复合损失反向显存控制在约 1.46 GiB。后续实验应
根据片段长度区分“micro-batch”和“有效 batch”，不能再把物理 batch 8 作为所有
阶段无条件不变量。

---

## 7. Phase 0 产物索引

运行目录：`runs/phase0_60ep_seed2026/`

### 配置与日志

- [config.json](runs/phase0_60ep_seed2026/config.json)
- [data_split.json](runs/phase0_60ep_seed2026/data_split.json)
- [history.jsonl](runs/phase0_60ep_seed2026/history.jsonl)
- [summary.json](runs/phase0_60ep_seed2026/summary.json)

### Checkpoint

- [latest.pt](runs/phase0_60ep_seed2026/checkpoints/latest.pt)：第 60 轮；
- [best_mean_nr.pt](runs/phase0_60ep_seed2026/checkpoints/best_mean_nr.pt)：第 35 轮；
- [best_robust_nr.pt](runs/phase0_60ep_seed2026/checkpoints/best_robust_nr.pt)：第 45 轮。

### 图表

- [已见路径时域图](runs/phase0_60ep_seed2026/anc_seen_paths_time_result.png)
- [已见路径频域图](runs/phase0_60ep_seed2026/anc_seen_paths_freq_result.png)
- [未见路径时域图](runs/phase0_60ep_seed2026/anc_unseen_paths_time_result.png)
- [未见路径频域图](runs/phase0_60ep_seed2026/anc_unseen_paths_freq_result.png)

---

## 8. Phase 1 产物索引

正式运行目录：`runs/phase1_suite_seed2026/`

### P1-E0 基线

- [配置](runs/phase1_suite_seed2026/P1-E0/config.json)
- [验证 manifest](runs/phase1_suite_seed2026/P1-E0/validation_manifest.json)
- [完整基线指标](runs/phase1_suite_seed2026/P1-E0/summary.json)

### P1-E1 主频带实验

- [配置](runs/phase1_suite_seed2026/P1-E1/config.json)
- [逐轮历史](runs/phase1_suite_seed2026/P1-E1/history.jsonl)
- [完整总结](runs/phase1_suite_seed2026/P1-E1/summary.json)
- [综合分最优 checkpoint](runs/phase1_suite_seed2026/P1-E1/checkpoints/best_official_composite.pt)
- [主指标最优 checkpoint](runs/phase1_suite_seed2026/P1-E1/checkpoints/best_primary.pt)
- [反弹最优 checkpoint](runs/phase1_suite_seed2026/P1-E1/checkpoints/best_rebound.pt)

### P1-E2 复合目标实验

- [配置](runs/phase1_suite_seed2026/P1-E2/config.json)
- [逐轮历史](runs/phase1_suite_seed2026/P1-E2/history.jsonl)
- [完整总结](runs/phase1_suite_seed2026/P1-E2/summary.json)
- [综合分最优 checkpoint](runs/phase1_suite_seed2026/P1-E2/checkpoints/best_official_composite.pt)
- [主指标最优 checkpoint](runs/phase1_suite_seed2026/P1-E2/checkpoints/best_primary.pt)
- [反弹最优 checkpoint](runs/phase1_suite_seed2026/P1-E2/checkpoints/best_rebound.pt)
- [最新 checkpoint](runs/phase1_suite_seed2026/P1-E2/checkpoints/latest.pt)

### 套件汇总

- [suite_summary.json](runs/phase1_suite_seed2026/suite_summary.json)：最终选择 P1-E2；
- [PHASE1_V6.md](PHASE1_V6.md)：运行和复现说明。

---

## 9. 当前结论与已知问题

### 已确认的改进

- 有效训练更新从约 60 次增加到 3,840 次；
- Phase 0 旧版平均 NR 明显提高；
- 所有实验配置和数据划分可以追溯；
- 能够避免错误选择性能较差的最后一轮模型；
- 平均性能和最差路径性能分别保存；
- 已实现与官方误差小于 `1e-4 dB` 的可微 v6 指标；
- 已严格接入 0.5 秒初始化、六个 0.5 秒计分窗口和官方 scorer；
- P1-E2 主指标相对 Phase 0 提高 4.2109 dB；
- P1-E2 平均窗口反弹相对 Phase 0 降低 85.07%；
- 三个验证场景和十条路径的改善方向一致；
- 当前显卡能够稳定完成 3.5 秒、micro-batch 2、有效 batch 8 的复合损失训练。

### 尚未解决的问题

- 路径 7 的 v6 主指标仍只有 0.973 dB；
- 未见路径 10 的 v6 主指标仍只有 1.923 dB；
- P1-E2 最差窗口主指标仍只有 0.961 dB；
- 当前固定验证只有两个未见噪声及其切换组合，不能替代更多噪声和路径交叉验证；
- Phase 2 因最差路径低于 1.5 dB 提前停止，LOPO 基础设施已实现但按规则没有运行；
- 当前模型尚未实现正式提交需要的逐采样流式缓存接口；
- 当前模型是纯前馈结构，不能利用 `e[t-1]` 在线适应未知路径；
- Phase 2 开发消融只运行 seed 2026；由于硬停止条件已经失败，没有浪费算力执行三种子最终复验；
- 当前 checkpoint 尚未转换为 Participant Kit 正式提交结构。

---

## 10. 后续阶段记录模板

后续每项改进在本文件中按以下格式追加：

```markdown
## Phase N：阶段名称

### 目标

### 代码改动

### 与上一正式阶段的唯一变量

### 训练配置

### 实验结果

### 参数量与复杂度变化

### 与上一正式阶段对比

### 结论

### Checkpoint 与产物
```

后续路径鲁棒阶段的默认不变量：

```text
seed              = 2026
模型初始化         = Phase 1 P1-E2 第 19 轮
segment_duration  = 3.5 s
micro_batch_size  = 2
gradient_accumulation = 4
effective_batch_size  = 8
samples_per_epoch = 256
v6 验证 manifest 不变
训练/测试噪声划分不变
训练/测试路径划分不变
```

若必须改变任一不变量，需要在对应实验记录中明确说明原因。

---

## 11. Phase 3：反馈式 FIR 专家在线自适应

### 11.1 目标与实现

- 用 8×2048-tap 因果 FIR 专家替代高计算量 TCN 正式候选；
- 每 240 点用上一完整块的 10 维 `x/e/y` 统计更新 GRU 门控；
- 严格实现 `process_sample(x[t], e[t-1])`，当前误差不能影响当前输出；
- 训练闭环保持 1,966 点次级路径历史，并在初始化、计分窗和路径切换处连续；
- 路径 1～8 用于开发，路径 9/10 由最终脚本物理隔离；
- 新增公共 FIR 蒸馏、oracle、冻结门控、联合微调、LOPO、最终评估和提交导出入口。

默认 hidden 24 模型为 19,176 参数，稳态 2,055 MAC/sample，块边界峰值
21,079 MAC。Participant Kit 逐采样检查通过，4096 点 CPU RTF=0.678，完整
3.5 秒公开闭环 RTF=0.798。

### 11.2 正式实验结果

固定开发基线 P1-E2（路径 1～8）为 `S=12.0627`、`R=1.8670`、最差路径
`0.9733`、首窗 `12.0558`。

| 实验 | S | R | D | 最差路径 | 首窗 | 路由准确率 | 结果 |
|---|---:|---:|---:|---:|---:|---:|---|
| P3-E0 公共 FIR | 12.7009 | 1.8977 | 11.3730 | 1.0217 | 12.7041 | — | 完成 |
| P3-E1 oracle | 16.5658 | 0.1632 | 16.9439 | 4.1513 | 16.6058 | oracle | 通过 |
| P3-E2 hidden 24 | 8.2564 | 4.2849 | 6.9139 | 1.4460 | 8.4844 | 11.91% | 失败 |
| P3-E2b hidden 32 | 4.4530 | 4.8338 | 3.3707 | 1.5416 | 4.6647 | 21.51% | 失败 |

P3-E1 同时满足主指标、反弹和最差路径上限要求，因此未触发 4096-tap 修正。
P3-E2 严重低于开发门槛后，按计划只执行一次 hidden 32 确定性修正；修正仍未通过。

### 11.3 结论与停止项

Phase 3 **未通过**。oracle 结果排除了 FIR 专家容量不足这一解释，主要失败点是当前
10 维块级反馈特征和监督方式不能从闭环残差中稳定辨识路径。增大 GRU hidden 并未
解决可观测性问题，反而降低声学指标，因此不应继续盲目扩大门控网络。

按硬门槛执行以下停止项：

- 未运行 P3-E3 联合鲁棒微调；
- 未运行八折 LOPO；
- 未运行 seeds 2027/2028；
- 未访问路径 9/10；
- 未把 oracle 分数或训练损失作为正式升级依据；
- 正式 checkpoint 保留 P1-E2。

正式套件位于 `runs/phase3_suite_seed2026_v2/`，汇总见
`runs/phase3_suite_seed2026_v2/suite_summary.json`，实现说明见
`PHASE3_FEEDBACK_FIR.md`。

---

## 12. Phase 3R：基于多模型创新误差的 FIR 专家路由修复

### 12.1 实现与固定模板

- 用训练路径 1～8 的显式创新误差替代 10 维统计、GRU 和路径分类损失；
- `P_i` 只使用六条训练噪声从 20 秒到末尾的全部对齐样本生成；
- 模板使用 4096 点对称 Hann、240 点 hop、50 Hz～8 kHz，并固化输入与产物 SHA-256；
- 候选次级路径卷积采用带 1,966 点历史的 240 点 overlap-save，与直接时域卷积一致；
- 路由严格接收 `e[t-1]`，4096 点历史未满时保持均匀专家权重；
- 推理模板为 buffer，训练参数量为 0；稳态 2,048 MAC/sample，估算平均 21,436 MAC/sample；
- 模板 SHA-256 为 `c08921c5c00f940cf29bdc09b959da5ea2e594da50ea6e0718338d1778d4d64e`。

模板与 manifest 位于 `artifacts/phase3r_innovation_templates.npz` 和
`artifacts/phase3r_innovation_templates.manifest.json`。manifest 自动拒绝少于/多于六条训练噪声、
非路径 1～8、出现 `scene_09/scene_10` 或产物 SHA 不匹配的情况。

### 12.2 固定开发候选

| 候选 | S | R | D | 最差路径 | 首窗 | 路由准确率 | 最慢三连正确恢复 | CPU RTF | 结果 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| P3R-E1a | 16.4634 | 0.2056 | 16.8628 | 4.1513 | 16.6221 | 99.77% | 75 ms | 0.757 | 通过 |
| P3R-E1b | 16.3529 | 0.2828 | 16.7602 | 4.1513 | — | 99.51% | 70 ms | 0.744 | 通过 |
| P3R-E1c | **16.5705** | **0.1899** | **16.9449** | **4.1513** | **16.6345** | **100%** | 85 ms | **0.704** | 选中 |

三个无探测候选全部通过路由、切换恢复、声学、安全、有限值和 CPU 门槛，因此按规则
没有运行 P3R-E2 PN 回退。开发选择只在固定三候选中进行，没有继续搜索窗长或超参数。

### 12.3 LOPO 结果与正式结论

| 留出路径 | 相对 P1-E2 主指标增益 |
|---:|---:|
| 1 | -1.1737 dB |
| 2 | +0.7548 dB |
| 3 | -1.0636 dB |
| 4 | -4.5874 dB |
| 5 | -2.9186 dB |
| 6 | +2.3064 dB |
| 7 | +0.6187 dB |
| 8 | -4.5710 dB |

LOPO 中位增益 `-1.1187 dB`，仅 3/8 折不退化，两个门槛均失败。这说明创新路由已经
解决“已知候选路径辨识”问题，但当前离散专家库不能可靠泛化到候选集合之外的路径；
失败原因不再是 10 维反馈特征，而是专家控制器/路径模板的插值与外推能力。

按停止规则：不运行 seeds 2027/2028，不读取路径 9/10，不进行正式升级。正式 checkpoint
继续保留 `runs/phase1_suite_seed2026/P1-E2/checkpoints/best_official_composite.pt`。
开发套件见 `runs/phase3r_suite_seed2026/suite_summary.json`，LOPO 见
`runs/phase3r_lopo_seed2026/lopo_summary.json`。

---

## 13. Phase 3G：冻结权重的创新条件生成式 FIR

### 13.1 实现

- 保留 P3R-E1c 的 4096 点创新窗、240 点更新周期和解析 `alpha`；
- 新增 `GRUCell(52,32)`、16 维 latent head 和 `16×2048` 可训练残差字典；
- 有效控制核为 `alpha @ frozen_experts + z @ dictionary`，默认可训练参数精确为 41,552；
- 推理只更新 hidden、latent 和卷积缓存，参数与 `state_dict` 保持不可变；
- 新增 DTW 半径 16 的确定性路径对齐、30% 插值、10% 外推、10% 有界增强和候选遮蔽；
- 新增完整 168,000 点可微闭环、24000 点 TBPTT、v6 复合损失及 FIR 能量/变化正则；
- 新增固定 48 条连续路径压力 manifest、严格八折 LOPO、oracle-latent E3 分支和三种子最终脚本；
- 最终评估脚本在 LOPO 和三种子开发门槛通过前拒绝读取路径 9/10。

### 13.2 实现验证

| 验证 | S | R | 最差路径 | CPU RTF | 结论 |
|---|---:|---:|---:|---:|---|
| P3G-E1 单 batch 烟测 | 16.5668 | 0.1800 | 4.1481 | 0.590 | 链路通过 |
| P3G-E2 单 batch 烟测 | 16.5463 | 0.1711 | 4.1446 | 0.595 | 链路通过 |

- 34 项全项目测试全部通过，包括与官方 scorer 的 v6 一致性；
- 批量闭环与正式逐点接口误差不超过 `2e-6`；
- 当前误差不能影响当前输出，reset 后输出和路由轨迹完全一致；
- 完整开发推理前后的状态字典 SHA-256 一致；
- Participant Kit 最终导出包 5000 点检查通过：41,552 参数、峰值输出 0.0447、CPU RTF 0.824；
- Participant Kit 完整 3.5 秒公开闭环通过：`S=23.1103`、`R=0.0064`、CPU RTF 0.939；
- 将复频谱功率由 `abs()²` 改为数学等价的 `real²+imag²`，避免离线 CUDA 环境依赖 NVRTC；
- 正式 E1/E2、48 条压力集和 LOPO 尚未运行，不能据烟测结果宣称 Phase 3G 成功；
- 路径 9/10 未访问，正式模型继续保留 P1-E2。

详细命令见 `PHASE3G_GENERATIVE_FIR.md`。烟测产物位于
`runs/phase3g_smoke_warmup_v4_seed2026` 和 `runs/phase3g_smoke_generalize_seed2026`。

---

## 14. Phase 4R：运行时加固与最差路径诊断

### 14.1 E09-A 等价运行时加固（通过）

- profile 确认主要浪费来自包装器每个采样点重复进入 `torch.inference_mode()`，而非
  候选路径 FFT 重复计算；
- 删除逐采样推理上下文，保留 240 点 GRU/Linear 更新处的局部推理模式；
- v2 三次完整闭环 RTF P50/最大为 `0.9427/0.9620`；v3 十次 P50/P95/最大为
  `0.6760/0.7000/0.7082`，十次均满足 `RTF≤0.8`；
- 块边界 P99 从 `1.7537 ms` 降到 `1.6023 ms`，最大 `2.2348 ms`；
- 168,000 点控制输出和闭环残差 `array_equal=True`，最大误差为 0；
- 参数量 41,552、峰值事件 MAC 2,484,720、权重 SHA-256、`state_dict` SHA-256 和
  官方 `S=25.1484/R=0.1373` 均不变；
- 达到停止条件后未执行 E09-B，也未采用延迟 FIR 更新、近似 FFT、量化或 JIT。

正式包由 v2 升级为 `phase3g_submission_final_seed2027_v3`。完整计时与哈希见
`artifacts/phase4r_runtime_report.json`。

### 14.2 最差路径因果诊断（诊断完成）

- 冻结 seed 2026 八折 LOPO checkpoint 和正式 v3，不修改模型或权重；
- 部署证据通道保持 held-out 路径隔离，Oracle 通道只在 checkpoint 冻结后使用 held-out
  数据；8/8 物理隔离通过，Path 9/10 未进入诊断训练输入；
- 历史 LOPO 最大复现误差 `7.11e-15`；
- Path 8 独立评估集的最佳保留专家/静态 simplex/静态加字典/块级加字典/自由 FIR
  主指标分别为 `12.9869/16.5526/18.9901/18.9901/10.3125 dB`；
- 旧 `_oracle_latent_filter` 被确认只是对旧专家的参数投影，不能证明容量上限；
- 动态块级 Oracle 未优于静态 Oracle，自由 FIR 也没有给出支持扩大记忆的单调证据；
- 按“覆盖/字典 → 映射 → 路由 → 记忆”的优先级，冻结根因为
  `training_coverage_or_dictionary`，唯一后续方案为 E10-A。

详细报告见 `PHASE4R_WORST_PATH_DIAGNOSIS.md` 与
`artifacts/phase4r_worst_path_diagnosis.json`。

### 14.3 E10-A 三邻居覆盖修正（未通过）

E10-A 保持模型、损失和训练参数不变，只将 interpolate/extrapolate 的第二路径端点从
确定性单最近邻改为多声学空间平均名次下三个最近 retained 路径的均匀采样。

| Path | 冻结主增益 | E10-A 主增益 | 变化 | E10-A 反弹 |
|---:|---:|---:|---:|---:|
| 1 | 1.1602 | 1.1514 | -0.0087 | 2.6246 |
| 2 | 2.6055 | 2.7330 | +0.1275 | 5.1187 |
| 3 | 1.1473 | 0.6599 | -0.4873 | 2.8762 |
| 4 | -0.1347 | -0.2973 | -0.1625 | 6.8781 |
| 5 | 0.2235 | -0.3896 | -0.6131 | 9.6728 |
| 6 | 2.5421 | 2.5147 | -0.0273 | 5.2455 |
| 7 | 0.6751 | 0.6677 | -0.0074 | 0.0200 |
| 8 | -2.3413 | -1.7770 | +0.5643 | 10.5620 |

预注册验收结果：Path 8 主增益、Path 8 反弹、最差三覆盖折至少改善两折、八折中位
增益和至少 6/8 不退化均失败；只有 8/8 物理隔离通过。最终中位增益为 `0.6638 dB`，
非退化折为 5/8。该结果表明扩大邻居覆盖对 Path 8 有部分作用，但三邻居均匀采样会
破坏其他路径的局部映射精度，不能作为正式升级。

按停止规则不运行 seeds 2027/2028，不修改正式 checkpoint、v3、`dist/CCFANC.zip` 或
技术报告。完整八折结果见 `runs/phase4r_e10a_seed2026_lopo/lopo_summary.json`。

---

## 15. 更新日志

### 2026-08-12：Phase 4R 运行时加固通过，E10-A 未通过

- E09-A 等价运行时优化通过，正式提交包升级为 v3；
- 完成最差路径两通道因果诊断，冻结根因为训练覆盖或字典子空间；
- 完成 E10-A seed 2026 八折严格 LOPO，Path 8 改善但全部预注册泛化门槛未通过；
- 停止 E10-A 三种子阶段，正式 v3 与交付 ZIP 保持不变；
- 全项目 56/56 单元测试通过。

### 2026-08-10：Phase 3G 完整实验与正式升级通过

- 完成 seed 2026 的 E1/E2 正式训练；开发 `S=19.5192`、`R=0.3395`、`D=19.5865`；
- 48 条不可见连续路径压力集中位主指标增益 `+2.7751 dB`，91.67% 场景不退化；
- 修复压力集逐记录反弹字段与 LOPO 七路径/八路径暖启动基线口径，并新增专项回归测试；
- 严格八折 LOPO 全部完成：中位增益 `+0.9112 dB`、6/8 折不退化、8/8 物理隔离通过；
- LOPO 已通过，因此按计划未运行 E3；
- seeds 2026/2027/2028 均通过开发、压力、切换、峰值、finite、状态不可变与实时门槛；
- 首次解封路径 9/10 后，三个种子均通过十路径正式升级门槛；按开发 D 选中 seed 2027；
- seed 2027 十路径 `S=17.9585`、`R=1.6870`、`C=12.0648`，较 P1-E2 分别改善
  `+6.9795 dB`、`-0.0522 dB` 和 `+4.9013`；
- 十路径最差/路径 10 为 `3.9775 dB`，路径 9 为 `15.2938 dB`，路径 9/10 平均增益 `+2.9915 dB`；
- 38/38 全量测试通过；正式 v2 包剔除训练期随机辅助代码，Participant Kit validate RTF 0.695，
  完整公开闭环 `S=25.1484/R=0.1373/RTF=0.959`；
- 正式 checkpoint 升级为 `runs/phase3g_suite_seed2027/P3G-E2/checkpoints/best_phase3g_selection.pt`，
  正式导出包为 `phase3g_submission_final_seed2027_v2`。

### 2026-08-09：Phase 3G 生成式 FIR 基础设施与烟测完成

- 新增冻结权重创新条件 GRU、16 维残差字典和严格逐点反馈接口；
- 新增 DTW 连续路径分布、候选遮蔽、可微闭环训练、压力集、LOPO、E3 和最终封存编排；
- E1/E2 单 batch GPU 烟测、34 项回归测试和 Participant Kit 官方检查通过；
- 推理参数哈希保持不变，公开 5000 点 RTF 0.824，完整公开闭环 RTF 0.939；
- 完整训练、压力集和 LOPO 尚未运行，路径 9/10 未访问，正式 checkpoint 仍为 P1-E2。

### 2026-08-09：Phase 3R 创新误差路由开发通过、LOPO 未通过

- 新增确定性全量传递模板生成、SHA-256 manifest 与路径 9/10 封存断言；
- 新增显式创新误差路由、三套固定无探测配置和有界 PN 回退决策；
- 候选次级路径卷积由逐采样直接卷积优化为严格 overlap-save，CPU RTF 降至 1 以下；
- 三个无探测候选均通过开发门槛，P3R-E1c 以最高 D 选中，PN 未运行；
- 八折 LOPO 中位增益 -1.1187 dB、仅 3/8 不退化，阶段按规则停止；
- 未运行三种子复训，未访问路径 9/10，正式 checkpoint 保留 P1-E2。

### 2026-08-09：Phase 3 反馈式 FIR 专家正式结论

- 新增 8×2048 因果 FIR 专家、10 维反馈统计、hidden 24 GRU 和 240 点块级门控；
- 训练 rollout 与正式 `process_sample(x[t], e[t-1])` 已实现逐点一致的反馈延迟；
- 新增公共 FIR 蒸馏、oracle 专家、冻结门控和联合鲁棒微调四阶段入口；
- 新增真实路径切换、窄范围增强、oracle/development 硬停止、LOPO 和三种子最终编排；
- 路径 9/10 只可由最终评估脚本在开发和 LOPO 通过后读取；
- 默认模型 19,176 参数、稳态 2,055 MAC/sample、峰值 21,079 MAC；
- 新增完整闭环梯度、流式一致性、反馈延迟、因果性、reset、数据隔离和复杂度测试；
- P3-E0/E1/E2/E3 四阶段一个 batch 的 GPU 烟雾链路均已通过；
- Participant Kit 提交检查通过，4096 点 CPU RTF=0.678；完整 3.5 秒公开闭环 RTF=0.798；
- 正式 P3-E0 公共 FIR 达到 `S=12.7009`，P3-E1 oracle 达到 `S=16.5658`、最差路径 `4.1513`；
- hidden 24/32 门控均未通过开发门槛，路由准确率仅 11.91%/21.51%；
- 按停止规则未运行联合微调、LOPO、三种子和路径 9/10 最终评估；
- Phase 3 记录为未通过，正式 checkpoint 保留 P1-E2；
- 详细说明见 `PHASE3_FEEDBACK_FIR.md`。

### 2026-08-08：Phase 2 路径鲁棒训练与正式结论

- 新增 `phase2_paths.py`，实现同参考片段的多路径联合采样与各路径独立 disturbance；
- 路径组固定保留原始真实成员，训练路径严格限制为 1～8；
- 新增 ±1.5 dB、±2 整数采样、-35～-30 dB 平滑尾部的保守 IR 增强；
- 新增 `mean_path + beta * top-25%` 鲁棒损失，K=4 时 top-25% 即硬最坏成员；
- 路径输出护栏每个参考样本只计算一次，不随路径组大小重复放大；
- 新增 development、固定 synthetic stress、final 三套 manifest；
- 路径 9/10 不参与梯度、逐轮验证、checkpoint 选择或 E3 判断；
- 新增 P2-E0/E1/E2/条件 E3 和三种子 P2-E4 自动编排；
- 新增八折成对 control/augment Leave-One-Path-Out 编排；
- 新增 Phase 2 全局、未见路径和 LOPO 硬验收逻辑；
- 保持 `TimeDomainANC` 结构和 42,764 参数不变；
- 正式 P2-E0/E1/E2 和条件 E3 均已完成，开发 robust 最高为 E1 的 10.6433；
- E2/E3 最好最差路径仅 0.9822 dB，低于 1.5 dB 停止阈值；
- 按规则未运行 LOPO/P2-E4，未访问路径 9/10，Phase 2 记录为未通过；
- 正式 checkpoint 保留 P1-E2，并建议下一步进入带残差反馈的 Phase 3；
- 详细设计、命令与结论见 `PHASE2_PATH_ROBUSTNESS.md`。

### 2026-08-08：Phase 1 v6 评分对齐实现

- 新增 `v6_metrics.py`，实现可微六窗口 1/3 倍频带主指标与反弹损失；
- 使用官方硬频带最大反弹，不再采用路线图草案中的平滑最大值；
- 时域项改为尺度无关残差能量比，控制项改为超过 1.0 才触发的安全护栏；
- 新增固定三场景验证：车载连续、餐厅连续、2.0 秒处车载切换餐厅；
- 验证直接调用 Participant Kit 1.5.0 的 `score_windowed_signals()`；
- 新增独立 `train_phase1.py`，不改变 Phase 0 训练入口；两者现归档于 `legacy_models/phase0_phase1/`；
- 新增 E0/E1/E2/条件 E3 自动编排与正式硬门槛选择；
- 七项单元测试全部通过，包含与官方 scorer 的 `1e-4 dB` 一致性检查；
- P1-E0 固定验证基线：主指标 6.7681 dB、反弹 11.6514 dB、综合分 1.2423；
- 一个 batch 的 GPU 烟雾训练完成，反向、验证、checkpoint 和验收链路正常；
- P1-E1 最终主指标 10.9839 dB、反弹 9.4301 dB、综合分 4.8597，因反弹降幅
  19.1% 略低于 20% 硬门槛而未通过；
- P1-E2 第 19 轮最优：主指标 10.9790 dB、反弹 1.7392 dB、综合分 7.1635；
- P1-E2 相对基线主指标提高 4.2109 dB、反弹降低 85.07%、综合分提高 5.9213，
  控制峰值 0.0663，全部硬门槛通过，因此未触发 E3；
- 当前最差路径主指标仍为 0.9733 dB，留待路径鲁棒阶段解决；
- 详细运行方法见 `PHASE1_V6.md`。

### 2026-08-07

- 完成 Phase 0 数据与训练基础设施改造；
- 完成 60 轮、每轮 512 样本的正式强基线训练；
- 确定平均最优 checkpoint 为第 35 轮；
- 确定稳健最优 checkpoint 为第 45 轮；
- 完成 batch 8/12/16 的显存和吞吐测试；
- 决定后续严格消融继续保持 `batch_size = 8`；
- 建立本持续改进记录文件。
