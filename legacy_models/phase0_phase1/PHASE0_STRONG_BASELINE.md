# Phase 0 强基线使用说明

> 本文件侧重 Phase 0 的运行方法。阶段改动、正式实验结果、batch-size 决策及后续模型改进统一记录在 [MODEL_IMPROVEMENT_LOG.md](../../MODEL_IMPROVEMENT_LOG.md)。

Phase 0 保持 `TimeDomainANC` 网络结构和时域 MSE 目标不变，重点修复训练与实验基础设施，使后续模型改进有可信、可复现的对照。

## 已完成的改进

- 大小写不敏感地扫描 WAV，完整识别 8 类噪声；
- 兼容普通的 `KTV_scene_01.wav` 和特殊的 `餐厅.WAV_scene_01.wav` 命名；
- 在 Dataset 初始化阶段检查原始噪声、期望噪声、采样率和路径索引；
- 使用虚拟 epoch，每轮默认随机抽取 512 个片段；
- 训练路径循环均衡采样，噪声和音频起点随机采样；
- 修复测试拼接每边误读 1 秒、最终返回 2 秒的问题；
- 固定 Python、NumPy、PyTorch、CUDA/CuBLAS 随机性；
- 自动记录配置、数据划分和逐轮 JSONL 日志；
- 定期执行已见/未见路径验证；
- 保存最新、最佳平均 NR 和最佳最差路径 checkpoint；
- 最终自动加载最佳平均 NR checkpoint 生成时域图、频域图和总结。

## 默认训练

使用本机 Conda 环境：

```powershell
conda run --no-capture-output -n pytorch2.5.1_py310 python -u train.py
```

默认参数：

```text
epochs              = 60
samples_per_epoch    = 512
batch_size           = 8
optimizer steps/epoch= 64
validation_interval  = 5
seed                 = 2026
device               = auto
```

这是真正的强基线训练：60 轮共有约 3840 次参数更新，而旧代码的 60 轮只有约 60 次更新，因此运行时间会明显增加。

## 较短的首次实验

建议先运行：

```powershell
conda run --no-capture-output -n pytorch2.5.1_py310 python -u train.py `
  --epochs 20 `
  --samples-per-epoch 128 `
  --batch-size 8 `
  --validation-interval 5 `
  --seed 2026
```

确认结果和耗时后，再增加到默认的 60 × 512 配置。

## 常用参数

```text
--dataset-dir             数据目录，默认 dataset
--output-dir              输出目录；不指定时自动创建带时间戳的目录
--epochs                  训练轮数
--samples-per-epoch       每轮随机训练片段数
--batch-size              batch size
--segment-duration        训练片段秒数，默认 1.0
--learning-rate           Adam 学习率，默认 0.001
--validation-interval     每隔多少轮验证
--num-workers             DataLoader worker 数，Windows 默认推荐 0
--seed                    随机种子
--device                  auto、cpu、cuda 或 cuda:N
```

## 输出结构

每次正式运行会生成独立目录：

```text
runs/phase0_YYYYMMDD_HHMMSS/
├─ config.json
├─ data_split.json
├─ history.jsonl
├─ summary.json
├─ checkpoints/
│  ├─ latest.pt
│  ├─ best_mean_nr.pt
│  └─ best_robust_nr.pt
├─ anc_seen_paths_time_result.png
├─ anc_seen_paths_freq_result.png
├─ anc_unseen_paths_time_result.png
└─ anc_unseen_paths_freq_result.png
```

Checkpoint 包含：

- 模型参数；
- optimizer 状态；
- epoch；
- 完整命令配置；
- 数据划分；
- 最近一次验证指标。

## 当前数据划分

文件名按大小写不敏感方式排序，最后两类作为未见测试噪声。当前数据会得到：

```text
训练噪声：KTV、公交、厨房、地铁、步行街、火车
测试噪声：车载、餐厅
训练路径：1～8
测试路径：9～10
```

划分会写入每次运行的 `data_split.json`，避免实验结束后无法确认实际使用的数据。

## 指标说明

Phase 0 仍使用旧版整段能量 NR：

```text
NR = 10 log10(sum(d²) / sum(e²))
```

其中：

- `best_mean_nr.pt`：十条验证路径的平均 NR 最优；
- `best_robust_nr.pt`：十条验证路径中最差一条的 NR 最优。

正式 v6 六窗口 1/3 倍频带指标和高频反弹损失属于 Phase 1，本阶段没有混入，以便明确衡量 Phase 0 工程修正本身的影响。
