# Phase 1：官方 v6 评分对齐

Phase 1 保持 `TimeDomainANC` 网络结构不变，将训练片段、损失、验证和 checkpoint
选择切换到 Participant Kit 1.5.0 / 协议 v6。Phase 0/1 历史入口现位于
`legacy_models/phase0_phase1/`，算法内容保持不变。

## 已实现内容

- 0.5 秒初始化加 3 秒计分的完整 3.5 秒训练；
- 六个互不重叠的 0.5 秒计分窗口；
- 与官方一致的 Hann/STFT/1/3 倍频带功率计算；
- 50 Hz～5 kHz 未加权主指标损失；
- 1 kHz～8 kHz 每窗口最坏频带反弹损失；
- 尺度无关时域稳定项和仅在输出超过 1.0 时触发的安全护栏；
- 车载、餐厅、车载切换餐厅三个固定验证场景；
- 直接调用 `DEEPANC_PARTICIPANT_KIT/public_demo_scoring.py` 验证；
- 综合、主指标、反弹和最新四类 checkpoint；
- E0/E1/E2 以及有条件 E3 的自动实验编排。

## 环境

GPU 训练必须通过完整 Conda 环境启动，使 CUDA/NVRTC 动态库路径正确加载：

```powershell
conda run --no-capture-output -n pytorch2.5.1_py310 python -u -m legacy_models.phase0_phase1.train_phase1 --help
```

不要直接运行环境目录中的 `python.exe`；该方式不会自动补齐 CUDA DLL 搜索路径。

## P1-E0 正式基线

```powershell
conda run --no-capture-output -n pytorch2.5.1_py310 python -u -m legacy_models.phase0_phase1.train_phase1 `
  --evaluate-only `
  --output-dir runs/phase1_P1-E0_seed2026 `
  --device cuda
```

当前 Phase 0 第 35 轮 checkpoint 在固定三场景验证集上的结果：

```text
primary_score_db = 6.7681 dB
rebound_score_db = 11.6514 dB
selection_score  = 1.2423
```

## 正式快速实验结果

固定种子 2026、20 轮、每轮 256 个 3.5 秒样本的正式套件已经完成：

| 实验 | 主指标 S | 反弹 R | 综合分 C | 验收 |
|---|---:|---:|---:|---|
| P1-E0 Phase 0 | 6.7681 | 11.6514 | 1.2423 | 对照 |
| P1-E1 主频带 | 10.9839 | 9.4301 | 4.8597 | 未通过：反弹降幅 19.1% |
| P1-E2 70/30 复合 | **10.9790** | **1.7392** | **7.1635** | **通过** |

P1-E2 最佳 checkpoint 位于：

```text
runs/phase1_suite_seed2026/P1-E2/checkpoints/best_official_composite.pt
```

最佳轮次为 19。相对 P1-E0，主指标提高 4.2109 dB，反弹降低 85.07%，综合分
提高 5.9213，控制输出峰值为 0.0663，所有硬门槛均通过。E2 已达标，因此没有
触发 E3。最差路径主指标仍只有 0.9733 dB，属于后续路径鲁棒阶段需要解决的问题。

## 单独运行 E1 或 E2

```powershell
# P1-E1：主频带对齐
conda run --no-capture-output -n pytorch2.5.1_py310 python -u -m legacy_models.phase0_phase1.train_phase1 `
  --experiment band `
  --output-dir runs/phase1_P1-E1_seed2026

# P1-E2：70/30 主指标与反弹复合目标
conda run --no-capture-output -n pytorch2.5.1_py310 python -u -m legacy_models.phase0_phase1.train_phase1 `
  --experiment composite `
  --output-dir runs/phase1_P1-E2_seed2026
```

默认使用 micro-batch 2、梯度累积 4、有效 batch 8，最多训练 20 轮，并以官方
综合分早停。两次实验都从同一个 Phase 0 checkpoint 开始，不继承优化器状态。

## 自动运行完整快速实验

```powershell
conda run --no-capture-output -n pytorch2.5.1_py310 python -u -m legacy_models.phase0_phase1.run_phase1_experiments `
  --output-root runs/phase1_suite_seed2026 `
  --device cuda
```

编排器依次运行 E0、E1、E2。E2 主指标下降超过 0.25 dB 时，E3 自动改用
`0.85/0.15`；否则反弹降幅不足 20% 时自动改用 `0.60/0.40`。最终只从满足全部
硬门槛的候选中按综合分选择模型；没有候选通过时继续保留 Phase 0。

## 测试

```powershell
conda run --no-capture-output -n pytorch2.5.1_py310 python -m unittest discover -s tests -v
```

测试覆盖官方 scorer 数值一致性、六窗口 dB 聚合、初始化段排除、缩放解析解、
反向梯度、安全护栏、固定场景形状和随机采样复现。

## 阶段边界

本阶段仍是整段因果训练和离线验证，不包含逐采样提交接口、卷积缓存、反馈分支、
路径增强或 CVaR，因此尚不能直接作为正式提交包。
