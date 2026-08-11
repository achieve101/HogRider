# Phase 3：反馈式 FIR 专家在线自适应

Phase 3 不把 P1 TCN 驻留在正式候选中。P1-E2 只用于公共 FIR 蒸馏和性能对照；
正式结构由 8 个因果 FIR 专家与上一采样点残差驱动的块级 GRU 门控组成。

## 已实现结构

```text
8 × 2048-tap causal FIR experts
10-dimensional causal block statistics
GRUCell(10, 24) + Linear(24, 8)
block size = 240 samples = 5 ms
alpha = 0.8 previous + 0.2 softmax(logits)
output = (0.98 - 1e-6) tanh(raw / (0.98 - 1e-6))
```

反馈统计只使用已经完成块中的 `x/e/y` RMS，以及 `e-x`、`e-y` 的固定滞后相关。
训练 rollout 和提交 `process_sample(x[t], e[t-1])` 使用相同的“一块延迟后更新”语义。
初始化 24,000 点和每个计分窗口都恰好包含 100 次门控更新，边界处不重置状态。

默认模型共 19,176 参数，稳态 2,055 MAC/sample，块边界峰值 21,079 MAC，低于
P1 TCN 的 42,048 MAC/sample。Participant Kit 的 4,096 点 CPU 严格逐采样检查已通过，
实时因子为 0.678；完整 3.5 秒公开闭环实时因子为 0.798。烟雾 checkpoint 的声学
得分没有意义，公开 demo 只用于接口和速度检查。

## 训练阶段

```powershell
# P3-E0：P1 蒸馏 + 公共 FIR 声学微调
conda run --no-capture-output -n pytorch2.5.1_py310 python -u -m legacy_models.phase3.train_phase3 `
  --stage common --output-dir runs/phase3_P3-E0_seed2026 --device cuda

# P3-E1：路径 1～8 oracle 专家上限
conda run --no-capture-output -n pytorch2.5.1_py310 python -u -m legacy_models.phase3.train_phase3 `
  --stage expert `
  --checkpoint runs/phase3_P3-E0_seed2026/checkpoints/best_phase3_selection.pt `
  --output-dir runs/phase3_P3-E1_seed2026 --device cuda

# P3-E2：冻结专家，只训练反馈门控
conda run --no-capture-output -n pytorch2.5.1_py310 python -u -m legacy_models.phase3.train_phase3 `
  --stage gate --checkpoint <P3-E1-checkpoint> `
  --output-dir runs/phase3_P3-E2_seed2026 --device cuda

# P3-E3：路径切换和窄范围增强下联合微调
conda run --no-capture-output -n pytorch2.5.1_py310 python -u -m legacy_models.phase3.train_phase3 `
  --stage joint --checkpoint <P3-E2-checkpoint> `
  --output-dir runs/phase3_P3-E3_seed2026 --device cuda
```

自动决策树使用 `python -m legacy_models.phase3.run_phase3_experiments`。它会先检查 2048-tap oracle；不达标时
只允许一次 4096-tap 修正。oracle 仍失败则停止，不训练门控。反馈开发门槛不通过时，
只允许一次 hidden 32 修正；门控本身未通过开发门槛时不会进入联合微调。所有开发
实验都不会读取路径 9/10。

## 数据隔离和验收

- 排序后前六条噪声训练，vehicle/restaurant 两条噪声固定验证；
- 路径 1～8 用于训练、development、路径切换 stress 和 LOPO；
- 路径 9/10 仅允许 `phase3_final_evaluation.py` 在三种子和 LOPO 通过后读取；
- 开发选模使用 `C + 0.5 worst_path + 0.2 first_window`；
- oracle 最差路径必须至少 2.0 dB；反馈候选在进入 LOPO 前最差路径至少 1.5 dB；
- 最终十路径、路径 10、未见路径增益、安全和 CPU RTF 门槛在最终脚本中统一执行。

## 提交导出

```powershell
conda run --no-capture-output -n pytorch2.5.1_py310 python export_phase3_submission.py `
  --checkpoint <accepted-checkpoint> `
  --output phase3_submission
```

导出目录自包含模型代码、权重、配置、runtime 和固定 `create_model(device)` 入口，
只依赖 PyTorch、NumPy 和 Python 标准库。

## 当前状态

实现、22 项单元测试、四阶段 GPU 烟雾训练、导出包检查和完整公开闭环均已通过。
seed 2026 正式开发结果如下：

| 实验 | 主指标 S | 反弹 R | 最差路径 | 首窗 | 路由准确率 | 结论 |
|---|---:|---:|---:|---:|---:|---|
| P3-E0 公共 FIR | 12.7009 | 1.8977 | 1.0217 | 12.7041 | — | 线性公共 FIR 可替代 TCN 作为基础控制器 |
| P3-E1 oracle 专家 | 16.5658 | 0.1632 | 4.1513 | 16.6058 | 100% oracle | 容量门槛通过，无需 4096 taps |
| P3-E2 hidden 24 | 8.2564 | 4.2849 | 1.4460 | 8.4844 | 11.91% | 开发门槛失败 |
| P3-E2b hidden 32 | 4.4530 | 4.8338 | 1.5416 | 4.6647 | 21.51% | 确定性修正仍失败 |

oracle 的强结果说明 8 个 FIR 专家具有足够容量；失败集中在当前 10 维、5 ms 块统计
无法可靠辨识次级路径。按硬停止规则未运行 P3-E3、LOPO、三种子复训或路径 9/10
最终评估。Phase 3 因此记录为未通过，正式模型继续保留 P1-E2。完整机器可读结论见
`runs/phase3_suite_seed2026_v2/suite_summary.json`。
