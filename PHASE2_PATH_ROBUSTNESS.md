# Phase 2：路径鲁棒训练

Phase 2 从 Phase 1 的 P1-E2 第 19 轮初始化，保持 `TimeDomainANC` 和 42,764 个参数
不变，只改变训练路径采样、鲁棒目标、验证隔离和实验选择。

## 现实修正

- 纯前馈模型对相同参考输入只能产生相同控制输出，多路径联合训练学习的是稳健折中，
  不是对未知次级路径的在线辨识；若本阶段仍不能改善路径 7/10，应进入 Phase 3 反馈结构。
- 路径 7/10 的问题更像结构差异和可控性不足，不先对某些陷波频带降权，也不移动人工陷波。
- 首轮增强限制为 ±1.5 dB、±2 个整数采样和 -35～-30 dB 平滑尾部扰动；小数延迟、
  时间伸缩、任意 IR 插值和逐 tap 白噪声暂缓。
- 路径 9/10 永不参与梯度、逐轮验证、checkpoint 选择或 E3 超参数判断。

## 实现

- `phase2_paths.py`
  - `Phase2GroupedDataset` 为同一参考片段加载各真实路径对应的 disturbance；
  - 每组至少包含锚点原始路径和另一条未增强真实路径；
  - `augment_secondary_path()` 实现确定性、定长、因果的保守 IR 增强；
  - `compute_phase2_group_loss()` 实现 `mean_path + beta * top_25_percent`。
- `legacy_models/phase2/phase2_validation.py`
  - development：路径 1～8，参与逐轮选模；
  - stress：路径 1～8 的固定确定性增强，只用于固定压力评估；
  - final：路径 9/10，仅最终候选显式请求时评估；
  - development 选模分数为 `C + 0.25 * P25(path_primary)`。
- `legacy_models/phase2/train_phase2.py`
  - `control`：K=1、无增强；
  - `real`：K=4、全真实路径；
  - `augment`：K=4、真实路径加保守增强；
  - 默认 Adam/AMSGrad、学习率 `1e-4`、micro-batch 2、累积 4、有效 batch 8；
  - 保存 `latest.pt`、`best_dev_robust.pt`、`best_primary.pt`、`best_rebound.pt`。
- `legacy_models/phase2/run_phase2_experiments.py`：P2-E0/E1/E2、条件 E3，以及显式启用的三种子 P2-E4。
- `legacy_models/phase2/phase2_lopo.py`：路径 1～8 的八折、成对 control/augment LOPO。

## 运行顺序

以下命令保留为可复现实验入口。当前正式 suite 已触发停止条件，因此不要继续运行其中的
`--run-final` 和 LOPO 命令，除非 Phase 3 后需要把它们作为新的对照重新启用。

所有命令都应通过完整 Conda 环境启动：

```powershell
# 单元测试
conda run --no-capture-output -n pytorch2.5.1_py310 python -m unittest discover -s tests -v

# P2-E0/E1/E2 和条件 E3；不会访问路径 9/10
conda run --no-capture-output -n pytorch2.5.1_py310 python -u -m legacy_models.phase2.run_phase2_experiments `
  --output-root runs/phase2_suite_seed2026 `
  --device cuda

# 确认开发集胜者后，训练 2026/2027/2028 三种子 P2-E4 并最终评估路径 9/10
conda run --no-capture-output -n pytorch2.5.1_py310 python -u -m legacy_models.phase2.run_phase2_experiments `
  --output-root runs/phase2_final_multiseed `
  --device cuda `
  --run-final

# 胜者配置确定后再执行 LOPO；此脚本不访问路径 9/10
conda run --no-capture-output -n pytorch2.5.1_py310 python -u -m legacy_models.phase2.phase2_lopo `
  --output-root runs/phase2_lopo_seed2026 `
  --device cuda
```

## 确定性 E3

- E2 开发集主指标下降超过 0.5 dB，或反弹增加超过 0.3 dB：`beta=0.10`、增强概率 `0.5`；
- 否则，若最差路径提升不足 0.5 dB：`beta=0.50`；
- E2 已满足上述条件则不运行 E3。

## 验收

相对 P1-E2，最终候选必须同时满足：全十路径主指标下降不超过 0.5 dB、反弹增加
不超过 0.3 dB、综合分下降不超过 0.2、最差路径至少 2.0 dB、路径 10 至少
2.5 dB、路径 9/10 平均主指标提高至少 0.75 dB、控制峰值不超过 1.0。

LOPO 另要求：八折主指标提升中位数至少 0.5 dB、至少 6/8 折不退化，且每折反弹
相对成对 control 增加不超过 0.3 dB。若 E2/E3 后最差路径仍低于 1.5 dB，或改善
需要牺牲超过 0.5 dB 的平均主指标，应停止扩展 Phase 2，转入反馈式 Phase 3。

正式 P2-E0/E1/E2/E3 已完成。开发 robust 最高的是 P2-E1（10.6433），但其最差
路径只有 0.9884 dB；路径鲁棒修正 E2/E3 的最好最差路径只有 0.9822 dB，低于
1.5 dB 停止阈值。因此 Phase 2 未通过，LOPO、三种子 P2-E4 和路径 9/10 最终验证
均未运行，正式模型继续使用 P1-E2。下一步应进入带 `e[t-1]` 的 Phase 3。

训练沿用 Phase 1 噪声划分：排序后的前六条噪声用于训练，末两条只用于固定
vehicle/restaurant/transition 验证。`control` 的 `beta` 强制解析为 0，保证它确实是
继续使用 Phase 1 单路径复合目标的学习率对照。
