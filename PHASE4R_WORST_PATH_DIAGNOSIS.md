# Phase 4R 最差路径因果诊断

> 本文档只记录冻结 checkpoint 的诊断结果。Oracle 使用 held-out 数据，因此不可部署、不可用于训练，正式 v3 模型与提交 ZIP 未改变。

## 结论

严格 LOPO Path 8 重放结果为 `17.2058280689 dB`，反弹为 `10.4783170692 dB`。
机械判据选择的根因是 **training_coverage_or_dictionary**，唯一冻结后续实验为 **E10-A / coverage_balanced_three_neighbor_synthesis**。本阶段未训练该实验。

## 声学 Oracle 层级（Path 8 独立评估集）

| 层级 | 主指标 (dB) | 反弹 (dB) |
|---|---:|---:|
| 最佳保留专家 | 12.986856 | 6.552050 |
| 静态 simplex | 16.552609 | 11.131566 |
| 静态 simplex + dictionary | 18.990100 | 10.539820 |
| 240 点块级 simplex + dictionary | 18.990101 | 10.539821 |
| 自由 2048-tap FIR | 10.312518 | 1.131607 |

自由 FIR 是三种固定初始化下的非凸最优已知解；未达门槛不能作为容量或记忆不足证明。容量判断以同架构已知 Path 8 三种子正对照为准。

## 对旧 Oracle 的修正

旧 `_oracle_latent_filter` 的 Path 8 结果为 `14.150668 dB`，但它所投影的旧 P3 Path 8 专家本身只有 `15.156740 dB`。因此该投影不能证明当前 FIR 或字典容量不足。

## 隔离与复现

- 8/8 折物理隔离：`True`。
- LOPO 最大主指标复现误差：`7.11e-15`。
- Path 9/10 未进入诊断训练输入：`True`。
- 完整逐场景、逐窗口、路由、覆盖相关性和哈希见 `artifacts/phase4r_worst_path_diagnosis.json`。
- 唯一预注册方案见 `artifacts/phase4r_preregistered_correction.json`。
