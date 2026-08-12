# 旧模型代码目录与文件用途

> 当前正式模型是经过 Phase 4R E09-A 运行时加固的 Phase 3G v3。本文说明已经退出
> 正式候选的历史代码、专项文档、辅助入口和旧提交包的放置位置及复现方式。

## 1. 整理原则

旧阶段代码分为两类：

1. **历史专属代码**：只用于旧模型训练、实验编排、LOPO 或导出，统一移动到 `legacy_models/`；
2. **当前共享代码**：虽然最早在旧阶段实现，但 Phase 3G 仍直接依赖，继续保留在项目根目录。

因此，根目录中的 `phase3_validation.py`、`phase3r_model.py` 等文件并不表示 Phase 3/3R 仍是正式模型，而是因为 Phase 3G 复用了其中的评分、manifest 或创新路由能力。

## 2. 目录结构

```text
legacy_models/
├── phase0_phase1/       # 原始 TCN 基线和官方 v6 对齐微调
├── phase2/              # 前馈 TCN 的多路径鲁棒训练
├── phase3/              # 反馈统计驱动的 FIR 专家实验
├── phase3r/             # 解析创新误差路由实验
├── phase3g/             # 已退出当前工作流的 Phase 3G 辅助入口
├── submissions/         # 历史 smoke、v1 和 v2 提交产物
└── evaluation_outputs/  # 旧自定义场景的录音、频谱图和评分说明截图
```

这些目录都包含 `__init__.py`，历史入口统一通过 `python -m legacy_models...` 运行。这样能够从项目根目录稳定解析共享模块，同时避免重新把旧脚本散落到根目录。

## 3. Phase 0 / Phase 1

目录：[legacy_models/phase0_phase1](legacy_models/phase0_phase1)

| 文件 | 用途 | 状态 |
|---|---|---|
| [model.py](legacy_models/phase0_phase1/model.py) | 42,764 参数因果 TCN `TimeDomainANC` 结构 | P1-E2 原正式模型结构 |
| [train.py](legacy_models/phase0_phase1/train.py) | Phase 0 强基线训练；同时提供设备、随机种子和数据扫描工具 | 历史复现及旧阶段共享工具 |
| [train_phase1.py](legacy_models/phase0_phase1/train_phase1.py) | Phase 1 官方六窗口、三分之一倍频程和反弹对齐训练 | P1-E0/E1/E2/E3 历史入口 |
| [run_phase1_experiments.py](legacy_models/phase0_phase1/run_phase1_experiments.py) | 顺序运行 Phase 1 有界实验并选择 checkpoint | 历史实验编排 |
| [evaluate_custom_scenes.py](legacy_models/phase0_phase1/evaluate_custom_scenes.py) | 早期拼接场景和旧版 EvaluationMetrics 可视化 | 仅旧评估参考，不是 v6 选模入口 |
| [PHASE0_STRONG_BASELINE.md](legacy_models/phase0_phase1/PHASE0_STRONG_BASELINE.md) | Phase 0 操作与复现说明 | 历史专项文档 |
| [PHASE1_V6.md](legacy_models/phase0_phase1/PHASE1_V6.md) | Phase 1 v6 对齐设计和结果 | 历史专项文档 |

复现示例：

```powershell
python -m legacy_models.phase0_phase1.train --help
python -m legacy_models.phase0_phase1.train_phase1 --help
python -m legacy_models.phase0_phase1.run_phase1_experiments --help
python -m legacy_models.phase0_phase1.evaluate_custom_scenes
```

## 4. Phase 2

目录：[legacy_models/phase2](legacy_models/phase2)

| 文件 | 用途 | 状态 |
|---|---|---|
| [train_phase2.py](legacy_models/phase2/train_phase2.py) | 路径分组采样、真实多路径和保守路径增强训练 | 阶段未通过，保留复现 |
| [phase2_validation.py](legacy_models/phase2/phase2_validation.py) | Phase 2 development/stress/final 隔离验证和门槛 | Phase 2 专属验证 |
| [run_phase2_experiments.py](legacy_models/phase2/run_phase2_experiments.py) | P2-E0/E1/E2、条件 E3 和可选最终实验编排 | 历史实验编排 |
| [phase2_lopo.py](legacy_models/phase2/phase2_lopo.py) | 路径 1～8 成对 control/augment LOPO | 历史泛化诊断 |
| [PHASE2_PATH_ROBUSTNESS.md](legacy_models/phase2/PHASE2_PATH_ROBUSTNESS.md) | Phase 2 路径鲁棒训练方案与结论 | 历史专项文档 |

复现示例：

```powershell
python -m legacy_models.phase2.train_phase2 --help
python -m legacy_models.phase2.run_phase2_experiments --help
python -m legacy_models.phase2.phase2_lopo --help
```

## 5. Phase 3

目录：[legacy_models/phase3](legacy_models/phase3)

| 文件 | 用途 | 状态 |
|---|---|---|
| [train_phase3.py](legacy_models/phase3/train_phase3.py) | 公共 FIR 蒸馏、oracle 专家、GRU 路由和联合微调 | 学习式路由失败，保留复现 |
| [run_phase3_experiments.py](legacy_models/phase3/run_phase3_experiments.py) | P3-E0～E3 阶段决策树 | 历史实验编排 |
| [phase3_lopo.py](legacy_models/phase3/phase3_lopo.py) | Phase 3 八折路径留出评估 | 历史泛化诊断 |
| [phase3_final_evaluation.py](legacy_models/phase3/phase3_final_evaluation.py) | Phase 3 三种子和封存路径最终评估 | 未成为正式升级 |
| [summarize_phase3_suite.py](legacy_models/phase3/summarize_phase3_suite.py) | 汇总被中断或部分完成的 Phase 3 suite | 辅助工具 |
| [export_phase3_submission.py](legacy_models/phase3/export_phase3_submission.py) | 导出 Phase 3 反馈 FIR 候选 | 诊断用，不是当前提交包 |
| [phase3_submission_runtime.py](legacy_models/phase3/phase3_submission_runtime.py) | Phase 3 Participant Kit 运行时包装 | 由旧导出器复制 |
| [PHASE3_FEEDBACK_FIR.md](legacy_models/phase3/PHASE3_FEEDBACK_FIR.md) | Phase 3 反馈 FIR 专家设计与失败结论 | 历史专项文档 |

复现示例：

```powershell
python -m legacy_models.phase3.train_phase3 --help
python -m legacy_models.phase3.run_phase3_experiments --help
python -m legacy_models.phase3.phase3_lopo --help
python -m legacy_models.phase3.export_phase3_submission --help
```

## 6. Phase 3R

目录：[legacy_models/phase3r](legacy_models/phase3r)

| 文件 | 用途 | 状态 |
|---|---|---|
| [run_phase3r_experiments.py](legacy_models/phase3r/run_phase3r_experiments.py) | E1a/E1b/E1c 和条件 PN 回退实验 | 路由通过、LOPO 未通过 |
| [phase3r_lopo.py](legacy_models/phase3r/phase3r_lopo.py) | 逐折移除专家和创新模板的严格 LOPO | 历史泛化诊断 |
| [export_phase3r_submission.py](legacy_models/phase3r/export_phase3r_submission.py) | 导出 Phase 3R 诊断候选 | 非正式提交包 |
| [phase3r_submission_runtime.py](legacy_models/phase3r/phase3r_submission_runtime.py) | Phase 3R Participant Kit 运行时包装 | 由旧导出器复制 |
| [PHASE3R_INNOVATION_ROUTING.md](legacy_models/phase3r/PHASE3R_INNOVATION_ROUTING.md) | Phase 3R 创新路由设计与 LOPO 结论 | 历史专项文档 |

复现示例：

```powershell
python phase3r_templates.py --help
python -m legacy_models.phase3r.run_phase3r_experiments --help
python -m legacy_models.phase3r.phase3r_lopo --help
python -m legacy_models.phase3r.export_phase3r_submission --help
```

## 7. 已归档的 Phase 3G 辅助入口与提交包

### 7.1 Phase 3G 辅助入口

| 文件 | 用途 | 状态 |
|---|---|---|
| [evaluate_phase3g_checkpoint.py](legacy_models/phase3g/evaluate_phase3g_checkpoint.py) | 从已有 checkpoint 恢复开发/压力验证 | 正式训练和最终评估已完成，仅保留复现 |

### 7.2 历史提交产物

目录：[legacy_models/submissions](legacy_models/submissions)

| 目录 | 用途 | 状态 |
|---|---|---|
| `phase3_submission_smoke/` | Phase 3 反馈 FIR 的早期 smoke 包 | 历史产物 |
| `phase3g_submission_smoke_v2/` | Phase 3G 导出链路 smoke 包 | 历史产物 |
| `phase3g_submission_final_seed2027/` | Phase 3G 正式导出 v1 | 已被 v2/v3 替代 |
| `phase3g_submission_final_seed2027_v2/` | Phase 3G 冻结对照 v2 | 已被 v3 替代；仍用于 E09-A 逐点等价回归 |

这些目录不再出现在项目根目录。v2 仍可通过
`legacy_models.submissions.phase3g_submission_final_seed2027_v2.submission:create_model`
加载，但不能误作当前交付包。

### 7.3 早期自定义场景输出

目录：[legacy_models/evaluation_outputs](legacy_models/evaluation_outputs)

这里保存 `evaluate_custom_scenes.py` 生成的两段 `Scene_*_Test` 录音、倍频程图、频谱图，
以及早期评分标准截图。这些文件只用于历史可视化，不参与 Participant Kit v6 官方评分、
当前训练、LOPO、Phase 4R 诊断或正式 ZIP。

## 8. 保留在根目录的共享模块

### 8.1 基础数据与官方评分

| 文件 | 当前用途 |
|---|---|
| [dataset.py](dataset.py) | 数据集读取与因果次级路径卷积；仍被固定验证复用 |
| [phase1_data.py](phase1_data.py) | 168,000 点样本和固定验证 manifest；Phase 3G 验证与 LOPO 继续使用 |
| [phase1_validation.py](phase1_validation.py) | 官方 scorer 适配、记录聚合和固定场景验证 |
| [v6_metrics.py](v6_metrics.py) | 官方六窗口、三分之一倍频程主指标和反弹的可微实现 |

### 8.2 路径和闭环共享能力

| 文件 | 当前用途 |
|---|---|
| [phase2_paths.py](phase2_paths.py) | `augment_secondary_path()` 被 Phase 3G 连续路径合成复用 |
| [phase3_model.py](phase3_model.py) | FIR 专家 checkpoint 的结构定义与初始化来源 |
| [phase3_closed_loop.py](phase3_closed_loop.py) | 旧 Phase 3 闭环实现和相关回归测试 |
| [phase3_data.py](phase3_data.py) | 旧专家训练数据与回归测试 |
| [phase3_validation.py](phase3_validation.py) | Phase 3G 仍使用其 manifest、计分记录和选择分数工具 |
| [phase3r_model.py](phase3r_model.py) | Phase 3G 初始化和基线对照所需的解析创新路由器 |
| [phase3r_templates.py](phase3r_templates.py) | 当前模型使用的固定 `P_i/S_i` 创新模板生成与校验 |
| [phase3r_validation.py](phase3r_validation.py) | Phase 3G 复用的逐采样闭环、开发集和路径切换验证 |

这些文件不能在不重构 Phase 3G 依赖的情况下移入“纯归档”目录。后续若希望进一步收紧根目录，应把它们迁入稳定的 `anc_core` 包，而不是归入旧模型。

## 9. 当前正式模型文件

Phase 3G 正式链路继续保留在根目录，便于识别当前有效入口：

- [phase3g_model.py](phase3g_model.py)：当前正式控制器；
- [phase3g_data.py](phase3g_data.py)：连续路径合成和训练 dataset；
- [phase3g_closed_loop.py](phase3g_closed_loop.py)：可微闭环与损失；
- [train_phase3g.py](train_phase3g.py)：当前训练入口；
- [phase3g_validation.py](phase3g_validation.py)：开发、压力和状态不可变验证；
- [phase3g_lopo.py](phase3g_lopo.py)：严格 LOPO；
- [phase3g_final_evaluation.py](phase3g_final_evaluation.py)：三种子和十路径最终评估；
- [export_phase3g_submission.py](export_phase3g_submission.py)：正式提交包导出；
- [phase3g_submission_runtime.py](phase3g_submission_runtime.py)：正式运行时包装。
- [phase3g_submission_final_seed2027_v3](phase3g_submission_final_seed2027_v3)：当前唯一保留在根目录的正式提交源目录。
- [benchmark_phase4r_runtime.py](benchmark_phase4r_runtime.py)：v2/v3 运行时等价与计时工具；
- [phase4r_worst_path_diagnosis.py](phase4r_worst_path_diagnosis.py)：当前 Phase 4R 诊断入口；
- [prepare_phase4r_e10a.py](prepare_phase4r_e10a.py)：E10-A 预注册协议闭合与邻居表生成。

## 10. 注意事项

- 历史 checkpoint 和 `runs/` 目录没有移动，原实验记录中的 checkpoint 路径保持不变；
- 已生成的旧提交 smoke/v1/v2 包只移动路径、未改写内容，保证历史产物可审计；
- 历史 Python 入口的调用方式从脚本路径改为模块形式，例如 `python -m legacy_models.phase2.train_phase2`；
- 当前正式提交目录是 `phase3g_submission_final_seed2027_v3/`，不在 `legacy_models/` 中；
- 根目录中的 Phase 1/2/3/3R 共享 Python 模块仍被正式 Phase 3G、Phase 4R 或回归测试直接导入，不能在不重构包结构的情况下继续移动。
