# Phase 3R：创新误差 FIR 专家路由

Phase 3R 是一个开发通过、但 LOPO 未通过的诊断候选。正式模型仍是 P1-E2。

## 实现

- `phase3r_templates.py`：从六条训练噪声和路径 1～8 生成固定 `P_i/S_i` 模板及 SHA-256 manifest；
- `phase3r_model.py`：逐采样公开接口、240 点 overlap-save 候选路径卷积和创新误差路由；
- `phase3r_validation.py`：固定开发集、路径切换、路由轨迹、官方声学指标和 CPU RTF；
- `legacy_models/phase3r/run_phase3r_experiments.py`：E1a/E1b/E1c 与条件 PN 回退的有界决策树；
- `legacy_models/phase3r/phase3r_lopo.py`：逐折物理移除留出专家、`P_i`、`S_i` 的八折评估；
- `legacy_models/phase3r/export_phase3r_submission.py`：导出仅用于接口诊断的自包含候选。

## 复现

```powershell
python phase3r_templates.py --dataset-dir dataset --output artifacts/phase3r_innovation_templates.npz
python -m legacy_models.phase3r.run_phase3r_experiments --dataset-dir dataset --template artifacts/phase3r_innovation_templates.npz --output-root runs/phase3r_suite_seed2026
python -m legacy_models.phase3r.phase3r_lopo --checkpoint runs/phase3r_suite_seed2026/P3R-E1c/candidate.pt --output-root runs/phase3r_lopo_seed2026
```

开发选中 P3R-E1c，但 LOPO 中位增益为 `-1.1187 dB`，仅 3/8 折不退化。因此不得运行
路径 9/10 最终评估，也不得把本候选描述为正式提交模型。
