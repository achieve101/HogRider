# Phase 3G：冻结权重的创新条件生成式 FIR

Phase 3G 解决 P3R 已识别路径、但离散 FIR 专家无法对未建模路径插值和外推的问题。
开发、连续路径压力集、严格 LOPO、三种子和最终十路径门槛已全部通过，Phase 3G 已正式升级。

## 1. 控制器

`GenerativeInnovationFIRController` 保留 P3R-E1c 的 4096 点创新窗、240 点更新周期和
解析路由，在其基础上增加固定权重 GRU 残差生成器：

```text
52 features -> GRUCell(hidden=32) -> 16 latent coefficients
W = alpha @ frozen_experts + z @ learned_dictionary
```

默认可训练参数为 41,552。推理时参数、字典、专家和路径模板均不修改；变化的有效 FIR
属于神经网络输出激活，并在 `reset()` 中清空。导出包不包含优化器、训练入口或随机采样。

## 2. 数据与训练

训练只使用六条训练噪声和路径 1～8。连续路径分布包含 50% 测量路径、30% DTW 对齐
插值、10% 有界外推和 10% 原增强，25% 样本在绝对时间 2.0 秒切换路径。候选遮蔽比例
为 40% 不遮蔽、30% 遮蔽一个端点、30% 遮蔽两个端点。

```powershell
# P3G-E1：5 轮测量路径暖启动
python train_phase3g.py --stage warmup --output-dir runs/phase3g_e1_seed2026

# P3G-E2：从 E1 最佳 checkpoint 进行连续路径训练
python train_phase3g.py --stage generalize `
  --checkpoint runs/phase3g_e1_seed2026/checkpoints/best_phase3g_selection.pt `
  --output-dir runs/phase3g_e2_seed2026

# 有界实验决策树；默认不会自动运行昂贵 LOPO
python run_phase3g_experiments.py --output-root runs/phase3g_suite_seed2026

# 开发通过后显式运行严格八折 LOPO
python phase3g_lopo.py --output-root runs/phase3g_lopo_seed2026
```

LOPO 每折会从训练 disturbance、次级路、专家、创新模板、字典初始化和合成端点中物理
删除留出路径。LOPO 不通过时会输出一次确定性 E3 容量修正建议。

## 3. 正式验证结果

- seed 2026 E2 开发：`S=19.5192`、`R=0.3395`、`D=19.5865`；
- 48 条连续路径压力集：中位主指标增益 `+2.7751 dB`，91.67% 场景不退化；
- 严格八折 LOPO：中位增益 `+0.9112 dB`，6/8 折不退化，8/8 物理隔离通过；
- LOPO 已通过，因此没有运行可选 E3；
- seeds 2026/2027/2028 均通过开发、压力、切换、输出、finite、状态不可变和 RTF 门槛；
- 三种子通过后才首次读取路径 9/10，三个种子均通过十路径最终门槛；
- 按开发 D 选中 seed 2027：十路径 `S=17.9585`、`R=1.6870`、`C=12.0648`；
- 路径 9 为 `15.2938 dB`，路径 10/十路径最差为 `3.9775 dB`；
- 38/38 全量回归测试通过；
- Participant Kit 提交检查：41,552 参数、输出峰值 0.0447、CPU RTF 0.695；
- 完整公开闭环：`S=25.1484`、`R=0.1373`、CPU RTF 0.959。

正式 checkpoint：

```text
runs/phase3g_suite_seed2027/P3G-E2/checkpoints/best_phase3g_selection.pt
```

正式提交包：`phase3g_submission_final_seed2027_v2`；已剔除训练期 artifact/字典初始化辅助方法。
