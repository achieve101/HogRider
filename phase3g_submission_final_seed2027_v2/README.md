# Phase 3G 冻结权重生成式 FIR 控制器

本目录是赛题二正式推理包源码。模型采用反馈式逐采样接口，在时刻 `t` 接收
参考信号 `x[t]` 和上一采样点残差 `e[t-1]`。采样率固定为 48 kHz。

## 推理环境

- Python 3.10 或更高版本；
- PyTorch 2.1 或更高版本；
- NumPy 1.24 或更高版本；
- 当前提交实现仅支持 CPU 推理；
- 不需要网络访问，也不依赖 SciPy、librosa、torchaudio、torchvision 或 einops。

`requirements.txt` 只保留说明性注释，因为 PyTorch 和 NumPy 由赛事运行环境提供。

## 文件说明

- `submission.py`：赛事规定的 `create_model(device="cpu")` 固定入口；
- `runtime.py`：Participant Kit 适配层、权重加载、输出安全检查和复杂度声明；
- `model.py`：Phase 3G 生成式 FIR 控制器及严格因果流式状态；
- `weights.pt`：冻结的正式模型配置与 `state_dict`；
- `config.json`：模型结构和推理策略的可读说明；
- `requirements.txt`：推理依赖声明；
- `report/`：正式技术报告目录。

所有运行文件都从本目录按相对路径加载，不依赖训练数据、训练脚本、优化器状态或
本机绝对路径。推理期间模型参数保持冻结；GRU 隐状态、创新统计、FIR 激活和环形缓存
会随输入更新，并由 `reset()` 在每段录音开始前完整清空。

## 模型和复杂度

- 模型类型：反馈式；
- 总参数量：41,552；
- 稳态 MAC：2,048 / sample；
- 平均 MAC：约 12,392 / sample；
- 峰值事件 MAC：2,484,720；
- 输出安全范围：`(-0.98, 0.98)`。

## 本地公开检查

从包含本提交目录和 `DEEPANC_PARTICIPANT_KIT/` 的项目根目录运行：

```powershell
python DEEPANC_PARTICIPANT_KIT/validate_submission.py `
  --entry-point phase3g_submission_final_seed2027_v2.submission:create_model `
  --device cpu

python DEEPANC_PARTICIPANT_KIT/run_public_demo.py `
  --entry-point phase3g_submission_final_seed2027_v2.submission:create_model `
  --device cpu
```

公开场景只用于本地检查，不代表正式隐藏测试成绩。不要把公开测试数据或生成的 JSON
报告放入本提交目录。
