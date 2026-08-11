# 赛题二交付准备状态

更新时间：2026-08-10

正式推理源码目录：`phase3g_submission_final_seed2027_v2/`

## 已完成

- Phase 3G seed 2027 正式权重已固化为 `weights.pt`；
- `submission.py` 位于提交源码根目录并提供 `create_model()`；
- 反馈式 `process_sample(x[t], e[t-1])`、`reset()` 和复杂度声明已实现；
- 推理仅依赖 Python 标准库、PyTorch 和 NumPy；
- 权重和配置使用相对路径，不包含训练数据、日志或本机绝对路径；
- 提交 README 已补全环境、设备、文件用途、模型类型和公开检查命令；
- `weights.pt` 已确认可以通过 `weights_only=True` 加载；
- Participant Kit CPU 接口检查已通过：反馈式、48 kHz、41,552 参数、
  峰值事件 MAC 2,484,720、输出峰值 0.0446637；
- Participant Kit 完整公开闭环已通过：主指标 25.1484 dB、平均反弹 0.1373 dB、
  本次 CPU RTF 0.952；
- 正式权重 SHA-256：
  `7544A871D07DCCC89BB76197D81F9218D1AADA01ABF32183C9D78368CC399C0A`；
- 静态审计未发现本机绝对路径、训练数据、网络调用或禁止的第三方推理依赖。

## 正式 ZIP

- Team ID：`CCFANC`；
- 技术报告：`phase3g_submission_final_seed2027_v2/report/CCFANC_Task2_Report.pdf`；
- 最终文件：`dist/CCFANC.zip`；
- ZIP SHA-256：
  `2950FEAEA6A8A8009252E96A021FCD1C23F0A7932FBF8AD4699639CE9A7BCC97`。

官方 `make_submission_zip.py` 已完成打包。ZIP 共 9 个文件，内部结构固定为
`CCFANC/Task2/`，不含缓存、训练数据、公开测试数据或本地验证 JSON。

最近一次公开闭环完整 JSON 位于项目的
`runs/ccfanc_packaged_public_demo.json`。该报告来自实际 ZIP 解压内容，主指标
25.1484 dB、平均反弹 0.1373 dB、CPU RTF 0.989；它是本地验证产物，不在最终 ZIP 中。

实际 ZIP 解压内容已再次通过官方 `validate_submission.py`：反馈式、48 kHz、
41,552 参数、峰值事件 MAC 2,484,720、reset 可复现。

## 最终打包命令

在项目根目录运行：

```powershell
python DEEPANC_PARTICIPANT_KIT/make_submission_zip.py `
  --team-id <TeamID> `
  --source phase3g_submission_final_seed2027_v2 `
  --output dist/<TeamID>.zip
```

打包完成后应确认 ZIP 内直接存在：

```text
<TeamID>/Task2/submission.py
```
