# 赛题二交付准备状态

更新时间：2026-08-11

正式推理源码目录：`phase3g_submission_final_seed2027_v3/`

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
  Phase 4R 十次 CPU RTF P50/P95/最大值为 0.6760/0.7000/0.7082；
- v3 与冻结 v2 的 168,000 点控制输出和闭环残差逐点完全一致，最大绝对误差为 0；
- 240 点块边界耗时 P99 为 1.6023 ms，最大值为 2.2348 ms；
- 正式权重 SHA-256：
  `7544A871D07DCCC89BB76197D81F9218D1AADA01ABF32183C9D78368CC399C0A`；
- 静态审计未发现本机绝对路径、训练数据、网络调用或禁止的第三方推理依赖。

## 正式 ZIP

- Team ID：`CCFANC`；
- 技术报告：`phase3g_submission_final_seed2027_v3/report/CCFANC_Task2_Report.pdf`；
- 最终文件：`dist/CCFANC.zip`；
- ZIP SHA-256：
  `2BC1AC57D9D98A089BCAEC2CFC731DF8D62DAB735344D8AEC54B9F790286EE84`。

官方 `make_submission_zip.py` 已完成打包。ZIP 共 9 个文件，内部结构固定为
`CCFANC/Task2/`，不含缓存、训练数据、公开测试数据或本地验证 JSON。

最近一次公开闭环完整 JSON 位于项目的
`runs/ccfanc_phase4r_packaged_public_demo.json`。该报告来自实际 ZIP 解压内容，主指标
25.1484 dB、平均反弹 0.1373 dB、CPU RTF 0.686；它是本地验证产物，不在最终 ZIP 中。

实际 ZIP 解压内容已再次通过官方 `validate_submission.py`：反馈式、48 kHz、
41,552 参数、峰值事件 MAC 2,484,720、reset 可复现；默认 4,096 点和长稳态
168,000 点 RTF 分别为 0.405 和 0.537。

## 最终打包命令

在项目根目录运行：

```powershell
python DEEPANC_PARTICIPANT_KIT/make_submission_zip.py `
  --team-id <TeamID> `
  --source phase3g_submission_final_seed2027_v3 `
  --output dist/<TeamID>.zip
```

打包完成后应确认 ZIP 内直接存在：

```text
<TeamID>/Task2/submission.py
```
