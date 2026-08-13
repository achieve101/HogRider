# Phase 3G 阶梯长训与精确续训

本实验用于判断正式 seed 2027 的 P3G-E2 是否因 15 轮预算而欠训练。当前正式
checkpoint、v3 提交目录和 ZIP 都是受保护基线，长训脚本在运行前后逐一校验其
SHA-256，不会自动覆盖或导出正式模型。

冻结协议位于 `artifacts/phase3g_longtrain_protocol.json`，基线留档位于
`artifacts/phase3g_epoch15_baseline.json`。本实验不重跑 LOPO，路径 9/10 在候选按
路径 1～8 的开发集 D 锁定前保持封存。

## 1. 精确续训接口

普通新训练和原 `--checkpoint` 行为保持不变。新版训练产生的 `latest.pt` 可以这样
精确续训到目标总轮数：

```powershell
python train_phase3g.py `
  --stage generalize `
  --resume-checkpoint <上一阶段>/checkpoints/latest.pt `
  --output-dir <新的输出目录> `
  --epochs 40 `
  --samples-per-epoch 128 `
  --batch-size 8 `
  --gradient-accumulation 1 `
  --patience 10 `
  --seed 2027 `
  --device cuda `
  --save-every-epoch
```

`--epochs` 表示目标总 epoch。输出目录必须不存在。精确续训恢复模型、Adam/AMSGrad、
全部 RNG、DataLoader generator、best epoch、best D 和 patience 状态，不重新校准特征。

旧 epoch 15 checkpoint 只能显式使用 `--allow-legacy-resume`。它恢复模型和优化器并
从 epoch 16 开始，但由于旧文件没有完整 RNG 状态，结果标记为 `legacy-compatible`，
不具备正式候选资格。

## 2. Seed 2027 阶梯探索

```powershell
python run_phase3g_longtrain.py `
  --mode explore `
  --output-root runs/phase3g_longtrain_seed2027_explore `
  --device cuda
```

脚本从头运行 E1 5 轮和 E2 30 轮。只有最佳 D 位于最后三轮且没有因 patience 停止，
才精确续训到 40；40 轮仍撞边界才到 50。探索必须通过协议中的 D、反弹、最差路径、
压力和切换门槛。

## 3. 旧 epoch 15 对照与三种子复验

探索通过后，将探索汇总传给后续独立任务：

```powershell
python run_phase3g_longtrain.py `
  --mode legacy-control `
  --exploration-summary runs/phase3g_longtrain_seed2027_explore/exploration_summary.json `
  --output-root runs/phase3g_longtrain_epoch15_control `
  --device cuda

python run_phase3g_longtrain.py `
  --mode replicate `
  --exploration-summary runs/phase3g_longtrain_seed2027_explore/exploration_summary.json `
  --output-root runs/phase3g_longtrain_three_seeds `
  --device cuda
```

三种子都通过现有开发、安全、压力和切换门槛后，脚本按最高开发集 D 生成唯一
`candidate_lock.json`。该文件记录候选 checkpoint 哈希，仍未访问路径 9/10。

## 4. 单次路径 9/10 最终评估

```powershell
python phase3g_longtrain_final_evaluation.py `
  --candidate-lock runs/phase3g_longtrain_three_seeds/candidate_lock.json `
  --output runs/phase3g_longtrain_final_evaluation.json
```

入口先复验路径 1～8，再以排他方式创建 `final_evaluation_receipt.json`，随后才读取路径
9/10。receipt 存在后拒绝再次评估，因此不能在失败后改选另一个 seed。通过加强门槛
只会生成升级建议，不会修改正式 checkpoint、提交目录或 ZIP。

## 5. 2026-08-13 实际运行结果

### 5.1 阶梯训练与 checkpoint 位置

seed 2027 已从头完成 E1 5 轮，并按撞边界规则完成 E2 的 `30 → 40 → 50` 三个
阶梯。各阶段均为 `epoch_budget_exhausted`，没有触发 patience 10 提前停止：

| 阶段 | 最佳 epoch | 阶段结束 epoch | 结果目录 |
|---|---:|---:|---|
| ep30 | 30 | 30 | `runs/phase3g_longtrain_seed2027_explore/P3G-E2/ep30/` |
| ep40 | 40 | 40 | `runs/phase3g_longtrain_seed2027_explore/P3G-E2/ep40/` |
| ep50 | **49** | 50 | `runs/phase3g_longtrain_seed2027_explore/P3G-E2/ep50/` |

50 轮阶段需要区分“验证集最优模型”和“第 50 轮训练状态”：

- 推荐用于评估的 epoch 49 最优 checkpoint：
  `runs/phase3g_longtrain_seed2027_explore/P3G-E2/ep50/checkpoints/best_phase3g_selection.pt`；
  SHA-256 为 `a2382cb85b75016e7a4c5f475aca354f1d3a6ac089ce4c3b3e3db3e7b80d9cbe`。
- `best_epoch_0049.pt` 与上述最优 checkpoint 完全相同，SHA-256 相同。
- 每轮原始快照 `epoch_0049.pt` 的 SHA-256 为
  `ffc2a99833f4087404cc5f5fe4925c07dcdfb10d696728d7a8f6d720b8ae7184`。
- 第 50 轮快照为 `checkpoints/epoch_0050.pt`，SHA-256 为
  `ab121f36a94f985fc7fe51c4fd3cbf20fef0fb8a6725aab883c85c5bd8c33ca6`。
- 可继续精确续训的结束状态为 `checkpoints/latest.pt`，其 epoch 为 50，SHA-256 为
  `f790611eec759d60f44289e6f3afa5eb1f90c60b12442b5611474e59dc765a58`。

完整探索汇总为
`runs/phase3g_longtrain_seed2027_explore/exploration_summary.json`。epoch 49 开发集
`S=21.457921`、`R=0.652480`、`D=21.235519`、最差路径 `4.090217 dB`。D 和最差路径
通过探索门槛，但反弹超过 `0.544739 dB` 上限，因此探索总判定为失败。自动监督器按协议
没有运行旧 epoch 15 兼容续训对照，也没有启动三种子重跑。

### 5.2 官方公开闭环对比

经用户授权，使用官方 `DEEPANC_PARTICIPANT_KIT/run_public_demo.py`，在相同公开场景、
`start-seconds=0`、CPU 和默认安全上限下，对正式 epoch 15 与 epoch 49 进行了闭环比较：

| 指标 | 正式 epoch 15 | epoch 49 | 差值（49-15） |
|---|---:|---:|---:|
| 1/3 倍频带平均降噪 | 25.148412 dB | 27.871704 dB | **+2.723292 dB** |
| 平均窗口反弹峰值 | 0.137311 dB | 0.099368 dB | **-0.037943 dB** |
| 最差高频带降噪 | -0.776432 dB | -0.475748 dB | **+0.300683 dB** |
| 最大单窗口反弹 | 0.776432 dB | 0.475748 dB | **-0.300683 dB** |
| 宽带降噪 | 18.718463 dB | 19.687008 dB | **+0.968545 dB** |
| 控制输出峰值 | 0.063431 | 0.063621 | +0.000190 |

两者均通过 Participant Kit 接口、复杂度与 reset 检查，参数量均为 41,552，峰值事件
MAC 均为 2,484,720。完整报告位于：

- `runs/public_demo_epoch49_vs_formal/formal_epoch15.json`；
- `runs/public_demo_epoch49_vs_formal/epoch49.json`。

### 5.3 用户授权跳过三种子后的路径 9/10 评估

用户随后明确授权跳过耗时较长的三种子稳定性实验，直接锁定 seed 2027 epoch 49，执行
一次路径 9/10 测试。这是对原冻结协议的显式偏离，记录为
`three_seed_stability_skipped=true`；候选在访问最终路径前已锁定，失败后未改选其他 seed。

| 指标 | 正式 epoch 15 | epoch 49 | 差值（49-15） |
|---|---:|---:|---:|
| 路径 9 主降噪 | 15.293780 dB | 16.279259 dB | **+0.985479 dB** |
| 路径 9 反弹 | 7.359550 dB | 7.306302 dB | **-0.053247 dB** |
| 路径 10 主降噪 | 3.977502 dB | 4.026830 dB | **+0.049329 dB** |
| 路径 10 反弹 | 5.952844 dB | 6.005812 dB | +0.052968 dB |
| 路径 9/10 平均主指标 | 9.635641 dB | 10.153045 dB | **+0.517404 dB** |
| 十路径主指标 | 17.958467 dB | 19.196946 dB | **+1.238479 dB** |
| 十路径综合分 | 12.064817 | 12.881904 | **+0.817086** |
| 十路径反弹 | 1.687031 dB | 1.853195 dB | **+0.166165 dB** |
| 十路径最差路径 | 3.977502 dB | 4.026830 dB | **+0.049329 dB** |

epoch 49 通过原 Phase 3 最终验收，但未通过长训加强升级门槛：十路径反弹
`1.853195 dB` 高于上限 `1.787031 dB`，超出 `0.066165 dB`。因此
`formal_upgrade_recommended=false`。三种子稳定性未验证，正式模型继续保持 epoch 15，
没有覆盖 checkpoint、v3 提交包或 ZIP。

本次一次性最终评估产物位于：

- `runs/phase3g_longtrain_epoch49_direct_final/candidate_lock.json`；
- `runs/phase3g_longtrain_epoch49_direct_final/final_evaluation_receipt.json`；
- `runs/phase3g_longtrain_epoch49_direct_final/final_evaluation.json`。

评估结束后，基线清单中的 7 个正式文件 SHA-256 全部保持不变。
