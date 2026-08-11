# 从训练模型到 DEEPANC 提交 ZIP：逐步教程

本文面向已经训练好模型、但还没有制作比赛提交包的参赛者。请先完成每一步
再进入下一步。

## 第 1 步：整理训练产物

先确认自己拥有哪些文件：

- 网络结构代码，例如 `model.py`；
- 推理权重，例如 `weights.pt`；
- 构造网络所需的配置；
- 归一化参数；
- 推理时必须使用的其他固定数据；
- 第三方 Python 依赖。

不要把以下文件作为推理依赖：

- 训练数据集；
- 优化器状态；
- 学习率调度器；
- TensorBoard 日志；
- 训练过程中生成的临时缓存；
- 只在训练时使用的数据增强代码。

建议优先保存和加载 PyTorch `state_dict`，不要依赖训练电脑上的完整 Python
对象路径。

## 第 2 步：判断是前馈模型还是反馈模型

### 前馈模型

如果模型只需要参考信号 \(x[t]\)，选择 `feedforward`：

```python
requires_error = False

def process_sample(self, reference_sample):
    ...
```

### 反馈模型

如果模型还需要误差麦克风的历史残差，选择 `feedback`：

```python
requires_error = True

def process_sample(
    self,
    reference_sample,
    previous_error_sample,
):
    ...
```

反馈接口在时刻 \(t\) 提供的是 \(e[t-1]\)，不是 \(e[t]\)。模型不能假设
自己能够提前获得当前残差。

## 第 3 步：创建自己的提交目录

在 VS Code 中打开 `DEEPANC_PARTICIPANT_KIT`，新建终端。

前馈模型：

```bat
python create_submission.py --model-type feedforward --output my_submission
```

反馈模型：

```bat
python create_submission.py --model-type feedback --output my_submission
```

不要直接修改 `submission_templates/`。`my_submission/` 才是你的工作目录。

## 第 4 步：复制模型代码和权重

例如：

```text
my_submission/
├─ __init__.py
├─ submission.py
├─ model.py
├─ weights.pt
├─ config.json
├─ requirements.txt
├─ README.md
└─ report/
   └─ README.md
```

如果模型结构分散在多个文件中，可以保留自己的子目录，但所有导入必须能够
在提交包内解析。

推荐相对导入：

```python
from .model import MyANCModel
```

不推荐：

```python
from training_project.model import MyANCModel
```

后一种写法依赖没有打包的训练项目，换一台电脑后通常无法导入。

## 第 5 步：在 __init__() 中加载模型

打开 `my_submission/submission.py`，保留模板提供的类、公开属性和工厂
函数。

典型 PyTorch 加载代码：

```python
from pathlib import Path

import torch

from .model import MyANCModel


SUBMISSION_DIR = Path(__file__).resolve().parent


class MySubmission:
    sample_rate = 48_000
    requires_error = False

    def __init__(self, device="cpu"):
        self.device = torch.device(device)
        self.model = MyANCModel(...)

        state = torch.load(
            SUBMISSION_DIR / "weights.pt",
            map_location=self.device,
            weights_only=True,
        )
        self.model.load_state_dict(state)
        self.model.to(self.device)
        self.model.eval()
```

注意：

1. 使用传入的 `device`，不要在代码中强制写死 `cuda:0`；
2. 使用相对 `submission.py` 的权重路径；
3. 推理前调用 `eval()`；
4. 不在 `process_sample()` 中重复加载权重；
5. 不在导入模块时启动长时间任务。

## 第 6 步：把整段模型改成严格因果逐点模型

正式接口每次只给一个 48 kHz 采样点。模型必须立即返回当前输出。

### 情况 A：模型本身已经有 step() 接口

这是最理想的情况：

```python
def reset(self):
    self.hidden = None

def process_sample(self, reference_sample):
    x_t = torch.tensor(
        [[reference_sample]],
        dtype=torch.float32,
        device=self.device,
    )
    y_t, self.hidden = self.model.step(x_t, self.hidden)
    return float(y_t.item())
```

### 情况 B：模型是因果卷积，但训练时整段输入

假设模型的感受野是 \(R\)，可以保留一个长度为 \(R\) 的历史窗口。每个
时刻把新采样点放在窗口末尾，只取网络最后一个输出：

```python
def reset(self):
    self.history = torch.zeros(
        1,
        1,
        self.receptive_field,
        dtype=torch.float32,
        device=self.device,
    )

def process_sample(self, reference_sample):
    self.history[..., :-1] = self.history[..., 1:].clone()
    self.history[..., -1] = reference_sample
    y = self.model(self.history)
    return float(y[..., -1].item())
```

这个写法容易理解，但会重复计算整个窗口，速度可能较慢。更高效的做法是为
每层因果卷积保存缓存，只计算最新输出。无论采用哪种实现，结果必须与因果
整段推理一致。

### 情况 C：模型使用双向网络或未来帧

双向 RNN、非因果卷积和依赖未来帧的注意力不能直接用于严格实时接口。需要：

- 改为单向结构；
- 改成左侧填充的因果卷积；
- 去掉未来上下文；
- 重新训练或微调。

不能通过缓存一段未来测试音频绕过因果要求。

## 第 7 步：接入反馈误差信号

反馈模型可以把当前参考和上一采样点误差组合为输入：

```python
def process_sample(
    self,
    reference_sample,
    previous_error_sample,
):
    input_t = torch.tensor(
        [[reference_sample, previous_error_sample]],
        dtype=torch.float32,
        device=self.device,
    )
    y_t, self.hidden = self.model.step(input_t, self.hidden)
    return float(y_t.item())
```

时间关系为：

$$
\bigl(x[t],e[t-1],\text{state}[t-1]\bigr)
\longrightarrow
y[t]
$$

不要把 `previous_error_sample` 当作当前误差或未降噪噪声。

## 第 8 步：正确实现 reset()

`reset()` 不只是一个空函数。它决定不同测试录音之间是否相互污染。

检查以下状态：

- 输入移位寄存器；
- 误差移位寄存器；
- 每层卷积缓存；
- RNN 隐状态和 cell state；
- 注意力 KV 缓存；
- 模型内部累计特征；
- 模型内部选择或调度状态；
- 自适应滤波器系数；
- 运行均值；
- 上次输出。

一个简单原则：创建新模型后第一次运行，与对旧模型调用 `reset()` 后运行，
应得到一致结果。

如果模型推理含随机过程，应固定或移除随机性。通常比赛推理不应使用 dropout
或随机采样。

## 第 9 步：填写复杂度

模板需要四个字段：

```python
def get_complexity(self):
    return {
        "parameter_count": 123456,
        "steady_state_macs_per_sample": 78900,
        "startup_macs": 0,
        "peak_macs_in_one_sample_event": 78900,
}
```

这些字段只定义统一的复杂度统计口径，不预设或推荐任何模型结构。
`peak_macs_in_one_sample_event` 必须覆盖完整3.5秒运行过程，包括初始化阶段
发生的最重单点事件。初始化阶段不计声学成绩，不代表计算量可以不申报。

### 参数量

如果只有一个 PyTorch 网络：

```python
parameter_count = sum(
    parameter.numel()
    for parameter in self.model.parameters()
)
```

如果多个网络同时驻留，要对全部网络统计，并避免对共享参数重复计数。公开
检查器会递归扫描可找到的 PyTorch 参数并核对。

### MAC

线性层：

$$
\mathrm{MAC}_{\mathrm{Linear}}
=C_{\mathrm{in}}C_{\mathrm{out}}
$$

一维卷积：

$$
\mathrm{MAC}_{\mathrm{Conv1d}}
=
L_{\mathrm{out}}C_{\mathrm{out}}
\frac{C_{\mathrm{in}}}{G}K
$$

动态模型的峰值不是长期平均值，而是最重采样边界：

$$
\mathrm{MAC}_{\mathrm{peak}}
=
\max_{q\in\mathcal{Q}}
\sum_{\ell\in\mathcal{E}(q)}
\mathrm{MAC}_{\ell}
$$

如果某个采样点会同时运行多种模型、多个子模型或多个计算模块，就要把这些
模块在该采样点实际发生的 MAC 相加。

## 第 10 步：声明依赖

正式推理代码只允许依赖 Python 标准库、PyTorch 和 NumPy。训练阶段可以使用
其他工具，但最终提交必须移除对这些训练工具的导入和调用。

`my_submission/requirements.txt` 按默认规则应保持为空或只保留注释，不必
重复填写举办方已经提供的 `torch` 和 `numpy`，例如：

```text
# 本模型没有额外推理依赖。
```

正式环境默认不提供 `torchaudio`、`torchvision`、`scipy`、`librosa`、
`einops` 等其他第三方库。如确有无法移除的额外推理依赖，必须在提交截止前
取得组委会确认，不能仅在 `requirements.txt` 中自行添加。

提交前建议在一个干净虚拟环境中重新安装并运行公开检查。

## 第 11 步：运行公开检查

前馈和反馈使用同一个检查脚本：

```bat
python validate_submission.py --entry-point my_submission.submission:create_model --device cpu
```

检查样本数可以增加：

```bat
python validate_submission.py --entry-point my_submission.submission:create_model --device cpu --samples 48000
```

这会运行 1 秒公开合成输入。逐点 Python 模型可能需要较长时间。

公开检查失败时，从错误信息最内层开始排查：

| 错误 | 常见原因 |
|---|---|
| 无法导入模块 | 当前目录错误、缺少 `__init__.py` 或依赖 |
| 创建模型失败 | 权重路径错误、网络构造参数错误或设备错误 |
| 参数量不一致 | 漏计模块、重复计数或加载了额外子模型 |
| process_sample 失败 | 输入形状、dtype、设备或函数参数错误 |
| 输出不是有限数 | 模型不稳定、除零、溢出或状态没有初始化 |
| reset 不可复现 | 有状态未清零、模型仍在训练模式或使用随机操作 |
| 峰值 MAC 小于稳态 MAC | 复杂度字段填写矛盾 |

## 第 12 步：在公开车载场景测试真实效果

`validate_submission.py` 通过后，在同一个 VS Code 终端运行：

```bat
python run_public_demo.py --entry-point my_submission.submission:create_model --device cpu
```

这条命令各部分的含义是：

- `python`：使用当前 VS Code 终端所选的 Python 环境；
- `run_public_demo.py`：启动公开场景闭环评测器；
- `--entry-point`：告诉评测器从哪个模块、哪个函数创建模型；
- `my_submission.submission`：对应
  `my_submission\submission.py`；
- 冒号后的 `create_model`：对应文件内的模型工厂函数；
- `--device cpu`：让工厂函数收到 `device="cpu"`。

评测器先调用一次 `reset()`，然后连续运行0.5秒初始化和3秒正式计分，共
3.5秒。0.5秒边界处不会再次重置模型或次级路径。反馈模型首点误差为0，随后
始终接收真实的上一采样点误差。占位模板始终输出0，因此降噪量和反弹量均为
0 dB；接入有效模型后才会出现非零结果。

如需保存完整报告，运行：

```bat
python run_public_demo.py --entry-point my_submission.submission:create_model --device cpu --report dist\public_demo_result.json
```

新增参数的含义：

- `--report dist\public_demo_result.json`：把终端中的指标和运行信息另存为
  JSON，便于比较模型版本；它不是正式提交文件。

如果希望从公开片段的第 2 秒开始，可使用：

```bat
python run_public_demo.py --entry-point my_submission.submission:create_model --device cpu --start-seconds 2
```

`--start-seconds 2` 只是在原始公开音频中选择3.5秒评测片段的截取起点。
此前2秒不会送入模型；模型实际收到的前0.5秒是协议规定的初始化阶段。

公开片段总长15秒，因此起始时间加3.5秒不能超过15秒。

脚本会自动读取：

1. `public_demo_data\reference.wav`，即当前参考信号 \(x[t]\)；
2. `public_demo_data\disturbance.wav`，即未降噪期望噪声 \(d[t]\)；
3. `public_demo_data\secondary_path.npy`，即公开 1 号次级路径 \(s[k]\)。

前馈模型在时刻 \(t\) 只收到 \(x[t]\)。反馈模型不需要换命令，脚本会根据
`requires_error = True` 自动向它提供 \(x[t]\) 和已经算出的
\(e[t-1]\)。整个闭环严格按以下顺序运行：

$$
y[t]=\mathcal{M}\!\left(x[t]\right)
$$

或反馈模型：

$$
y[t]=\mathcal{M}\!\left(x[t],e[t-1]\right)
$$

然后计算：

$$
a[t]=\sum_{k=0}^{L_s-1}s[k]y[t-k],
\qquad
e[t]=d[t]-a[t]
$$

终端重点查看：

- `未加权 1/3 倍频带平均降噪量`：越高越好，是公开场景主指标；
- `平均窗口1/3倍频带反弹峰值`：每窗先取最坏频带，再对六窗平均，越低越好；
- `参数量` 和 `峰值事件 MAC`：必须与申报一致；
- `实时因子`：小于等于 1 表示本机运行时间不超过音频时间。

这是固定公开场景上的真实闭环结果，但不是正式测试成绩。正式测试会更换
未公开场景。不要把 `public_demo_data/` 或 JSON 结果放进
`my_submission/`。

评测器必须逐点等待模型输出，公开脚本固定处理3.5秒，不能自行缩短初始化或
计分时长。复杂网络可能明显慢于音频时长，应根据终端进度耐心等待。

## 第 13 步：检查提交目录

确认：

- `submission.py` 在提交目录根部；
- `create_model(device=...)` 可以调用；
- 所有模型代码都在目录内；
- 所有权重路径都是相对路径；
- `requirements.txt` 完整；
- `report/` 中已经放入一份正式 PDF 技术报告；
- 没有训练数据或本机绝对路径；
- CPU 检查已经通过；
- 如果声明支持 GPU，GPU 检查也已通过；
- 没有网络访问和外部进程；
- 输出使用正确的控制信号缩放和符号约定。

可以在 VS Code 中全局搜索以下内容：

```text
C:\
/home/
http://
https://
subprocess
requests
```

搜索结果不一定都是问题，但需要逐项确认。

## 第 14 步：生成 ZIP

```bat
python make_submission_zip.py --team-id TeamID --source my_submission --output dist\TeamID.zip
```

脚本不会覆盖同名 ZIP。如果要重新打包，请先把旧 ZIP 移到其他位置，或使用
新的文件名。

打包脚本会检查提交目录根部的 `submission.py`、`requirements.txt`、
`README.md` 和 `report/` 中唯一的 PDF；如果误把 `public_demo_data` 放进
提交目录，也会停止打包并提示删除。

用解压工具打开 ZIP，应看到：

```text
TeamID/
└─ Task2/
   ├─ submission.py
   ├─ requirements.txt
   ├─ README.md
   ├─ 模型代码和权重
   └─ report/
      └─ TeamID_Task2_Report.pdf
```

不要使用以下结构：

```text
Task2\submission.py
TeamID\submission.py
TeamID\Task2\my_submission\submission.py
```

`TeamID` 必须以英文字母开头，只能包含字母、数字和下划线。打包脚本会
自动添加 `TeamID/Task2/` 两层目录。最后保存脚本输出的 SHA-256，再上传
ZIP。完整格式见 `赛题二交付格式说明.md`。

## 最终检查清单

- [ ] 已选择正确的前馈或反馈模板；
- [ ] `sample_rate` 为 48000；
- [ ] `requires_error` 为正确的布尔值；
- [ ] 模型只使用当前和过去输入；
- [ ] 权重能够通过相对路径加载；
- [ ] 模型已经 `eval()`；
- [ ] `reset()` 清空全部状态；
- [ ] 每次调用只返回一个有限标量；
- [ ] 参数量包含全部驻留网络；
- [ ] 峰值 MAC 覆盖最坏事件；
- [ ] CPU 公开检查通过；
- [ ] 固定公开车载场景能够完成闭环评测；
- [ ] 模型能够从 `reset()` 后第一个采样点开始稳定运行；
- [ ] 依赖文件完整；
- [ ] `report/` 中有且仅有一份有效 PDF；
- [ ] ZIP 结构为 `TeamID/Task2/`；
- [ ] `TeamID/Task2/` 直接包含 `submission.py`；
- [ ] ZIP 不含训练数据和无关文件；
- [ ] 已记录最终 ZIP 的 SHA-256。
