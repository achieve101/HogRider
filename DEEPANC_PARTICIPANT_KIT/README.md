# DEEPANC 参赛模型提交工具包

版本：1.5.0

声学评分协议：v6

本工具包用于帮助参赛者完成以下工作：

1. 根据前馈或反馈模型模板准备提交文件；
2. 在本地检查模型入口、逐采样调用、状态重置、输出和复杂度声明；
3. 在固定公开车载场景上计算真实闭环降噪指标；
4. 把已经检查通过的目录打包成规范 ZIP。

`validate_submission.py` 使用确定性合成输入，只检查提交形式。
`run_public_demo.py` 使用一段公开的真实车载数据和公开 1 号次级路径，能够
计算本地声学指标。公开场景不是正式测试集，本地结果不等于最终成绩；正式
评测仍会使用不公开的测试场景和次级路径。

## 推理环境与允许依赖

正式推理环境以本项目 baseline 能够正常运行的版本范围为基准，不要求参赛者
使用与举办方完全相同的补丁版本。提交模型必须能够在以下环境中通过公开检查：

| 项目 | 支持范围 | 用途 |
|---|---|---|
| Python | 3.10 或更高版本 | 运行模型包装和逐采样接口 |
| PyTorch | 2.1 或更高版本 | 模型构建、权重加载和推理 |
| NumPy | 1.24 或更高版本 | 数组、配置和流式状态处理 |
| Python 标准库 | 随 Python 提供 | 路径、JSON、容器和基础工具 |

参赛模型的推理代码只允许依赖 Python 标准库、PyTorch 和 NumPy。正式推理
环境默认不提供 `torchaudio`、`torchvision`、`scipy`、`librosa`、
`einops` 等其他第三方库。训练阶段可以使用任意工具，但依赖这些工具产生的
滤波器系数、归一化参数和固定配置必须在提交前固化为模型权重、NumPy 数组
或 JSON 等提交文件，推理时不能再次调用训练工具生成。

提交目录中的 `requirements.txt` 用于说明推理依赖，按默认规则应保持为空或
只保留注释，不必重复填写举办方已经提供的 `torch` 和 `numpy`。如确有无法
移除的额外推理依赖，必须在提交截止前取得组委会确认；未经确认的第三方依赖
不属于正式评测环境。

工具包自身还使用 `SoundFile` 读取公开演示 WAV。它由工具包根目录的
`requirements.txt` 安装，只属于举办方评测工具依赖，不是参赛模型可以依赖
的推理库。最终兼容性以模型能否在上述环境中通过 `validate_submission.py`
和 `run_public_demo.py` 为准。

## 1. 文件说明

```text
DEEPANC_PARTICIPANT_KIT/
├─ .gitignore                        忽略缓存、临时 ZIP 和 dist 目录
├─ .vscode/
│  └─ settings.json                  Windows 下默认使用 Command Prompt
├─ README.md                         本文件：工具包总说明和快速使用流程
├─ MODEL_PREPARATION_GUIDE.md        从训练模型到提交 ZIP 的详细教程
├─ 赛题二交付格式说明.md             正式交付内容和 TeamID/Task2 结构
├─ VERSION.txt                       工具包和接口版本
├─ requirements.txt                  运行公开工具所需的基础依赖
├─ check_environment.py              检查 Python、NumPy、PyTorch 和设备
├─ create_submission.py              复制前馈或反馈模板，创建个人提交目录
├─ validate_submission.py            公开接口、逐点状态和复杂度检查
├─ make_submission_zip.py            生成规范提交 ZIP 并输出 SHA-256
├─ run_public_demo.py                在公开车载场景上逐点闭环评测
├─ participant_api.py                评测器调用参赛模型的公开接口定义
├─ complexity_check.py               参数量和 MAC 声明检查
├─ public_demo_scoring.py            公开演示场景的声学指标计算
├─ public_demo_data/
│  ├─ README.md                      三种公开数据的含义和闭环公式
│  ├─ manifest.json                  数据版本、长度和文件哈希
│  ├─ reference.wav                  15 秒车载参考信号 x[t]
│  ├─ disturbance.wav                对应耳侧期望噪声 d[t]
│  └─ secondary_path.npy             公开 1 号次级路径 s[k]
├─ submission_templates/
│  ├─ __init__.py                    使模板目录可以作为 Python 包导入
│  ├─ feedforward_model/             不使用误差信号的前馈模板
│  │  ├─ __init__.py
│  │  ├─ submission.py
│  │  ├─ requirements.txt
│  │  ├─ README.md
│  │  └─ report/
│  │     └─ README.md                PDF 技术报告放置说明
│  └─ feedback_model/                使用上一采样点误差的反馈模板
│     ├─ __init__.py
│     ├─ submission.py
│     ├─ requirements.txt
│     ├─ README.md
│     └─ report/
│        └─ README.md                PDF 技术报告放置说明
```

各脚本的职责互不重叠：

| 文件 | 什么时候使用 | 是否需要修改 |
|---|---|---|
| `README.md` | 第一次打开工具包时阅读 | 否 |
| `MODEL_PREPARATION_GUIDE.md` | 接入自己的训练模型时逐步阅读 | 否 |
| `赛题二交付格式说明.md` | 最终打包前核对正式目录和交付物 | 否 |
| `VERSION.txt` | 确认工具包和公开协议版本 | 否 |
| `requirements.txt` | 安装公开检查工具的基础依赖 | 一般不修改 |
| `check_environment.py` | 安装环境后运行 | 否 |
| `create_submission.py` | 创建自己的提交目录 | 否 |
| `validate_submission.py` | 每次修改模型后运行 | 否 |
| `run_public_demo.py` | 接口通过后查看公开场景真实降噪量 | 否 |
| `make_submission_zip.py` | 最终检查通过后运行 | 否 |
| `participant_api.py` | 了解评测器如何加载模型 | 否 |
| `complexity_check.py` | 了解公开复杂度检查方法 | 否 |
| `public_demo_scoring.py` | 了解本地声学指标的实现 | 否 |
| `public_demo_data/` | 固定公开车载场景；只用于本地验证 | 否 |
| `.vscode/settings.json` | 在 Windows VS Code 中默认选择 CMD | 否 |
| `.gitignore` | 防止缓存和临时打包文件进入版本控制 | 否 |
| 模板中的 `__init__.py` | 使模板和提交目录可作为 Python 包导入 | 否 |
| 模板中的 `submission.py` | 加载权重并实现逐点推理 | 复制后修改 |
| 模板中的 `requirements.txt` | 说明推理依赖范围 | 一般保持为空或只保留注释 |
| 模板中的 `README.md` | 说明该模板的模型类型和固定入口 | 复制后可补充 |

## 2. 在 VS Code 中快速完成一次流程

以下命令均在 VS Code 的集成终端中运行。Windows 用户打开本文件夹后，
点击“终端 → 新建终端”，默认会使用 Command Prompt。

### 第一步：安装基础依赖

```bat
python -m pip install -r requirements.txt
```

如果已有比赛指定环境，可以直接使用该环境，不必重复安装 PyTorch。

### 第二步：检查环境

```bat
python check_environment.py --device cpu
```

使用 GPU 时：

```bat
python check_environment.py --device cuda:0
```

看到“环境检查通过”后再继续。

### 第三步：选择模型类型并创建提交目录

不需要误差信号的前馈模型：

```bat
python create_submission.py --model-type feedforward --output my_submission
```

需要上一采样点误差信号的反馈模型：

```bat
python create_submission.py --model-type feedback --output my_submission
```

脚本不会覆盖已存在的目录。创建后会得到：

```text
my_submission/
├─ __init__.py
├─ submission.py
├─ requirements.txt
├─ README.md
└─ report/
   └─ README.md
```

### 第四步：放入并接入自己的模型

把网络代码、配置和权重复制到 `my_submission/`。然后打开
`my_submission/submission.py`，按文件中的中文注释完成：

1. 在 `__init__()` 中创建模型、加载权重、移动到指定设备并调用
   `eval()`；
2. 在 `reset()` 中清空输入历史、卷积缓存、RNN 状态和其他流式状态；
3. 在 `process_sample()` 中实现严格因果的逐采样推理；
4. 在 `get_complexity()` 中填写参数量和 MAC；
5. 确认 `my_submission/requirements.txt` 没有未经确认的额外依赖；
6. 在 `my_submission/report/` 中放入一份 PDF 技术报告。

详细接入方法见 [MODEL_PREPARATION_GUIDE.md](MODEL_PREPARATION_GUIDE.md)。

### 第五步：运行公开提交检查

```bat
python validate_submission.py --entry-point my_submission.submission:create_model --device cpu
```

通过时会显示：

```text
公开提交检查通过
模型类型：前馈式或反馈式
采样率：48000 Hz
模型总参数量：...
峰值事件 MAC：...
```

这一步应在 CPU 环境至少运行一次。如果提交支持 GPU，也应再运行：

```bat
python validate_submission.py --entry-point my_submission.submission:create_model --device cuda:0
```

### 第六步：运行公开场景真实效果测试

运行与正式评测单个场景时序一致的公开测试：

```bat
python run_public_demo.py --entry-point my_submission.submission:create_model --device cpu
```

脚本会逐点向模型提供 48 kHz 参考信号。反馈模型还会自动收到上一采样点
残差。控制输出经过公开 1 号次级路径后，脚本会显示未加权 1/3 倍频带平均
降噪量、1/3 倍频带反弹峰值、复杂度和运行速度。

每个场景开始时，评测器先调用一次 `reset()`，反馈模型首点收到
`e[-1] = 0`。模型随后连续处理0.5秒初始化信号和3秒正式计分信号，共3.5秒。
初始化与计分之间不会再次调用 `reset()`，模型状态和次级路径状态连续。

初始化阶段不计入声学成绩，但其模型输出仍接受有限值和幅值检查，运行时间与
复杂度也覆盖完整3.5秒。3秒计分段被划分为六个连续、互不重叠的0.5秒窗口；
各窗口分别计算指标后等权平均。

保存完整 JSON 报告：

```bat
python run_public_demo.py --entry-point my_submission.submission:create_model --device cpu --report dist\public_demo_result.json
```

公开音频总长为 15 秒，必须满足：

```text
start-seconds + 3.5 <= 15
```

该结果只反映固定公开场景，不是正式成绩。不要把 `public_demo_data/` 或
生成的 JSON 报告复制进 `my_submission/`。

### 第七步：生成提交 ZIP

```bat
python make_submission_zip.py --team-id TeamID --source my_submission --output dist\TeamID.zip
```

脚本会输出 ZIP 文件的 SHA-256。请保存该值，用于确认上传文件没有变化。

ZIP 内应包含：

```text
TeamID/
└─ Task2/
   ├─ submission.py
   ├─ requirements.txt
   ├─ README.md
   ├─ 模型代码、配置和权重
   └─ report/
      └─ TeamID_Task2_Report.pdf
```

不要把整个工具包、训练数据、训练日志、优化器状态或与推理无关的文件打入
ZIP。

## 3. 参赛模型公开接口

每个提交必须提供：

```python
def create_model(device: str = "cpu"):
    return YourSubmission(device=device)
```

返回对象必须具有以下属性和方法：

```python
sample_rate = 48_000
requires_error = False  # 或 True

reset()
process_sample(...)
get_complexity()
```

### 3.1 前馈模型

```python
sample_rate = 48_000
requires_error = False

def process_sample(self, reference_sample: float) -> float:
    ...
```

时刻 \(t\) 只接收当前参考信号 \(x[t]\)。模型可以使用自己保存的当前及
过去状态，但不能使用未来输入。

### 3.2 反馈模型

```python
sample_rate = 48_000
requires_error = True

def process_sample(
    self,
    reference_sample: float,
    previous_error_sample: float,
) -> float:
    ...
```

时刻 \(t\) 接收 \(x[t]\) 和 \(e[t-1]\)。第一次调用时上一采样点误差按
初始值处理。接口不会把当前 \(e[t]\) 传给模型。

### 3.3 reset()

每段新录音开始前评测器会调用一次 `reset()`。它必须清空：

- 输入和误差历史；
- 流式卷积缓存；
- RNN、GRU 或 LSTM 隐状态；
- 模型内部的选择、调度和累计状态；
- 自适应算法内部状态；
- 上一段录音产生的其他状态。

相同模型在两次 `reset()` 后接收相同输入，应在允许的数值误差内得到相同
输出。

### 3.4 输出要求

每次 `process_sample()` 必须返回一个标量，并且：

- 可以转换为 Python `float`；
- 不是 NaN 或正负无穷；
- 幅值处于合理范围；
- 不更新训练参数；
- 不读取未来采样点。

## 4. 模型文件和权重路径

建议提交结构：

```text
my_submission/
├─ __init__.py
├─ submission.py
├─ model.py
├─ config.json
├─ weights.pt
├─ requirements.txt
├─ README.md
└─ report/
   └─ TeamID_Task2_Report.pdf
```

权重必须相对 `submission.py` 定位：

```python
from pathlib import Path

SUBMISSION_DIR = Path(__file__).resolve().parent
weight_path = SUBMISSION_DIR / "weights.pt"
```

不要写类似 `C:\Users\...\weights.pt` 或 `/home/user/weights.pt` 的本机
绝对路径。正式环境中的目录位置与参赛者电脑不同。

加载 PyTorch 模型时建议：

```python
state = torch.load(
    weight_path,
    map_location=self.device,
    weights_only=True,
)
self.model.load_state_dict(state)
self.model.to(self.device)
self.model.eval()
```

如果必须保存完整 Python 对象，请确保对应类定义和依赖也包含在 ZIP 内。
优先提交 `state_dict`。

## 5. 复杂度声明

模板要求：

```python
def get_complexity(self) -> dict:
    return {
        "parameter_count": ...,
        "steady_state_macs_per_sample": ...,
        "startup_macs": ...,
        "peak_macs_in_one_sample_event": ...,
    }
```

正式复杂度指标是 `parameter_count` 和
`peak_macs_in_one_sample_event`；另外两个字段用于复核峰值声明。
本规则只规定统计口径，不预设或推荐任何模型结构。
峰值事件必须在完整3.5秒内取最大值，包括0.5秒初始化阶段；不能把初始化时
发生的一次性计算排除在峰值MAC之外。

### 5.1 参数量

$$
N_{\mathrm{param}}
=
\sum_{\theta_j\in\Theta_{\mathrm{resident}}^{\mathrm{unique}}}
\operatorname{numel}(\theta_j)
$$

所有驻留参数都要计入，包括冻结参数、暂未执行但仍加载的子模型和辅助模块。
共享参数按同一参数对象只计算一次。

### 5.2 峰值事件 MAC

$$
\mathrm{MAC}_{\mathrm{peak}}
=
\max_{q\in\mathcal{Q}}
\left[
\sum_{\ell\in\mathcal{E}(q)}
\mathrm{MAC}_{\ell}
\right]
$$

\(\mathcal{Q}\) 是运行过程中所有可能的采样边界和动态分支，
\(\mathcal{E}(q)\) 是事件 \(q\) 实际执行的神经网络层。如果多种模型、
多个子模型或多个计算模块会在同一采样点同时运行，必须把该采样点实际发生
的 MAC 相加，并按最坏事件统计。

常用层：

$$
\mathrm{MAC}_{\mathrm{Linear}}
=
C_{\mathrm{in}}C_{\mathrm{out}}
$$

$$
\mathrm{MAC}_{\mathrm{Conv1d}}
=
L_{\mathrm{out}}C_{\mathrm{out}}
\frac{C_{\mathrm{in}}}{G}K
$$

1 MAC 表示一次乘法累加。偏置相加、激活函数、逐元素运算、内存搬运和
Python 调度不计入神经网络 MAC。

## 6. 声学指标定义

`run_public_demo.py` 会按本节公式计算固定公开场景的声学指标。正式评测使用
相同的指标口径，但测试信号和次级路径不同，因此公开结果不等于最终成绩。
公开测试和正式评测的单场景时序相同：0.5秒初始化后连续进行3秒计分，边界处
不重置模型或次级路径。正式评测包含三个不同测试场景，场景之间重新调用
`reset()`。

3秒计分段划分为六个0.5秒窗口。每窗为24000点；频谱计算采用8192点FFT、
2048点帧移和 Hann 窗。为使窗口末尾样本进入频谱分析，每窗末尾只在STFT
内部补零至完整帧，补零不会送入模型，也不会增加推理时间。

评测信号先转换为 1/3 倍频带功率。中心频率 \(f_i\) 对应的边界为：

$$
f_{\mathrm{low},i}=\frac{f_i}{2^{1/6}},
\qquad
f_{\mathrm{high},i}=f_i2^{1/6}
$$

令 \(P_{\mathrm{off},i,w}\) 和 \(P_{\mathrm{on},i,w}\) 为第 \(w\) 个
0.5秒窗口中 ANC 关闭和开启时第 \(i\) 个频带的功率：

$$
\mathrm{NR}_{i,w}
=
10\log_{10}
\left(
\frac{\max(P_{\mathrm{off},i,w},\varepsilon)}
     {\max(P_{\mathrm{on},i,w},\varepsilon)}
\right)
$$

### 6.1 未加权 1/3 倍频带平均降噪量

在
\(\mathcal{B}_{\mathrm{NR}}=\{i\mid50\ \mathrm{Hz}\le f_i\le
5000\ \mathrm{Hz}\}\) 内，令
\(N_{\mathrm{B}}=|\mathcal{B}_{\mathrm{NR}}|\)：

$$
S_w
=
\frac{1}{N_{\mathrm{B}}}
\sum_{i\in\mathcal{B}_{\mathrm{NR}}}\mathrm{NR}_{i,w}
$$

每个有效 1/3 倍频带等权参与，不添加频率计权。单场景降噪成绩为六个窗口
成绩的等权算术平均：

$$
S_{\mathrm{scene}}
=
\frac{1}{6}\sum_{w=1}^{6}S_w
$$

正式评测的三个场景再次等权平均：

$$
S_{\mathrm{final}}
=
\frac{1}{3}\sum_{q=1}^{3}S_{\mathrm{scene},q}
$$

最终值对应
`primary_score_db` 和 `third_octave_average_nr_50_5000_db`，数值越高
越好。`average_nr_50_5000_db` 是数值相同的兼容字段。

### 6.2 总频带能量降噪量（用作参考，不作记分）

$$
\mathrm{NR}_{\mathrm{total}}
=
10\log_{10}
\left(
\frac{
\max\!\left(
\sum_{i\in\mathcal{B}_{\mathrm{NR}}}
P_{\mathrm{off},i},
\varepsilon
\right)
}{
\max\!\left(
\sum_{i\in\mathcal{B}_{\mathrm{NR}}}
P_{\mathrm{on},i},
\varepsilon
\right)
}
\right)
$$

它对应 `total_band_nr_50_5000_db`。

### 6.3 平均窗口1/3倍频带反弹峰值

在
\(\mathcal{B}_{\mathrm{R}}=\{i\mid1000\ \mathrm{Hz}\le f_i\le
8000\ \mathrm{Hz}\}\) 内：

$$
R_w
=
\max
\left(
0,
-\min_{i\in\mathcal{B}_{\mathrm{R}}}\mathrm{NR}_{i,w}
\right)
$$

每个窗口先在1 kHz～8 kHz内取最严重的1/3倍频带反弹，再计算单场景平均：

$$
R_{\mathrm{scene}}
=
\frac{1}{6}\sum_{w=1}^{6}R_w
$$

三个正式场景的最终反弹量为：

$$
R_{\mathrm{final}}
=
\frac{1}{3}\sum_{q=1}^{3}R_{\mathrm{scene},q}
$$

它对应 `third_octave_rebound_peak_1000_8000_db`，数值越低越好。这里的
“峰值”是每个窗口内部最坏1/3倍频带的峰值；正式字段是这些窗口峰值的平均，
不使用单个FFT栅格，也不取18个窗口中的全局最大值。

## 7. validate_submission.py 会检查什么

公开检查包含：

- 入口能否按 `module:create_model` 导入；
- 工厂函数能否接收 `device`；
- 采样率是否严格为 48000；
- `requires_error` 是否为布尔值；
- 前馈或反馈接口能否严格逐采样运行；
- 输出能否转换为有限 `float`；
- 输出是否超过公开安全上限；
- 两次 `reset()` 后相同输入是否可复现；
- `get_complexity()` 是否包含必需字段；
- 参数量是否与可扫描到的 PyTorch 参数一致；
- 峰值事件 MAC 是否不小于稳态 MAC。

公开检查不能证明：

- 模型一定具有降噪效果；
- 模型在所有正式测试场景中稳定；
- MAC 声明已经覆盖所有自定义算子；
- 提交是否包含正式环境未提供的第三方依赖；
- 提交一定满足全部比赛规则。

因此在最终上传前还应进行自己的训练集外验证和代码审查。

通过 `validate_submission.py` 后，可运行 `run_public_demo.py` 检查固定
公开场景上的真实闭环效果。后者仍不能证明模型在所有正式场景中稳定。

## 8. 提交 ZIP 规则

- ZIP 第一层目录必须为与队伍 ID 一致的 `TeamID/`；
- 第二层目录必须固定为 `Task2/`；
- `TeamID/Task2/` 必须直接包含 `submission.py`、`requirements.txt` 和
  `README.md`；
- `TeamID/Task2/report/` 必须且只能包含一份有效 PDF；
- 目录和 Python 文件名使用英文字母、数字和下划线；
- 包含推理所需的全部代码、配置、权重和依赖说明；
- 不包含训练数据、缓存、日志、优化器状态和无关大型文件；
- 不包含符号链接；
- 文件数不超过 1000；
- 解压后总大小不超过 2 GiB；
- 上传前记录 ZIP 的 SHA-256。

`make_submission_zip.py` 会自动忽略 `__pycache__`、`.git`、`.vscode`、
`.pytest_cache`、`.pyc`、`.pyo` 和已有 ZIP，并自动为提交内容增加
`TeamID/Task2/` 前缀。它还会检查三个根部必需文件，拒绝误放入提交目录的
`public_demo_data`。模板中的 `report/README.md` 也不会进入最终 ZIP。
完整要求见
[赛题二交付格式说明.md](赛题二交付格式说明.md)。

## 9. 运行和安全要求

参赛模型必须能够离线运行。推理过程中不得：

- 访问网络；
- 读取提交目录之外的文件；
- 获取正式测试数据或评测程序内部状态；
- 启动外部服务或子进程；
- 修改评测环境；
- 使用未来输入；
- 在不同测试录音之间保留状态；
- 在推理过程中更新训练参数。

只应使用模型自身、提交包内文件以及公开接口提供的当前/历史输入。

## 10. 常见问题

### 无法导入 my_submission

确认当前终端位于本工具包根目录，并且存在：

```text
my_submission\__init__.py
my_submission\submission.py
```

入口必须写成：

```text
my_submission.submission:create_model
```

### 找不到权重

不要依赖当前终端目录。使用：

```python
Path(__file__).resolve().parent / "weights.pt"
```

### 参数量不一致

公开检查器会递归扫描提交对象中可找到的 `torch.nn.Module`。确保
`parameter_count` 包含所有常驻模块，不得遗漏暂未执行但仍加载的模块。

### reset() 检查失败

检查是否遗漏了缓存、隐藏状态、随机采样状态或其他内部状态。推理模型应调用
`eval()`，并避免在 `process_sample()` 中使用随机操作。

### 模型只能整段运行

不能把未来样本传给模型。必须实现因果缓存或网络的 `step()` 接口，使当前
输出只依赖当前及过去输入。详细方法见 `MODEL_PREPARATION_GUIDE.md`。

### 公开检查通过后，怎样看到真实降噪量

运行：

```bat
python run_public_demo.py --entry-point my_submission.submission:create_model --device cpu
```

脚本固定运行0.5秒初始化和3秒正式计分，不允许自行改变时长。它严格逐采样
运行，复杂网络处理3.5秒可能明显慢于音频时长，这是模型真实逐点调用成本的
一部分。
