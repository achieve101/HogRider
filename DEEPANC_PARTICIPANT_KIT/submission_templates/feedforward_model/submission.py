"""DEEPANC 前馈式参赛模型模板。

使用方法：
1. 把网络结构代码和权重放在本目录；
2. 在 __init__() 中创建网络并加载权重；
3. 在 reset() 中清空所有流式状态；
4. 在 process_sample() 中实现严格因果的逐采样推理；
5. 在 get_complexity() 中填写经过复核的复杂度。

当前占位实现始终输出 0，只用于验证文件结构和接口，不能产生降噪效果。
"""

from __future__ import annotations

from pathlib import Path

import torch


SUBMISSION_DIR = Path(__file__).resolve().parent


class FeedforwardSubmission:
    """只接收参考信号 x[t] 的前馈式控制器包装。"""

    # 竞赛固定采样率，不能修改。
    sample_rate = 48_000

    # False 表示 process_sample() 只接收 reference_sample。
    requires_error = False

    def __init__(self, device: str = "cpu") -> None:
        self.device = torch.device(device)

        # ===== 参赛者修改区域：创建并加载自己的模型 =====
        #
        # 推荐把网络类放在本目录的 model.py：
        #
        # from .model import MyStreamingANC
        # self.model = MyStreamingANC(...)
        #
        # 权重路径必须相对 submission.py 解析，不能写本机绝对路径：
        #
        # weight_path = SUBMISSION_DIR / "weights.pt"
        # state = torch.load(
        #     weight_path,
        #     map_location=self.device,
        #     weights_only=True,
        # )
        # self.model.load_state_dict(state)
        # self.model.to(self.device)
        # self.model.eval()
        #
        # 如果网络需要输入历史、卷积缓存或 RNN 状态，请在这里创建容器，
        # 并在 reset() 中真正清零。

        self.model = None

    def reset(self) -> None:
        """每段新录音开始前清空全部因果状态。

        不能保留上一段录音的输入历史、卷积缓存、RNN 隐状态、内部决策
        状态或自适应状态。
        """

        # ===== 参赛者修改区域 =====
        #
        # 示例：
        # self.input_history.zero_()
        # self.hidden_state = None
        # self.model.reset_streaming_state()
        return None

    def process_sample(self, reference_sample: float) -> float:
        """根据当前和过去的 x 产生一个当前控制输出 y[t]。

        这里不能读取未来参考信号，也不能一次接收完整测试音频。返回值必须
        能转换为有限的 Python float。
        """

        # ===== 参赛者必须替换下面的占位实现 =====
        #
        # 伪代码示例：
        # x_t = torch.tensor(
        #     [[reference_sample]],
        #     dtype=torch.float32,
        #     device=self.device,
        # )
        # y_t, self.hidden_state = self.model.step(
        #     x_t,
        #     self.hidden_state,
        # )
        # return float(y_t.item())

        del reference_sample
        return 0.0

    def get_complexity(self) -> dict:
        """申报全部驻留参数和最坏事件 MAC。

        正式复杂度指标是 parameter_count 和
        peak_macs_in_one_sample_event。另外两个字段用于公开检查和
        复杂度复核。
        """

        # ===== 参赛者必须按自己的网络修改 =====
        return {
            "parameter_count": 0,
            "steady_state_macs_per_sample": 0,
            "startup_macs": 0,
            "peak_macs_in_one_sample_event": 0,
        }


def create_model(device: str = "cpu") -> FeedforwardSubmission:
    """评测器使用的固定工厂函数；名称和 device 参数不要删除。"""
    return FeedforwardSubmission(device=device)
