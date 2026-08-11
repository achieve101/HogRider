"""DEEPANC 反馈式参赛模型模板。

反馈模型在时刻 t 接收 x[t] 和 e[t-1]。previous_error_sample 是已经
测得的上一采样点残差，不是当前残差，也不是未降噪期望噪声。

当前占位实现始终输出 0，只用于验证文件结构和接口，不能产生降噪效果。
"""

from __future__ import annotations

from pathlib import Path

import torch


SUBMISSION_DIR = Path(__file__).resolve().parent


class FeedbackSubmission:
    """接收参考信号 x[t] 和上一采样点残差 e[t-1] 的控制器包装。"""

    sample_rate = 48_000
    requires_error = True

    def __init__(self, device: str = "cpu") -> None:
        self.device = torch.device(device)

        # ===== 参赛者修改区域：创建并加载自己的模型 =====
        #
        # from .model import MyFeedbackANC
        # self.model = MyFeedbackANC(...)
        # state = torch.load(
        #     SUBMISSION_DIR / "weights.pt",
        #     map_location=self.device,
        #     weights_only=True,
        # )
        # self.model.load_state_dict(state)
        # self.model.to(self.device)
        # self.model.eval()

        self.model = None

    def reset(self) -> None:
        """清空模型的输入历史、误差历史和所有隐藏状态。"""

        # ===== 参赛者修改区域 =====
        #
        # self.reference_history.zero_()
        # self.error_history.zero_()
        # self.hidden_state = None
        # self.model.reset_streaming_state()
        return None

    def process_sample(
        self,
        reference_sample: float,
        previous_error_sample: float,
    ) -> float:
        """由 x[t]、e[t-1] 及更早历史产生当前控制输出 y[t]。

        previous_error_sample 已经延迟一个采样点，因此不会形成代数环。
        模型不能访问 e[t] 或任何未来数据。
        """

        # ===== 参赛者必须替换下面的占位实现 =====
        #
        # input_t = torch.tensor(
        #     [[reference_sample, previous_error_sample]],
        #     dtype=torch.float32,
        #     device=self.device,
        # )
        # y_t, self.hidden_state = self.model.step(
        #     input_t,
        #     self.hidden_state,
        # )
        # return float(y_t.item())

        del reference_sample, previous_error_sample
        return 0.0

    def get_complexity(self) -> dict:
        """申报所有常驻模块和最坏动态分支的复杂度。"""

        # 如果多个模型、子模型或辅助模块同时驻留，parameter_count 必须
        # 包含它们的全部参数。峰值事件 MAC 必须覆盖最坏分支；多个模块
        # 在同一采样点运行时，必须累计该采样点的全部 MAC。
        return {
            "parameter_count": 0,
            "steady_state_macs_per_sample": 0,
            "startup_macs": 0,
            "peak_macs_in_one_sample_event": 0,
        }


def create_model(device: str = "cpu") -> FeedbackSubmission:
    """评测器使用的固定工厂函数；名称和 device 参数不要删除。"""
    return FeedbackSubmission(device=device)
