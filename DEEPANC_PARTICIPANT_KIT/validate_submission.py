"""在公开合成输入上检查参赛模型接口、状态和复杂度声明。"""

from __future__ import annotations

import argparse
import math
import time

import numpy as np
import torch

from complexity_check import validate_complexity
from participant_api import SAMPLE_RATE, load_submission


class ValidationError(RuntimeError):
    """公开提交检查未通过。"""


def _test_inputs(samples: int) -> tuple[np.ndarray, np.ndarray]:
    """生成确定性的公开参考信号和上一采样点误差信号。"""
    time_seconds = np.arange(samples, dtype=np.float64) / SAMPLE_RATE
    reference = (
        0.040 * np.sin(2.0 * np.pi * 233.0 * time_seconds)
        + 0.015 * np.sin(2.0 * np.pi * 997.0 * time_seconds)
    )
    synthetic_error = (
        0.025 * np.sin(2.0 * np.pi * 317.0 * time_seconds)
    )
    previous_error = np.zeros(samples, dtype=np.float64)
    if samples > 1:
        previous_error[1:] = synthetic_error[:-1]
    return reference, previous_error


def _run_once(
    model: object,
    reference: np.ndarray,
    previous_error: np.ndarray,
    output_limit: float,
) -> tuple[np.ndarray, float]:
    """重置模型并严格逐采样运行一次。"""
    model.reset()
    outputs = np.empty(reference.size, dtype=np.float64)
    started = time.perf_counter()

    with torch.inference_mode():
        for index in range(reference.size):
            x_t = float(reference[index])
            try:
                if model.requires_error:
                    output = model.process_sample(
                        x_t,
                        float(previous_error[index]),
                    )
                else:
                    output = model.process_sample(x_t)
            except Exception as exc:
                raise ValidationError(
                    f"第 {index} 个采样点调用 process_sample() 失败。"
                ) from exc

            try:
                output_float = float(output)
            except (TypeError, ValueError) as exc:
                raise ValidationError(
                    f"第 {index} 个输出不能转换为 Python float："
                    f"{output!r}"
                ) from exc

            if not math.isfinite(output_float):
                raise ValidationError(
                    f"第 {index} 个输出不是有限数：{output_float!r}。"
                )
            if abs(output_float) > output_limit:
                raise ValidationError(
                    f"第 {index} 个输出绝对值 {abs(output_float):.6g} "
                    f"超过公开检查上限 {output_limit:g}。"
                )
            outputs[index] = output_float

    return outputs, time.perf_counter() - started


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--entry-point",
        required=True,
        help="例如 my_submission.submission:create_model。",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="模型运行设备，例如 cpu 或 cuda:0。",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=4096,
        help="公开检查样本数，默认 4096。",
    )
    parser.add_argument(
        "--output-limit",
        type=float,
        default=100.0,
        help="公开检查允许的输出绝对值上限。",
    )
    args = parser.parse_args()

    if args.samples <= 0:
        raise ValueError("--samples 必须大于 0。")
    if not math.isfinite(args.output_limit) or args.output_limit <= 0:
        raise ValueError("--output-limit 必须是有限正数。")

    print("1/4 正在加载参赛模型……")
    model = load_submission(args.entry_point, args.device)

    print("2/4 正在核对复杂度声明……")
    complexity = validate_complexity(model)

    print("3/4 正在进行严格逐采样调用……")
    reference, previous_error = _test_inputs(args.samples)
    first_outputs, elapsed = _run_once(
        model,
        reference,
        previous_error,
        args.output_limit,
    )

    print("4/4 正在检查 reset() 可复现性……")
    second_outputs, _ = _run_once(
        model,
        reference,
        previous_error,
        args.output_limit,
    )
    if not np.allclose(
        first_outputs,
        second_outputs,
        rtol=1e-6,
        atol=1e-7,
        equal_nan=False,
    ):
        maximum_difference = float(
            np.max(np.abs(first_outputs - second_outputs))
        )
        raise ValidationError(
            "两次 reset() 后相同输入没有得到可复现输出；"
            f"最大差值为 {maximum_difference:.6g}。"
        )

    signal_seconds = args.samples / SAMPLE_RATE
    print("\n公开提交检查通过")
    print(f"入口：{args.entry_point}")
    print(
        "模型类型："
        + ("反馈式" if model.requires_error else "前馈式")
    )
    print(f"采样率：{model.sample_rate} Hz")
    print(f"检查样本数：{args.samples}")
    print(f"输出峰值：{np.max(np.abs(first_outputs)):.6g}")
    print(f"模型总参数量：{complexity['parameter_count']:,}")
    audited = complexity["parameter_count_audited"]
    if audited is not None:
        print(f"PyTorch 参数扫描值：{audited:,}")
    print(
        "峰值事件 MAC："
        f"{complexity['peak_macs_in_one_sample_event']:,.0f}"
    )
    print(f"逐点运行时间：{elapsed:.3f} s")
    print(f"公开输入实时系数：{elapsed / signal_seconds:.3f} x")
    print(
        "\n注意：通过本检查只表示提交接口和文件结构基本正确，"
        "不代表私有声学测试分数达标。"
    )


if __name__ == "__main__":
    main()
