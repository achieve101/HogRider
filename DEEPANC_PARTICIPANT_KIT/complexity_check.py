"""参赛模型复杂度声明的公开检查工具。"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import torch


class ComplexityError(RuntimeError):
    """复杂度声明缺失、非法或与模型参数不一致。"""


def _audit_torch_parameters(model: Any) -> int | None:
    """递归查找 PyTorch 模块，并按参数对象去重统计。"""
    visited_objects: set[int] = set()
    parameter_objects: dict[int, torch.nn.Parameter] = {}

    def visit(value: Any, depth: int = 0) -> None:
        if value is None or depth > 12:
            return
        object_id = id(value)
        if object_id in visited_objects:
            return
        visited_objects.add(object_id)

        if isinstance(value, torch.nn.Module):
            for parameter in value.parameters():
                parameter_objects[id(parameter)] = parameter
            return
        if isinstance(
            value,
            (
                str,
                bytes,
                int,
                float,
                bool,
                complex,
                Path,
                np.ndarray,
                torch.Tensor,
                torch.device,
            ),
        ):
            return
        if isinstance(value, dict):
            for item in value.values():
                visit(item, depth + 1)
            return
        if isinstance(value, (list, tuple, set)):
            for item in value:
                visit(item, depth + 1)
            return

        attributes = getattr(value, "__dict__", None)
        if isinstance(attributes, dict):
            for item in attributes.values():
                visit(item, depth + 1)

    visit(model)
    if not parameter_objects:
        return None
    return int(
        sum(parameter.numel() for parameter in parameter_objects.values())
    )


def _nonnegative_number(values: dict, key: str) -> float:
    value = values.get(key)
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or float(value) < 0
    ):
        raise ComplexityError(
            f"复杂度字段 {key!r} 必须是有限非负数。"
        )
    return float(value)


def validate_complexity(model: Any) -> dict:
    """校验 ``get_complexity()`` 并返回规范化结果。"""
    declared = model.get_complexity()
    if not isinstance(declared, dict):
        raise ComplexityError(
            "model.get_complexity() 必须返回 dict。"
        )

    parameter_count_value = _nonnegative_number(
        declared, "parameter_count"
    )
    if not parameter_count_value.is_integer():
        raise ComplexityError("parameter_count 必须是整数。")
    parameter_count = int(parameter_count_value)

    steady_macs = _nonnegative_number(
        declared, "steady_state_macs_per_sample"
    )
    startup_macs = _nonnegative_number(declared, "startup_macs")
    peak_macs = _nonnegative_number(
        declared, "peak_macs_in_one_sample_event"
    )
    if peak_macs < steady_macs:
        raise ComplexityError(
            "peak_macs_in_one_sample_event 不能小于"
            " steady_state_macs_per_sample。"
        )

    audited_count = _audit_torch_parameters(model)
    if (
        audited_count is not None
        and audited_count != parameter_count
    ):
        raise ComplexityError(
            "get_complexity() 申报的参数量与公开检查器扫描结果不一致："
            f"申报 {parameter_count}，扫描 {audited_count}。"
        )

    return {
        "parameter_count": parameter_count,
        "parameter_count_audited": audited_count,
        "steady_state_macs_per_sample": steady_macs,
        "startup_macs": startup_macs,
        "peak_macs_in_one_sample_event": peak_macs,
    }
