"""DEEPANC 参赛模型公开接口。

本文件只定义参赛模型必须实现的公开契约，不包含声学路径、测试场景或
评分数据。
"""

from __future__ import annotations

import importlib
from typing import Any


SAMPLE_RATE = 48_000


class SubmissionInterfaceError(RuntimeError):
    """参赛模型没有满足公开接口要求。"""


def load_submission(entry_point: str, device: str) -> Any:
    """从 ``python.module:create_model`` 入口创建并检查模型。"""
    module_name, separator, factory_name = str(entry_point).partition(":")
    if not separator or not module_name or not factory_name:
        raise SubmissionInterfaceError(
            "入口必须采用 python.module:create_model 格式。"
        )

    try:
        module = importlib.import_module(module_name)
    except Exception as exc:
        raise SubmissionInterfaceError(
            f"无法导入模块 {module_name!r}。请从工具包根目录运行命令，"
            "并检查目录名、文件名和依赖。"
        ) from exc

    factory = getattr(module, factory_name, None)
    if not callable(factory):
        raise SubmissionInterfaceError(
            f"{entry_point!r} 指向的工厂函数不存在或不可调用。"
        )

    try:
        model = factory(device=device)
    except Exception as exc:
        raise SubmissionInterfaceError(
            f"调用 {entry_point!r} 创建模型失败。"
        ) from exc

    validate_model(model)
    return model


def validate_model(model: Any) -> None:
    """检查私有评测器调用模型时依赖的公开属性。"""
    sample_rate = getattr(model, "sample_rate", None)
    if sample_rate != SAMPLE_RATE:
        raise SubmissionInterfaceError(
            f"model.sample_rate 必须为 {SAMPLE_RATE}，"
            f"当前为 {sample_rate!r}。"
        )

    requires_error = getattr(model, "requires_error", None)
    if not isinstance(requires_error, bool):
        raise SubmissionInterfaceError(
            "model.requires_error 必须明确设置为 bool。"
        )

    if not callable(getattr(model, "reset", None)):
        raise SubmissionInterfaceError("模型必须实现 reset()。")
    if not callable(getattr(model, "process_sample", None)):
        raise SubmissionInterfaceError(
            "模型必须实现 process_sample(...)。"
        )
    if not callable(getattr(model, "get_complexity", None)):
        raise SubmissionInterfaceError(
            "模型必须实现 get_complexity()。"
        )
