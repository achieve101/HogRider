"""检查参赛者本地 Python、NumPy、SoundFile、PyTorch 和计算设备。"""

from __future__ import annotations

import argparse
import platform
import sys
from pathlib import Path


KIT_DIR = Path(__file__).resolve().parent


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--device",
        default="cpu",
        help="希望检查的设备，例如 cpu 或 cuda:0。",
    )
    args = parser.parse_args()

    try:
        import numpy as np
    except ImportError as exc:
        raise RuntimeError(
            "没有安装 NumPy。请运行：pip install -r requirements.txt"
        ) from exc

    try:
        import torch
    except ImportError as exc:
        raise RuntimeError(
            "没有安装 PyTorch。请运行：pip install -r requirements.txt"
        ) from exc

    try:
        import soundfile as sf
    except ImportError as exc:
        raise RuntimeError(
            "没有安装 SoundFile。请运行：pip install -r requirements.txt"
        ) from exc

    if sys.version_info < (3, 10):
        raise RuntimeError(
            f"需要 Python 3.10 或更高版本，当前为 {platform.python_version()}。"
        )

    device = torch.device(args.device)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "指定了 CUDA 设备，但当前 PyTorch 检测不到可用 CUDA。"
            )
        if device.index is not None:
            torch.cuda.get_device_properties(device)

    print("环境检查通过")
    print(f"工具包目录：{KIT_DIR}")
    print(f"Python：{platform.python_version()}")
    print(f"Python 可执行文件：{sys.executable}")
    print(f"NumPy：{np.__version__}")
    print(f"SoundFile：{sf.__version__}")
    print(f"PyTorch：{torch.__version__}")
    print(f"CUDA 可用：{torch.cuda.is_available()}")
    print(f"目标设备：{device}")


if __name__ == "__main__":
    main()
