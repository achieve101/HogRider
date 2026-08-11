"""从公开模板创建一个不会覆盖已有文件的参赛目录。"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


KIT_DIR = Path(__file__).resolve().parent


def _resolve_output(path_text: str) -> tuple[Path, Path]:
    relative = Path(path_text)
    if relative.is_absolute() or not relative.parts:
        raise ValueError("--output 必须是当前目录内的相对路径。")
    if any(part in {"", ".", ".."} for part in relative.parts):
        raise ValueError("--output 不能包含空目录、'.' 或 '..'。")
    invalid = [part for part in relative.parts if not part.isidentifier()]
    if invalid:
        raise ValueError(
            "提交目录名必须能作为 Python 模块导入，只能使用字母、"
            "数字和下划线，且不能以数字开头。无效部分："
            f"{invalid}"
        )

    working_directory = Path.cwd().resolve()
    resolved = (working_directory / relative).resolve()
    if (
        resolved != working_directory
        and working_directory not in resolved.parents
    ):
        raise ValueError("--output 必须位于当前工作目录内。")
    return relative, resolved


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-type",
        required=True,
        choices=["feedforward", "feedback"],
        help="feedforward=前馈；feedback=需要上一采样点误差。",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="新提交目录，例如 my_submission。",
    )
    args = parser.parse_args()

    relative, target = _resolve_output(args.output)
    if target.exists():
        raise FileExistsError(
            f"目标已经存在，为防止覆盖已停止：{target}"
        )

    source = (
        KIT_DIR
        / "submission_templates"
        / f"{args.model_type}_model"
    )
    shutil.copytree(source, target)

    module_path = ".".join(relative.parts)
    entry_point = f"{module_path}.submission:create_model"
    print("提交目录创建完成")
    print(f"模型类型：{args.model_type}")
    print(f"目录：{target}")
    print(f"入口：{entry_point}")
    print("\n下一步：")
    print(f"1. 编辑 {target / 'submission.py'}")
    print("2. 放入模型代码和权重")
    print(f"3. 把一份 PDF 技术报告放入 {target / 'report'}")
    print("4. 运行公开检查：")
    print(
        "python validate_submission.py "
        f"--entry-point {entry_point} --device cpu"
    )


if __name__ == "__main__":
    main()
