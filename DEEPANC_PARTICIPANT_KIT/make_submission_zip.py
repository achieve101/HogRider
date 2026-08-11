"""把已经通过公开检查的参赛目录打包成规范 ZIP。"""

from __future__ import annotations

import argparse
import hashlib
import re
import zipfile
from pathlib import Path


EXCLUDED_DIRECTORIES = {
    "__pycache__",
    ".git",
    ".vscode",
    ".pytest_cache",
}
EXCLUDED_SUFFIXES = {".pyc", ".pyo", ".zip"}
EXCLUDED_RELATIVE_PATHS = {"report/README.md"}
REQUIRED_ROOT_FILES = {
    "submission.py",
    "requirements.txt",
    "README.md",
}
PROHIBITED_PATH_PARTS = {"public_demo_data"}
MAX_FILE_COUNT = 1_000
MAX_TOTAL_BYTES = 2 * 1024**3
TEAM_ID_PATTERN = re.compile(r"^[A-Za-z][A-Za-z0-9_]*$")


def _submission_files(source: Path) -> list[Path]:
    files: list[Path] = []
    for path in sorted(source.rglob("*")):
        relative = path.relative_to(source)
        if any(part in EXCLUDED_DIRECTORIES for part in relative.parts):
            continue
        if relative.as_posix() in EXCLUDED_RELATIVE_PATHS:
            continue
        if path.is_symlink():
            raise ValueError(f"提交目录不允许包含符号链接：{path}")
        if not path.is_file() or path.suffix.lower() in EXCLUDED_SUFFIXES:
            continue
        files.append(path)
    return files


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--team-id",
        required=True,
        help="队伍 ID：英文字母开头，只能包含字母、数字和下划线。",
    )
    parser.add_argument(
        "--source",
        required=True,
        help="已经通过公开检查的提交目录。",
    )
    parser.add_argument(
        "--output",
        required=True,
        help=r"输出 ZIP，例如 dist\TeamID.zip。",
    )
    args = parser.parse_args()

    team_id = str(args.team_id)
    if not TEAM_ID_PATTERN.fullmatch(team_id):
        raise ValueError(
            "--team-id 必须以英文字母开头，且只能包含"
            "英文字母、数字和下划线。"
        )

    source = Path(args.source).expanduser().resolve()
    output = Path(args.output).expanduser().resolve()
    if not source.is_dir():
        raise FileNotFoundError(f"找不到提交目录：{source}")
    missing_root_files = sorted(
        name
        for name in REQUIRED_ROOT_FILES
        if not (source / name).is_file()
    )
    if missing_root_files:
        raise FileNotFoundError(
            "提交目录根部缺少必需文件："
            f"{missing_root_files}。"
        )
    if output.suffix.lower() != ".zip":
        raise ValueError("--output 必须以 .zip 结尾。")
    if output.exists():
        raise FileExistsError(
            f"为防止覆盖已有压缩包，操作已停止：{output}"
        )
    if output == source or source in output.parents:
        raise ValueError("输出 ZIP 不能放在待打包提交目录内部。")

    report_directory = source / "report"
    report_pdfs = (
        sorted(report_directory.rglob("*.pdf"))
        if report_directory.is_dir()
        else []
    )
    if len(report_pdfs) != 1:
        raise FileNotFoundError(
            "提交目录的 report/ 中必须且只能包含一份 PDF 技术报告，"
            f"当前找到 {len(report_pdfs)} 份。"
        )
    with report_pdfs[0].open("rb") as report_file:
        if report_file.read(5) != b"%PDF-":
            raise ValueError(
                f"技术报告不是有效的 PDF 文件：{report_pdfs[0]}"
            )

    files = _submission_files(source)
    prohibited_files = [
        path.relative_to(source).as_posix()
        for path in files
        if any(
            part in PROHIBITED_PATH_PARTS
            for part in path.relative_to(source).parts
        )
    ]
    if prohibited_files:
        raise ValueError(
            "提交目录包含只允许本地验证使用的 public_demo_data；"
            "请删除后重新打包。"
        )
    if len(files) > MAX_FILE_COUNT:
        raise ValueError(
            f"提交文件数 {len(files)} 超过上限 {MAX_FILE_COUNT}。"
        )
    total_bytes = sum(path.stat().st_size for path in files)
    if total_bytes > MAX_TOTAL_BYTES:
        raise ValueError("提交解压后的总大小不能超过 2 GiB。")

    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(
        output,
        mode="x",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=6,
    ) as archive:
        for path in files:
            archive_path = (
                Path(team_id)
                / "Task2"
                / path.relative_to(source)
            )
            archive.write(
                path,
                arcname=archive_path.as_posix(),
            )

    print("提交 ZIP 创建完成")
    print(f"文件：{output}")
    print(f"文件数：{len(files)}")
    print(f"原始总大小：{total_bytes:,} 字节")
    print(f"ZIP 大小：{output.stat().st_size:,} 字节")
    print(f"SHA-256：{_sha256(output)}")
    print(
        "\n请保存上面的 SHA-256，并在上传前确认 ZIP 中存在："
    )
    print(
        f"{team_id}/Task2/submission.py"
    )


if __name__ == "__main__":
    main()
