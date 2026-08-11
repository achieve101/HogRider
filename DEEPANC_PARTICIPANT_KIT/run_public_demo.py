"""在固定公开场景上逐采样运行参赛模型并计算真实闭环指标。"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
import torch

from complexity_check import validate_complexity
from participant_api import SAMPLE_RATE, load_submission
from public_demo_scoring import score_windowed_signals


KIT_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = KIT_DIR / "public_demo_data"
SCENE_ID = "vehicle_public_scene_01"
PROTOCOL_VERSION = 6
INITIALIZATION_SECONDS = 0.5
SCORING_SECONDS = 3.0
SCORING_WINDOW_SECONDS = 0.5
INITIALIZATION_SAMPLES = int(round(INITIALIZATION_SECONDS * SAMPLE_RATE))
SCORING_SAMPLES = int(round(SCORING_SECONDS * SAMPLE_RATE))
SCORING_WINDOW_SAMPLES = int(
    round(SCORING_WINDOW_SECONDS * SAMPLE_RATE)
)
SCORING_WINDOW_COUNT = SCORING_SAMPLES // SCORING_WINDOW_SAMPLES
TOTAL_SAMPLES = INITIALIZATION_SAMPLES + SCORING_SAMPLES


class PublicDemoError(RuntimeError):
    """公开演示数据或模型运行不符合要求。"""


class _SecondaryPath:
    """严格因果的逐点 FIR 次级路径。"""

    def __init__(self, impulse_response: np.ndarray) -> None:
        self.impulse_response = np.asarray(
            impulse_response, dtype=np.float64
        ).reshape(-1)
        if self.impulse_response.size == 0:
            raise PublicDemoError("公开次级路径不能为空。")
        self.length = int(self.impulse_response.size)
        self.buffer = np.zeros(2 * self.length, dtype=np.float64)
        self.position = 0

    def process(self, controller_sample: float) -> float:
        self.position = (self.position - 1) % self.length
        self.buffer[self.position] = controller_sample
        self.buffer[self.position + self.length] = controller_sample
        history = self.buffer[
            self.position : self.position + self.length
        ]
        return float(np.dot(self.impulse_response, history))


def _finite_nonnegative(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed < 0:
        raise argparse.ArgumentTypeError("必须是有限非负数。")
    return parsed


def _finite_positive(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed <= 0:
        raise argparse.ArgumentTypeError("必须是有限正数。")
    return parsed


def _read_audio(
    path: Path,
    start_sample: int,
    frames: int,
) -> np.ndarray:
    if not path.is_file():
        raise PublicDemoError(f"找不到公开演示音频：{path}")
    info = sf.info(path)
    if int(info.samplerate) != SAMPLE_RATE:
        raise PublicDemoError(
            f"{path.name} 的采样率为 {info.samplerate} Hz，"
            f"要求 {SAMPLE_RATE} Hz。"
        )
    if info.channels != 1:
        raise PublicDemoError(f"{path.name} 必须是单通道音频。")
    if start_sample + frames > int(info.frames):
        available_seconds = info.frames / SAMPLE_RATE
        raise PublicDemoError(
            "所选起点和固定3.5秒评测时长超过公开音频范围；"
            f"公开音频总长为 {available_seconds:.3f} 秒。"
        )
    audio, _ = sf.read(
        path,
        start=start_sample,
        frames=frames,
        dtype="float32",
        always_2d=False,
    )
    audio_array = np.asarray(audio, dtype=np.float64).reshape(-1)
    if not np.all(np.isfinite(audio_array)):
        raise PublicDemoError(f"{path.name} 包含 NaN 或 Inf。")
    return audio_array


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file_object:
        for chunk in iter(lambda: file_object.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verify_data_manifest(data_dir: Path) -> None:
    manifest_path = data_dir / "manifest.json"
    if not manifest_path.is_file():
        raise PublicDemoError(f"找不到公开场景清单：{manifest_path}")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PublicDemoError("公开场景 manifest.json 无法读取。") from exc

    if manifest.get("sample_rate_hz") != SAMPLE_RATE:
        raise PublicDemoError("公开场景清单中的采样率不正确。")
    declared_files = manifest.get("files")
    if not isinstance(declared_files, dict):
        raise PublicDemoError("公开场景清单缺少 files。")
    for file_name in (
        "reference.wav",
        "disturbance.wav",
        "secondary_path.npy",
    ):
        path = data_dir / file_name
        file_info = declared_files.get(file_name)
        expected_hash = (
            file_info.get("sha256")
            if isinstance(file_info, dict)
            else None
        )
        if not path.is_file() or not isinstance(expected_hash, str):
            raise PublicDemoError(
                f"公开场景文件或哈希声明缺失：{file_name}"
            )
        if _sha256(path) != expected_hash.lower():
            raise PublicDemoError(
                f"{file_name} 的 SHA-256 不匹配；"
                "请重新取得未经修改的参赛者工具包。"
            )


def _load_scene(
    data_dir: Path,
    start_seconds: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    _verify_data_manifest(data_dir)
    start_sample = int(round(start_seconds * SAMPLE_RATE))

    reference = _read_audio(
        data_dir / "reference.wav",
        start_sample,
        TOTAL_SAMPLES,
    )
    disturbance = _read_audio(
        data_dir / "disturbance.wav",
        start_sample,
        TOTAL_SAMPLES,
    )
    path_file = data_dir / "secondary_path.npy"
    if not path_file.is_file():
        raise PublicDemoError(f"找不到公开次级路径：{path_file}")
    impulse_response = np.asarray(
        np.load(path_file, allow_pickle=False),
        dtype=np.float64,
    )
    if impulse_response.ndim != 1 or impulse_response.size == 0:
        raise PublicDemoError("secondary_path.npy 必须是一维非空数组。")
    if not np.all(np.isfinite(impulse_response)):
        raise PublicDemoError("公开次级路径中包含 NaN 或 Inf。")
    return (
        reference,
        disturbance,
        np.ascontiguousarray(impulse_response),
    )


def _validated_output(
    raw_output: Any,
    index: int,
    output_limit: float,
) -> float:
    try:
        output = float(raw_output)
    except (TypeError, ValueError) as exc:
        raise PublicDemoError(
            f"第 {index} 个采样点的模型输出不能转换为 float。"
        ) from exc
    if not math.isfinite(output):
        raise PublicDemoError(
            f"第 {index} 个采样点的模型输出为 NaN 或 Inf。"
        )
    if abs(output) > output_limit:
        raise PublicDemoError(
            f"第 {index} 个采样点的输出幅值 {abs(output):.6g} "
            f"超过安全上限 {output_limit:.6g}。"
        )
    return output


def _run_closed_loop(
    model: Any,
    reference: np.ndarray,
    disturbance: np.ndarray,
    impulse_response: np.ndarray,
    output_limit: float,
    progress_every_seconds: float,
) -> tuple[np.ndarray, np.ndarray, float]:
    secondary_path = _SecondaryPath(impulse_response)
    outputs = np.empty(reference.size, dtype=np.float64)
    residuals = np.empty(reference.size, dtype=np.float64)
    previous_error = 0.0
    progress_samples = max(
        1, int(round(progress_every_seconds * SAMPLE_RATE))
    )
    model.reset()

    started = time.perf_counter()
    with torch.inference_mode():
        for index, reference_value in enumerate(reference):
            reference_sample = float(reference_value)
            if model.requires_error:
                raw_output = model.process_sample(
                    reference_sample,
                    float(previous_error),
                )
            else:
                raw_output = model.process_sample(reference_sample)

            output = _validated_output(
                raw_output,
                index=index,
                output_limit=output_limit,
            )
            anti_noise = secondary_path.process(output)
            current_error = float(disturbance[index] - anti_noise)
            if not math.isfinite(current_error):
                raise PublicDemoError(
                    f"第 {index} 个采样点的闭环残差为 NaN 或 Inf。"
                )
            outputs[index] = output
            residuals[index] = current_error
            previous_error = current_error

            completed = index + 1
            if (
                completed < reference.size
                and completed % progress_samples == 0
            ):
                elapsed = time.perf_counter() - started
                print(
                    "进度："
                    f"{completed / reference.size:.1%}，"
                    f"已处理 {completed / SAMPLE_RATE:.2f} 秒音频，"
                    f"耗时 {elapsed:.1f} 秒",
                    flush=True,
                )

    return outputs, residuals, time.perf_counter() - started


def evaluate(
    entry_point: str,
    device: str,
    start_seconds: float,
    output_limit: float,
    progress_every_seconds: float,
    data_dir: Path = DEFAULT_DATA_DIR,
) -> dict[str, Any]:
    """加载模型，在公开场景运行，并返回可序列化的标量结果。"""
    model = load_submission(entry_point, device)
    complexity = validate_complexity(model)
    reference, disturbance, impulse_response = _load_scene(
        data_dir=data_dir,
        start_seconds=start_seconds,
    )
    outputs, residuals, runtime_seconds = _run_closed_loop(
        model=model,
        reference=reference,
        disturbance=disturbance,
        impulse_response=impulse_response,
        output_limit=output_limit,
        progress_every_seconds=progress_every_seconds,
    )
    score_slice = slice(INITIALIZATION_SAMPLES, TOTAL_SAMPLES)
    metrics = score_windowed_signals(
        disturbance[score_slice],
        residuals[score_slice],
        outputs[score_slice],
        sample_rate=SAMPLE_RATE,
        window_samples=SCORING_WINDOW_SAMPLES,
    )
    audio_seconds = reference.size / SAMPLE_RATE
    return {
        "protocol_version": PROTOCOL_VERSION,
        "public_scene_id": SCENE_ID,
        "is_final_test_result": False,
        "entry_point": entry_point,
        "device": device,
        "sample_rate": SAMPLE_RATE,
        "execution_mode": "strict_sample_by_sample",
        "initialization_acoustic_score_excluded": True,
        "model_reset_between_initialization_and_scoring": False,
        "secondary_path_reset_between_initialization_and_scoring": False,
        "runtime_includes_initialization": True,
        "peak_macs_scope_includes_initialization": True,
        "requires_error": bool(model.requires_error),
        "feedback_error_delay_samples": (
            1 if model.requires_error else None
        ),
        "start_seconds": float(start_seconds),
        "initialization_seconds": INITIALIZATION_SECONDS,
        "initialization_samples": INITIALIZATION_SAMPLES,
        "scoring_seconds": SCORING_SECONDS,
        "scored_samples": SCORING_SAMPLES,
        "scoring_window_seconds": SCORING_WINDOW_SECONDS,
        "scoring_window_samples": SCORING_WINDOW_SAMPLES,
        "scoring_window_count": SCORING_WINDOW_COUNT,
        "processed_samples": TOTAL_SAMPLES,
        "runtime_seconds": float(runtime_seconds),
        "audio_seconds": float(audio_seconds),
        "real_time_factor": float(
            runtime_seconds / max(audio_seconds, 1e-12)
        ),
        "samples_per_second": float(
            reference.size / max(runtime_seconds, 1e-12)
        ),
        **metrics,
        "controller_peak_abs": float(np.max(np.abs(outputs))),
        "full_run_controller_rms": float(
            np.sqrt(np.mean(outputs**2))
        ),
        "initialization_controller_peak_abs": float(
            np.max(np.abs(outputs[:INITIALIZATION_SAMPLES]))
        ),
        "initialization_residual_peak_abs": float(
            np.max(np.abs(residuals[:INITIALIZATION_SAMPLES]))
        ),
        "scoring_controller_peak_abs": float(
            np.max(np.abs(outputs[score_slice]))
        ),
        "scoring_residual_peak_abs": float(
            np.max(np.abs(residuals[score_slice]))
        ),
        **complexity,
    }


def _print_result(result: dict[str, Any]) -> None:
    model_type = "反馈式" if result["requires_error"] else "前馈式"
    print()
    print("公开演示场景评测完成")
    print(f"模型类型：{model_type}")
    print("初始化：0.5 秒（不计声学成绩），正式计分：3 秒")
    print("正式计分段划分为 6 个 0.5 秒窗口并等权平均")
    print(
        "未加权 1/3 倍频带平均降噪量（50 Hz～5 kHz）："
        f"{result['third_octave_average_nr_50_5000_db']:.4f} dB"
    )
    print(
        "平均窗口1/3倍频带反弹峰值（1 kHz～8 kHz）："
        f"{result['third_octave_rebound_peak_1000_8000_db']:.4f} dB"
    )
    print(f"参数量：{result['parameter_count']}")
    print(
        "峰值事件 MAC："
        f"{result['peak_macs_in_one_sample_event']:.0f}"
    )
    print(
        f"运行耗时：{result['runtime_seconds']:.3f} 秒，"
        f"实时因子：{result['real_time_factor']:.3f}"
    )
    print("说明：以上只是在固定公开场景上的结果，不是正式成绩。")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "在固定公开车载场景上严格逐采样运行参赛模型，"
            "并计算真实闭环降噪指标。"
        )
    )
    parser.add_argument(
        "--entry-point",
        required=True,
        help="模型入口，格式为 python.module:create_model。",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="模型运行设备，例如 cpu 或 cuda:0。",
    )
    parser.add_argument(
        "--start-seconds",
        type=_finite_nonnegative,
        default=0.0,
        help="从15秒公开片段的第几秒开始，默认0。",
    )
    parser.add_argument(
        "--output-limit",
        type=_finite_positive,
        default=100.0,
        help="控制输出绝对值安全上限，默认100。",
    )
    parser.add_argument(
        "--progress-every-seconds",
        type=_finite_positive,
        default=1.0,
        help="每处理多少秒音频显示一次进度，默认1秒。",
    )
    parser.add_argument(
        "--report",
        type=Path,
        help="可选：把完整标量结果保存为JSON文件。",
    )
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    try:
        result = evaluate(
            entry_point=args.entry_point,
            device=args.device,
            start_seconds=args.start_seconds,
            output_limit=args.output_limit,
            progress_every_seconds=args.progress_every_seconds,
        )
        _print_result(result)
        if args.report is not None:
            report_path = args.report.expanduser().resolve()
            report_path.parent.mkdir(parents=True, exist_ok=True)
            report_path.write_text(
                json.dumps(result, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            print(f"完整JSON报告：{report_path}")
    except Exception as exc:
        print(f"公开演示场景评测失败：{exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
