"""赛题二公开演示场景使用的宽带与 1/3 倍频带评分实现。

本文件与正式评分协议使用相同的公开计算口径，但公开演示场景不是正式
测试集，因此这里得到的数值不等于最终比赛成绩。
"""

from __future__ import annotations

import math
from typing import Any, Dict

import numpy as np
import torch


_CENTER_FREQUENCIES = np.array(
    [
        12.5,
        16.0,
        20.0,
        25.0,
        31.5,
        40.0,
        50.0,
        63.0,
        80.0,
        100.0,
        125.0,
        160.0,
        200.0,
        250.0,
        315.0,
        400.0,
        500.0,
        630.0,
        800.0,
        1000.0,
        1250.0,
        1600.0,
        2000.0,
        2500.0,
        3150.0,
        4000.0,
        5000.0,
        6300.0,
        8000.0,
        10000.0,
        12500.0,
        16000.0,
        20000.0,
    ],
    dtype=np.float64,
)


def _rms(signal: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(signal, dtype=np.float64))))


def _power_spectrum(
    signal: np.ndarray,
    sample_rate: int,
    n_fft: int,
    hop_length: int,
) -> tuple[np.ndarray, np.ndarray]:
    waveform = torch.from_numpy(
        np.asarray(signal, dtype=np.float64).reshape(1, -1)
    )
    waveform = waveform - waveform.mean(dim=-1, keepdim=True)
    signal_samples = int(waveform.shape[-1])
    padded_samples = max(signal_samples, n_fft)
    remainder = (padded_samples - n_fft) % hop_length
    if remainder:
        padded_samples += hop_length - remainder
    if padded_samples > signal_samples:
        waveform = torch.nn.functional.pad(
            waveform,
            (0, padded_samples - signal_samples),
        )

    window = torch.hann_window(
        n_fft,
        periodic=True,
        dtype=waveform.dtype,
        device=waveform.device,
    )
    stft_result = torch.stft(
        waveform,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=n_fft,
        window=window,
        center=False,
        normalized=False,
        onesided=True,
        return_complex=True,
    )
    mean_power = stft_result.abs().square().mean(dim=(0, 2))
    mean_power = mean_power / window.square().sum()
    if n_fft % 2 == 0:
        if mean_power.numel() > 2:
            mean_power[1:-1] *= 2.0
    elif mean_power.numel() > 1:
        mean_power[1:] *= 2.0

    frequencies = torch.fft.rfftfreq(
        n_fft,
        d=1.0 / sample_rate,
        device=waveform.device,
    )
    return (
        frequencies.cpu().numpy(),
        mean_power.cpu().numpy(),
    )


def _third_octave(
    frequencies: np.ndarray,
    power_spectrum: np.ndarray,
    sample_rate: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    edge_factor = 2.0 ** (1.0 / 6.0)
    lower_frequencies = _CENTER_FREQUENCIES / edge_factor
    upper_frequencies = _CENTER_FREQUENCIES * edge_factor
    band_power = np.full(
        _CENTER_FREQUENCIES.shape,
        np.nan,
        dtype=np.float64,
    )
    valid_mask = upper_frequencies <= sample_rate / 2.0

    for index, (lower, upper) in enumerate(
        zip(lower_frequencies, upper_frequencies)
    ):
        if not valid_mask[index]:
            continue
        frequency_mask = (
            (frequencies >= lower)
            & (frequencies < upper)
        )
        if np.any(frequency_mask):
            band_power[index] = np.sum(
                power_spectrum[frequency_mask]
            )

    valid_mask &= np.isfinite(band_power)
    band_level_db = np.full_like(band_power, np.nan)
    band_level_db[valid_mask] = 10.0 * np.log10(
        np.maximum(band_power[valid_mask], 1e-30)
    )
    return band_power, band_level_db, valid_mask


def _official_style_metrics(
    off_power: np.ndarray,
    off_level: np.ndarray,
    off_valid: np.ndarray,
    on_power: np.ndarray,
    on_level: np.ndarray,
    on_valid: np.ndarray,
) -> Dict[str, float]:
    valid = (
        off_valid
        & on_valid
        & np.isfinite(off_level)
        & np.isfinite(on_level)
    )
    noise_reduction = off_level - on_level
    average_mask = (
        (_CENTER_FREQUENCIES >= 50.0)
        & (_CENTER_FREQUENCIES <= 5000.0)
        & valid
    )
    rebound_mask = (
        (_CENTER_FREQUENCIES >= 1000.0)
        & (_CENTER_FREQUENCIES <= 8000.0)
        & valid
    )
    if not np.any(average_mask):
        raise ValueError("50 Hz～5 kHz 内没有有效的 1/3 倍频带数据。")
    if not np.any(rebound_mask):
        raise ValueError("1 kHz～8 kHz 内没有有效的 1/3 倍频带数据。")

    average_nr = float(np.mean(noise_reduction[average_mask]))
    total_band_nr = float(
        10.0
        * np.log10(
            max(float(np.sum(off_power[average_mask])), 1e-30)
            / max(float(np.sum(on_power[average_mask])), 1e-30)
        )
    )

    rebound_frequencies = _CENTER_FREQUENCIES[rebound_mask]
    rebound_reductions = noise_reduction[rebound_mask]
    worst_index = int(np.argmin(rebound_reductions))
    worst_nr = float(rebound_reductions[worst_index])
    third_octave_level_changes = -rebound_reductions
    return {
        "third_octave_average_noise_reduction_db": average_nr,
        # 兼容既有未加权字段名。
        "average_noise_reduction_db": average_nr,
        "total_band_noise_reduction_db": total_band_nr,
        "third_octave_rebound_peak_db": max(0.0, -worst_nr),
        "mean_third_octave_level_change_db": float(
            np.mean(third_octave_level_changes)
        ),
        "mean_positive_third_octave_rebound_db": float(
            np.mean(
                np.maximum(third_octave_level_changes, 0.0)
            )
        ),
        "worst_third_octave_noise_reduction_db": worst_nr,
        "third_octave_rebound_center_frequency_hz": float(
            rebound_frequencies[worst_index]
        ),
    }


def score_signals(
    disturbance: np.ndarray,
    residual: np.ndarray,
    controller_output: np.ndarray,
    sample_rate: int,
    n_fft: int = 8192,
    hop_length: int = 2048,
) -> Dict[str, float]:
    """仅返回标量成绩，不向参赛模型暴露私有时域信号。"""
    disturbance = np.asarray(
        disturbance, dtype=np.float64
    ).reshape(-1)
    residual = np.asarray(residual, dtype=np.float64).reshape(-1)
    controller_output = np.asarray(
        controller_output, dtype=np.float64
    ).reshape(-1)
    if not (
        disturbance.size
        == residual.size
        == controller_output.size
        and disturbance.size > 0
    ):
        raise ValueError("评分信号必须是长度相同的非空一维数组。")

    eps = 1e-20
    disturbance_power = float(np.sum(disturbance**2)) + eps
    residual_power = float(np.sum(residual**2)) + eps
    broadband_nr = 10.0 * math.log10(
        disturbance_power / residual_power
    )

    frequencies, off_spectrum = _power_spectrum(
        disturbance,
        sample_rate,
        n_fft,
        hop_length,
    )
    _, on_spectrum = _power_spectrum(
        residual,
        sample_rate,
        n_fft,
        hop_length,
    )
    off_power, off_level, off_valid = _third_octave(
        frequencies,
        off_spectrum,
        sample_rate,
    )
    on_power, on_level, on_valid = _third_octave(
        frequencies,
        on_spectrum,
        sample_rate,
    )
    octave_metrics = _official_style_metrics(
        off_power,
        off_level,
        off_valid,
        on_power,
        on_level,
        on_valid,
    )

    disturbance_mean_power = float(np.mean(disturbance**2)) + eps
    control_mean_power = float(np.mean(controller_output**2)) + eps
    return {
        "primary_score_db": float(
            octave_metrics[
                "third_octave_average_noise_reduction_db"
            ]
        ),
        "third_octave_average_nr_50_5000_db": float(
            octave_metrics[
                "third_octave_average_noise_reduction_db"
            ]
        ),
        # 兼容既有未加权字段名。
        "average_nr_50_5000_db": float(
            octave_metrics[
                "third_octave_average_noise_reduction_db"
            ]
        ),
        "broadband_nr_db": float(broadband_nr),
        "total_band_nr_50_5000_db": float(
            octave_metrics["total_band_noise_reduction_db"]
        ),
        "third_octave_rebound_peak_1000_8000_db": float(
            octave_metrics["third_octave_rebound_peak_db"]
        ),
        # 兼容既有通用字段名。
        "maximum_rebound_1000_8000_db": float(
            octave_metrics["third_octave_rebound_peak_db"]
        ),
        "mean_third_octave_level_change_1000_8000_db": float(
            octave_metrics[
                "mean_third_octave_level_change_db"
            ]
        ),
        "mean_positive_third_octave_rebound_1000_8000_db": float(
            octave_metrics[
                "mean_positive_third_octave_rebound_db"
            ]
        ),
        "worst_third_octave_nr_1000_8000_db": float(
            octave_metrics["worst_third_octave_noise_reduction_db"]
        ),
        "third_octave_rebound_center_frequency_hz": float(
            octave_metrics[
                "third_octave_rebound_center_frequency_hz"
            ]
        ),
        "disturbance_rms": _rms(disturbance),
        "residual_rms": _rms(residual),
        "controller_rms": _rms(controller_output),
        "controller_peak_abs": float(
            np.max(np.abs(controller_output))
        ),
        "control_to_disturbance_db": float(
            10.0
            * math.log10(
                control_mean_power / disturbance_mean_power
            )
        ),
    }


_WINDOW_AVERAGE_KEYS = (
    "primary_score_db",
    "third_octave_average_nr_50_5000_db",
    "average_nr_50_5000_db",
    "broadband_nr_db",
    "total_band_nr_50_5000_db",
    "third_octave_rebound_peak_1000_8000_db",
    "maximum_rebound_1000_8000_db",
    "mean_third_octave_level_change_1000_8000_db",
    "mean_positive_third_octave_rebound_1000_8000_db",
)


def score_windowed_signals(
    disturbance: np.ndarray,
    residual: np.ndarray,
    controller_output: np.ndarray,
    sample_rate: int,
    window_samples: int,
    n_fft: int = 8192,
    hop_length: int = 2048,
) -> Dict[str, Any]:
    """逐窗口评分，并对各窗口的 dB 指标做等权算术平均。"""
    disturbance = np.asarray(
        disturbance, dtype=np.float64
    ).reshape(-1)
    residual = np.asarray(residual, dtype=np.float64).reshape(-1)
    controller_output = np.asarray(
        controller_output, dtype=np.float64
    ).reshape(-1)
    if not (
        disturbance.size
        == residual.size
        == controller_output.size
        and disturbance.size > 0
    ):
        raise ValueError("评分信号必须是长度相同的非空一维数组。")
    if not isinstance(window_samples, int) or window_samples <= 0:
        raise ValueError("window_samples 必须是正整数。")
    if disturbance.size % window_samples != 0:
        raise ValueError("评分信号长度必须是窗口长度的整数倍。")

    window_count = disturbance.size // window_samples
    window_results: list[Dict[str, Any]] = []
    for window_index in range(window_count):
        start = window_index * window_samples
        end = start + window_samples
        metrics = score_signals(
            disturbance[start:end],
            residual[start:end],
            controller_output[start:end],
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
        )
        window_results.append(
            {
                "window_index": window_index,
                "start_sample_in_scored_segment": start,
                "end_sample_in_scored_segment": end,
                **metrics,
            }
        )

    result: Dict[str, Any] = {
        key: float(np.mean([item[key] for item in window_results]))
        for key in _WINDOW_AVERAGE_KEYS
    }
    largest_rebound_window = max(
        window_results,
        key=lambda item: item[
            "third_octave_rebound_peak_1000_8000_db"
        ],
    )
    eps = 1e-20
    result.update(
        {
            "scoring_window_count": int(window_count),
            "scoring_window_samples": int(window_samples),
            "window_aggregation": "equal_arithmetic_mean_of_window_db_metrics",
            "worst_third_octave_nr_1000_8000_db": float(
                min(
                    item["worst_third_octave_nr_1000_8000_db"]
                    for item in window_results
                )
            ),
            "third_octave_rebound_center_frequency_hz": float(
                largest_rebound_window[
                    "third_octave_rebound_center_frequency_hz"
                ]
            ),
            "largest_single_window_rebound_1000_8000_db": float(
                largest_rebound_window[
                    "third_octave_rebound_peak_1000_8000_db"
                ]
            ),
            "largest_single_window_rebound_index": int(
                largest_rebound_window["window_index"]
            ),
            "disturbance_rms": _rms(disturbance),
            "residual_rms": _rms(residual),
            "controller_rms": _rms(controller_output),
            "controller_peak_abs": float(
                np.max(np.abs(controller_output))
            ),
            "control_to_disturbance_db": float(
                10.0
                * math.log10(
                    (float(np.mean(controller_output**2)) + eps)
                    / (float(np.mean(disturbance**2)) + eps)
                )
            ),
            "window_results": window_results,
        }
    )
    return result
