"""Differentiable acoustic metrics for DEEPANC Participant Kit protocol v6."""

from __future__ import annotations

from functools import lru_cache
from typing import Dict, Tuple

import torch
import torch.nn.functional as F


SAMPLE_RATE = 48_000
INITIALIZATION_SAMPLES = 24_000
SCORING_WINDOW_SAMPLES = 24_000
SCORING_WINDOW_COUNT = 6
SCORING_SAMPLES = SCORING_WINDOW_SAMPLES * SCORING_WINDOW_COUNT
TOTAL_SAMPLES = INITIALIZATION_SAMPLES + SCORING_SAMPLES
N_FFT = 8192
HOP_LENGTH = 2048
POWER_FLOOR = 1e-20

CENTER_FREQUENCIES = (
    12.5, 16.0, 20.0, 25.0, 31.5, 40.0, 50.0, 63.0, 80.0,
    100.0, 125.0, 160.0, 200.0, 250.0, 315.0, 400.0, 500.0,
    630.0, 800.0, 1000.0, 1250.0, 1600.0, 2000.0, 2500.0,
    3150.0, 4000.0, 5000.0, 6300.0, 8000.0, 10000.0, 12500.0,
    16000.0, 20000.0,
)


def _validate_signal(name: str, signal: torch.Tensor) -> torch.Tensor:
    if signal.ndim == 1:
        signal = signal.unsqueeze(0)
    if signal.ndim != 2:
        raise ValueError(f"{name} must have shape [batch, time].")
    if signal.shape[-1] != TOTAL_SAMPLES:
        raise ValueError(
            f"{name} must contain exactly {TOTAL_SAMPLES} samples, "
            f"received {signal.shape[-1]}."
        )
    if not signal.is_floating_point():
        raise TypeError(f"{name} must be a floating-point tensor.")
    return signal


@lru_cache(maxsize=32)
def _analysis_tensors(
    device_type: str,
    device_index: int | None,
    dtype: torch.dtype,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    device = torch.device(device_type, device_index)
    frequencies = torch.fft.rfftfreq(
        N_FFT, d=1.0 / SAMPLE_RATE, device=device, dtype=dtype,
    )
    centers = torch.tensor(CENTER_FREQUENCIES, device=device, dtype=dtype)
    edge_factor = 2.0 ** (1.0 / 6.0)
    masks = torch.stack([
        ((frequencies >= center / edge_factor)
         & (frequencies < center * edge_factor)).to(dtype)
        for center in centers
    ])
    primary_mask = (centers >= 50.0) & (centers <= 5000.0)
    rebound_mask = (centers >= 1000.0) & (centers <= 8000.0)
    window = torch.hann_window(
        N_FFT, periodic=True, device=device, dtype=dtype,
    )
    return masks, primary_mask, rebound_mask, window


def _windowed_band_power(signal: torch.Tensor) -> torch.Tensor:
    """Return band power with shape [batch, six_windows, bands]."""
    batch_size = signal.shape[0]
    scored = signal[:, INITIALIZATION_SAMPLES:]
    windows = scored.reshape(
        batch_size * SCORING_WINDOW_COUNT, SCORING_WINDOW_SAMPLES,
    )
    windows = windows - windows.mean(dim=-1, keepdim=True)

    padded_samples = max(SCORING_WINDOW_SAMPLES, N_FFT)
    remainder = (padded_samples - N_FFT) % HOP_LENGTH
    if remainder:
        padded_samples += HOP_LENGTH - remainder
    if padded_samples > SCORING_WINDOW_SAMPLES:
        windows = F.pad(
            windows, (0, padded_samples - SCORING_WINDOW_SAMPLES),
        )

    device = signal.device
    masks, _, _, window = _analysis_tensors(
        device.type, device.index, signal.dtype,
    )
    complex_spectrum = torch.stft(
        windows,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        win_length=N_FFT,
        window=window,
        center=False,
        normalized=False,
        onesided=True,
        return_complex=True,
    )
    # Avoid CUDA's complex abs JIT kernel.  The explicit real/imaginary form
    # is mathematically identical and works in minimal offline runtimes where
    # NVRTC is not installed.
    spectrum = (
        complex_spectrum.real.square() + complex_spectrum.imag.square()
    ).mean(dim=-1)
    spectrum = spectrum / window.square().sum()

    one_sided_scale = torch.ones(
        spectrum.shape[-1], device=device, dtype=signal.dtype,
    )
    if N_FFT % 2 == 0:
        one_sided_scale[1:-1] = 2.0
    else:
        one_sided_scale[1:] = 2.0
    spectrum = spectrum * one_sided_scale
    band_power = spectrum @ masks.transpose(0, 1)
    return band_power.reshape(
        batch_size, SCORING_WINDOW_COUNT, len(CENTER_FREQUENCIES),
    )


def compute_v6_metrics(
    disturbance: torch.Tensor,
    residual: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    """Compute differentiable official-style window and aggregate metrics."""
    disturbance = _validate_signal("disturbance", disturbance)
    residual = _validate_signal("residual", residual)
    if disturbance.shape != residual.shape:
        raise ValueError("disturbance and residual must have identical shapes.")
    if disturbance.device != residual.device or disturbance.dtype != residual.dtype:
        raise ValueError("disturbance and residual must share device and dtype.")

    target_power = _windowed_band_power(disturbance).detach()
    residual_power = _windowed_band_power(residual)
    target_power = target_power.clamp_min(POWER_FLOOR)
    residual_power = residual_power.clamp_min(POWER_FLOOR)
    change_db = 10.0 * torch.log10(residual_power / target_power)

    device = disturbance.device
    _, primary_mask, rebound_mask, _ = _analysis_tensors(
        device.type, device.index, disturbance.dtype,
    )
    primary_window_db = -change_db[..., primary_mask].mean(dim=-1)
    rebound_window_db = torch.relu(
        change_db[..., rebound_mask].amax(dim=-1)
    )
    return {
        "change_db": change_db,
        "primary_window_db": primary_window_db,
        "rebound_window_db": rebound_window_db,
        "primary_score_db": primary_window_db.mean(),
        "rebound_score_db": rebound_window_db.mean(),
    }


def compute_v6_loss(
    disturbance: torch.Tensor,
    residual: torch.Tensor,
    controller_output: torch.Tensor,
    primary_weight: float,
    rebound_weight: float,
    time_weight: float = 0.1,
    guard_weight: float = 1.0,
    guard_limit: float = 1.0,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Compute the Phase-1 score-aligned objective and detached diagnostics."""
    disturbance = _validate_signal("disturbance", disturbance)
    residual = _validate_signal("residual", residual)
    controller_output = _validate_signal("controller_output", controller_output)
    if not (disturbance.shape == residual.shape == controller_output.shape):
        raise ValueError("All signals must have identical shapes.")
    if guard_limit <= 0:
        raise ValueError("guard_limit must be positive.")

    metrics = compute_v6_metrics(disturbance, residual)
    primary_loss = -metrics["primary_score_db"] / 10.0
    rebound_loss = metrics["rebound_score_db"] / 10.0

    target_scored = disturbance[:, INITIALIZATION_SAMPLES:]
    residual_scored = residual[:, INITIALIZATION_SAMPLES:]
    time_loss = residual_scored.square().mean() / (
        target_scored.square().mean().detach() + 1e-12
    )
    violation = torch.relu(controller_output.abs() - guard_limit)
    guard_loss = violation.square().mean() + violation.square().amax()

    total_loss = (
        float(primary_weight) * primary_loss
        + float(rebound_weight) * rebound_loss
        + float(time_weight) * time_loss
        + float(guard_weight) * guard_loss
    )
    components = {
        "total_loss": total_loss.detach(),
        "primary_loss": primary_loss.detach(),
        "rebound_loss": rebound_loss.detach(),
        "time_loss": time_loss.detach(),
        "guard_loss": guard_loss.detach(),
        "primary_score_db": metrics["primary_score_db"].detach(),
        "rebound_score_db": metrics["rebound_score_db"].detach(),
        "controller_peak_abs": controller_output.detach().abs().amax(),
    }
    return total_loss, components
