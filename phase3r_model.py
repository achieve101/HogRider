"""Streaming innovation-routed FIR experts for DEEPANC Phase 3R."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn


class InnovationRoutedFIRController(nn.Module):
    """Route fixed FIR experts by candidate-model innovation error."""

    requires_error = True
    sample_rate = 48_000

    def __init__(
        self,
        num_experts: int = 8,
        fir_length: int = 2048,
        path_length: int = 1967,
        n_fft: int = 4096,
        block_size: int = 240,
        ewma_lambda: float = 0.5,
        temperature: float = 0.15,
        alpha_update: float = 0.35,
        output_limit: float = 0.98,
        probe_rms: float = 0.0,
        probe_samples: int = 20_000,
        probe_seed: int = 2026,
    ) -> None:
        super().__init__()
        if num_experts <= 1 or min(fir_length, path_length, n_fft, block_size) <= 0:
            raise ValueError("Invalid Phase-3R dimensions.")
        if not 0 <= ewma_lambda < 1 or temperature <= 0 or not 0 < alpha_update <= 1:
            raise ValueError("Invalid routing hyperparameters.")
        if not 0 < output_limit < 1 or probe_rms < 0:
            raise ValueError("Invalid limiter/probe configuration.")
        self.num_experts = int(num_experts)
        self.fir_length = int(fir_length)
        self.path_length = int(path_length)
        self.n_fft = int(n_fft)
        self.block_size = int(block_size)
        self.ewma_lambda = float(ewma_lambda)
        self.temperature = float(temperature)
        self.alpha_update = float(alpha_update)
        self.output_limit = float(output_limit)
        self.probe_rms = float(probe_rms)
        self.probe_samples = int(probe_samples)
        self.probe_seed = int(probe_seed)
        bins = n_fft // 2 + 1
        self.register_buffer("expert_filters", torch.zeros(num_experts, fir_length))
        self.register_buffer("primary_real", torch.zeros(num_experts, bins))
        self.register_buffer("primary_imag", torch.zeros(num_experts, bins))
        self.register_buffer("secondary_paths", torch.zeros(num_experts, path_length))
        self.register_buffer("hann_window", torch.hann_window(n_fft, periodic=False))
        frequencies = torch.fft.rfftfreq(n_fft, 1.0 / self.sample_rate)
        self.register_buffer("band_mask", (frequencies >= 50.0) & (frequencies <= 8000.0))
        self._stream: dict[str, Any] | None = None

    @classmethod
    def from_artifacts(
        cls, oracle_checkpoint: str | Path, template_path: str | Path, **route_config: Any,
    ) -> "InnovationRoutedFIRController":
        checkpoint = torch.load(oracle_checkpoint, map_location="cpu", weights_only=False)
        state = checkpoint["model_state_dict"]
        experts = state["expert_filters"].detach().cpu()
        artifact = np.load(template_path, allow_pickle=False)
        secondary = torch.from_numpy(artifact["secondary_paths"])
        model = cls(
            num_experts=experts.shape[0], fir_length=experts.shape[1],
            path_length=secondary.shape[1], n_fft=artifact["hann_window"].shape[0],
            **route_config,
        )
        with torch.no_grad():
            model.expert_filters.copy_(experts)
            model.primary_real.copy_(torch.from_numpy(artifact["primary_real"]))
            model.primary_imag.copy_(torch.from_numpy(artifact["primary_imag"]))
            model.secondary_paths.copy_(secondary)
            model.hann_window.copy_(torch.from_numpy(artifact["hann_window"]))
            model.band_mask.copy_(torch.from_numpy(artifact["band_mask"]))
        return model

    @property
    def model_config(self) -> dict[str, Any]:
        return {key: getattr(self, key) for key in (
            "num_experts", "fir_length", "path_length", "n_fft", "block_size",
            "ewma_lambda", "temperature", "alpha_update", "output_limit",
            "probe_rms", "probe_samples", "probe_seed",
        )}

    def reset_streaming_state(self) -> None:
        experts = self.expert_filters.detach().cpu().numpy().astype(np.float64, copy=True)
        secondary = self.secondary_paths.detach().cpu().numpy().astype(np.float64, copy=True)
        primary = (
            self.primary_real.detach().cpu().numpy().astype(np.float64)
            + 1j * self.primary_imag.detach().cpu().numpy().astype(np.float64)
        )
        rng = np.random.default_rng(self.probe_seed)
        probe = self.probe_rms * rng.choice((-1.0, 1.0), size=self.probe_samples)
        self._stream = {
            "experts": experts,
            "secondary": secondary,
            "secondary_spectrum": np.fft.rfft(secondary, n=self.n_fft, axis=-1),
            "primary": primary,
            "window": self.hann_window.detach().cpu().numpy().astype(np.float64),
            "band": self.band_mask.detach().cpu().numpy().astype(bool),
            "alpha": np.full(self.num_experts, 1.0 / self.num_experts, dtype=np.float64),
            "effective_filter": experts.mean(axis=0),
            "ewma_log_j": None,
            "expert_ring": np.zeros(2 * self.fir_length, dtype=np.float64),
            "expert_pointer": 0,
            "path_output_history": np.zeros(self.path_length - 1, dtype=np.float64),
            "history_x": np.zeros(self.n_fft, dtype=np.float64),
            "history_e": np.zeros(self.n_fft, dtype=np.float64),
            "history_y": np.zeros(self.n_fft, dtype=np.float64),
            "history_a": np.zeros((self.num_experts, self.n_fft), dtype=np.float64),
            "history_pointer": 0,
            "history_count": 0,
            "completed_feedback": 0,
            "sample_index": 0,
            "pending_x": None,
            "pending_y": None,
            "pending_a": None,
            "block_x": [], "block_e": [], "block_y": [],
            "last_candidate_anti_block": None,
            "probe": probe,
            "trace": [],
        }

    def reset(self) -> None:
        self.reset_streaming_state()

    @staticmethod
    def _ordered(ring: np.ndarray, pointer: int, count: int) -> np.ndarray:
        if count < ring.shape[-1]:
            return ring[..., :count]
        return np.concatenate((ring[..., pointer:], ring[..., :pointer]), axis=-1)

    @staticmethod
    def _ring_append(ring: np.ndarray, pointer: int, values: np.ndarray) -> int:
        length = ring.shape[-1]
        count = values.shape[-1]
        first = min(count, length - pointer)
        ring[..., pointer:pointer + first] = values[..., :first]
        if first < count:
            ring[..., :count - first] = values[..., first:]
        return (pointer + count) % length

    def _append_completed_feedback(self, error: float) -> None:
        stream = self._stream
        assert stream is not None and stream["pending_x"] is not None
        stream["block_x"].append(stream["pending_x"])
        stream["block_e"].append(error)
        stream["block_y"].append(stream["pending_y"])
        if len(stream["block_e"]) == self.block_size:
            self._finalize_feedback_block()

    def _finalize_feedback_block(self) -> None:
        stream = self._stream
        assert stream is not None
        x = np.asarray(stream["block_x"], dtype=np.float64)
        e = np.asarray(stream["block_e"], dtype=np.float64)
        y = np.asarray(stream["block_y"], dtype=np.float64)
        overlap_input = np.concatenate((stream["path_output_history"], y))
        output_spectrum = np.fft.rfft(overlap_input, n=self.n_fft)
        convolution = np.fft.irfft(stream["secondary_spectrum"] * output_spectrum[None, :],
                                   n=self.n_fft, axis=-1)
        first = self.path_length - 1
        anti = convolution[:, first:first + self.block_size]
        stream["last_candidate_anti_block"] = anti.copy()
        stream["path_output_history"] = overlap_input[-(self.path_length - 1):].copy()
        pointer = int(stream["history_pointer"])
        self._ring_append(stream["history_x"], pointer, x)
        self._ring_append(stream["history_e"], pointer, e)
        self._ring_append(stream["history_y"], pointer, y)
        stream["history_pointer"] = self._ring_append(stream["history_a"], pointer, anti)
        stream["history_count"] = min(self.n_fft, int(stream["history_count"]) + self.block_size)
        stream["completed_feedback"] += self.block_size
        stream["block_x"].clear(); stream["block_e"].clear(); stream["block_y"].clear()
        self._update_route()

    def _update_route(self) -> None:
        stream = self._stream
        assert stream is not None
        if stream["history_count"] < self.n_fft:
            return
        pointer = int(stream["history_pointer"])
        x = self._ordered(stream["history_x"], pointer, self.n_fft)
        e = self._ordered(stream["history_e"], pointer, self.n_fft)
        anti = self._ordered(stream["history_a"], pointer, self.n_fft)
        window = stream["window"]
        band = stream["band"]
        spectrum_x = np.fft.rfft(x * window)
        reference_power = float(np.sum(np.abs(spectrum_x[band]) ** 2))
        if not np.isfinite(reference_power) or reference_power <= 1e-12:
            return
        disturbance = e[None, :] + anti
        spectrum_d = np.fft.rfft(disturbance * window[None, :], axis=-1)
        denominator = np.sum(np.abs(spectrum_d[:, band]) ** 2, axis=-1)
        prediction = stream["primary"] * spectrum_x[None, :]
        numerator = np.sum(np.abs(spectrum_d[:, band] - prediction[:, band]) ** 2, axis=-1)
        score = numerator / np.maximum(denominator, 1e-20)
        if not np.all(np.isfinite(score)) or np.any(denominator <= 1e-20):
            return
        log_j = np.log(score + 1e-12)
        previous = stream["ewma_log_j"]
        ewma = log_j if previous is None else self.ewma_lambda * previous + (1.0 - self.ewma_lambda) * log_j
        centered = ewma - np.min(ewma)
        logits = -centered / self.temperature
        proposal = np.exp(logits - np.max(logits))
        proposal /= np.sum(proposal)
        alpha = (1.0 - self.alpha_update) * stream["alpha"] + self.alpha_update * proposal
        if not np.all(np.isfinite(alpha)):
            return
        stream["ewma_log_j"] = ewma
        stream["alpha"] = alpha
        stream["effective_filter"] = alpha @ stream["experts"]
        ordered = np.argsort(score)
        stream["trace"].append({
            "completed_samples": int(stream["completed_feedback"]),
            "innovation": score.tolist(), "posterior": proposal.tolist(), "alpha": alpha.tolist(),
            "winner_zero_based": int(np.argmax(alpha)),
            "minimum_to_second_ratio": float(score[ordered[0]] / max(score[ordered[1]], 1e-20)),
        })

    def process_sample(self, reference_sample: float, previous_error_sample: float) -> float:
        if self._stream is None:
            self.reset_streaming_state()
        stream = self._stream
        assert stream is not None
        if stream["pending_x"] is not None:
            self._append_completed_feedback(float(previous_error_sample))

        pointer = (int(stream["expert_pointer"]) - 1) % self.fir_length
        stream["expert_pointer"] = pointer
        stream["expert_ring"][pointer] = float(reference_sample)
        stream["expert_ring"][pointer + self.fir_length] = float(reference_sample)
        history = stream["expert_ring"][pointer:pointer + self.fir_length]
        raw = float(np.dot(stream["effective_filter"], history))
        sample_index = int(stream["sample_index"])
        if sample_index < self.probe_samples:
            raw += float(stream["probe"][sample_index])
        safe_limit = self.output_limit - 1e-6
        output = float(safe_limit * np.tanh(raw / safe_limit))
        if not np.isfinite(output):
            raise FloatingPointError("InnovationRoutedFIRController produced NaN or Inf.")

        stream["pending_x"], stream["pending_y"] = float(reference_sample), output
        stream["sample_index"] = sample_index + 1
        return output

    def route_diagnostics(self) -> list[dict[str, Any]]:
        return [] if self._stream is None else list(self._stream["trace"])

    def current_alpha(self) -> np.ndarray:
        if self._stream is None:
            return np.full(self.num_experts, 1.0 / self.num_experts)
        return self._stream["alpha"].copy()

    def get_complexity(self) -> dict[str, int]:
        fft_macs = int((self.num_experts + 1) * 5 * self.n_fft * np.log2(self.n_fft))
        path_boundary_macs = fft_macs + self.num_experts * (self.n_fft // 2 + 1) * 6
        route_boundary_macs = fft_macs + self.num_experts * (self.n_fft // 2 + 1) * 8
        boundary_macs = path_boundary_macs + route_boundary_macs
        steady = self.fir_length
        return {
            "parameter_count": 0,
            "trainable_parameter_count": 0,
            "buffer_scalar_count": int(sum(value.numel() for value in self.buffers())),
            "steady_state_macs_per_sample": int(steady),
            "startup_macs": 0,
            "average_macs_per_sample": int(steady + boundary_macs / self.block_size),
            "peak_macs_in_one_sample_event": int(steady + boundary_macs),
        }
