"""Frozen-weight innovation-conditioned generative FIR controller."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn


CONFIDENCE_DIM = 4


class GenerativeInnovationFIRController(nn.Module):
    """Generate a bounded FIR residual from causal multi-model innovations.

    The learned tensors are never modified by :meth:`process_sample`.  The
    effective FIR is an inference activation cached in ``_stream`` and is
    cleared by :meth:`reset`.
    """

    requires_error = True
    sample_rate = 48_000

    def __init__(
        self,
        num_experts: int = 8,
        fir_length: int = 2048,
        path_length: int = 1967,
        n_fft: int = 4096,
        block_size: int = 240,
        hidden_size: int = 32,
        latent_size: int = 16,
        ewma_lambda: float = 0.8,
        temperature: float = 0.20,
        alpha_update: float = 0.20,
        latent_update: float = 0.20,
        output_limit: float = 0.98,
    ) -> None:
        super().__init__()
        if num_experts <= 1 or min(fir_length, path_length, n_fft, block_size) <= 0:
            raise ValueError("Invalid Phase-3G dimensions.")
        if hidden_size <= 0 or latent_size <= 0:
            raise ValueError("hidden_size and latent_size must be positive.")
        if not 0 <= ewma_lambda < 1 or temperature <= 0:
            raise ValueError("Invalid innovation smoothing configuration.")
        if not 0 < alpha_update <= 1 or not 0 < latent_update <= 1:
            raise ValueError("Update factors must lie in (0, 1].")
        if not 0 < output_limit < 1:
            raise ValueError("output_limit must lie in (0, 1).")

        self.num_experts = int(num_experts)
        self.fir_length = int(fir_length)
        self.path_length = int(path_length)
        self.n_fft = int(n_fft)
        self.block_size = int(block_size)
        self.hidden_size = int(hidden_size)
        self.latent_size = int(latent_size)
        self.ewma_lambda = float(ewma_lambda)
        self.temperature = float(temperature)
        self.alpha_update = float(alpha_update)
        self.latent_update = float(latent_update)
        self.output_limit = float(output_limit)
        self.feature_dim = 4 * self.num_experts + CONFIDENCE_DIM + self.latent_size

        self.register_buffer("expert_filters", torch.zeros(num_experts, fir_length))
        self.register_buffer("primary_real", torch.zeros(num_experts, n_fft // 2 + 1))
        self.register_buffer("primary_imag", torch.zeros(num_experts, n_fft // 2 + 1))
        self.register_buffer("secondary_paths", torch.zeros(num_experts, path_length))
        self.register_buffer("hann_window", torch.hann_window(n_fft, periodic=False))
        frequencies = torch.fft.rfftfreq(n_fft, 1.0 / self.sample_rate)
        self.register_buffer("band_mask", (frequencies >= 50.0) & (frequencies <= 8000.0))
        self.register_buffer("candidate_mask", torch.ones(num_experts, dtype=torch.bool))
        self.register_buffer("feature_mean", torch.zeros(self.feature_dim))
        self.register_buffer("feature_std", torch.ones(self.feature_dim))

        self.residual_dictionary = nn.Parameter(torch.zeros(latent_size, fir_length))
        self.gru = nn.GRUCell(self.feature_dim, hidden_size)
        self.latent_head = nn.Linear(hidden_size, latent_size)
        nn.init.zeros_(self.latent_head.weight)
        nn.init.zeros_(self.latent_head.bias)
        self._stream: dict[str, Any] | None = None

    @classmethod
    def from_artifacts(
        cls,
        oracle_checkpoint: str | Path,
        template_path: str | Path,
        *,
        initialize_dictionary: bool = True,
        seed: int = 2026,
        **config: Any,
    ) -> "GenerativeInnovationFIRController":
        checkpoint = torch.load(oracle_checkpoint, map_location="cpu", weights_only=False)
        experts = checkpoint["model_state_dict"]["expert_filters"].detach().cpu()
        with np.load(template_path, allow_pickle=False) as artifact:
            secondary = torch.from_numpy(artifact["secondary_paths"].copy())
            model = cls(
                num_experts=experts.shape[0], fir_length=experts.shape[1],
                path_length=secondary.shape[1], n_fft=artifact["hann_window"].shape[0],
                **config,
            )
            with torch.no_grad():
                model.expert_filters.copy_(experts)
                model.primary_real.copy_(torch.from_numpy(artifact["primary_real"].copy()))
                model.primary_imag.copy_(torch.from_numpy(artifact["primary_imag"].copy()))
                model.secondary_paths.copy_(secondary)
                model.hann_window.copy_(torch.from_numpy(artifact["hann_window"].copy()))
                model.band_mask.copy_(torch.from_numpy(artifact["band_mask"].copy()))
        if initialize_dictionary:
            model.initialize_dictionary(seed)
        return model

    @property
    def model_config(self) -> dict[str, Any]:
        keys = (
            "num_experts", "fir_length", "path_length", "n_fft", "block_size",
            "hidden_size", "latent_size", "ewma_lambda", "temperature",
            "alpha_update", "latent_update", "output_limit",
        )
        return {key: getattr(self, key) for key in keys}

    def initialize_dictionary(self, seed: int = 2026) -> None:
        """SVD initialise seven atoms and deterministic orthogonal residuals."""
        experts = self.expert_filters.detach().to(torch.float64)
        centered = experts - experts.mean(dim=0, keepdim=True)
        _, singular, vectors = torch.linalg.svd(centered, full_matrices=False)
        atoms = torch.zeros(
            self.latent_size, self.fir_length, dtype=torch.float64,
            device=experts.device,
        )
        svd_count = min(self.latent_size, self.num_experts - 1, vectors.shape[0])
        if svd_count:
            coefficient_scale = torch.clamp(singular[:svd_count], min=1e-8)
            coefficient_scale = coefficient_scale / max(1.0, float(np.sqrt(self.num_experts - 1)))
            atoms[:svd_count] = vectors[:svd_count] * coefficient_scale[:, None]
        if svd_count < self.latent_size:
            generator = torch.Generator(device="cpu").manual_seed(int(seed))
            random = torch.randn(
                self.latent_size - svd_count, self.fir_length,
                generator=generator, dtype=torch.float64,
            ).to(experts.device)
            existing = atoms[:svd_count]
            for row in range(random.shape[0]):
                value = random[row]
                if existing.numel():
                    coefficients = (value @ existing.T) / existing.square().sum(dim=1).clamp_min(1e-12)
                    value = value - coefficients @ existing
                value = value / value.norm().clamp_min(1e-12)
                scale = centered.square().mean().sqrt().clamp_min(1e-5) * np.sqrt(self.fir_length) * 0.05
                atoms[svd_count + row] = value * scale
                existing = atoms[:svd_count + row + 1]
        with torch.no_grad():
            self.residual_dictionary.copy_(atoms.to(self.residual_dictionary))

    def set_candidate_mask(self, mask: torch.Tensor | np.ndarray) -> None:
        value = torch.as_tensor(mask, dtype=torch.bool, device=self.candidate_mask.device)
        if value.shape != (self.num_experts,) or not bool(value.any()):
            raise ValueError("candidate mask must retain at least one expert.")
        self.candidate_mask.copy_(value)

    def set_feature_statistics(self, mean: torch.Tensor, std: torch.Tensor) -> None:
        if mean.shape != (self.feature_dim,) or std.shape != (self.feature_dim,):
            raise ValueError(f"feature statistics must have shape [{self.feature_dim}].")
        if not torch.isfinite(mean).all() or not torch.isfinite(std).all() or torch.any(std <= 0):
            raise ValueError("feature statistics must be finite and positive.")
        self.feature_mean.copy_(mean.to(self.feature_mean))
        self.feature_std.copy_(std.to(self.feature_std))

    def reset_streaming_state(self) -> None:
        experts = self.expert_filters.detach().cpu().numpy().astype(np.float64, copy=True)
        dictionary = self.residual_dictionary.detach().cpu().numpy().astype(np.float64, copy=True)
        secondary = self.secondary_paths.detach().cpu().numpy().astype(np.float64, copy=True)
        primary = (
            self.primary_real.detach().cpu().numpy().astype(np.float64)
            + 1j * self.primary_imag.detach().cpu().numpy().astype(np.float64)
        )
        mask = self.candidate_mask.detach().cpu().numpy().astype(bool, copy=True)
        alpha = mask.astype(np.float64) / float(mask.sum())
        self._stream = {
            "experts": experts, "dictionary": dictionary, "secondary": secondary,
            "secondary_spectrum": np.fft.rfft(secondary, n=self.n_fft, axis=-1),
            "primary": primary,
            "window": self.hann_window.detach().cpu().numpy().astype(np.float64),
            "band": self.band_mask.detach().cpu().numpy().astype(bool), "mask": mask,
            "feature_mean": self.feature_mean.detach().cpu().numpy().astype(np.float64),
            "feature_std": self.feature_std.detach().cpu().numpy().astype(np.float64),
            "alpha": alpha, "z": np.zeros(self.latent_size, dtype=np.float64),
            "hidden": torch.zeros(1, self.hidden_size, dtype=torch.float32),
            "effective_filter": alpha @ experts,
            "ewma_log_j": None,
            "fir_ring": np.zeros(2 * self.fir_length, dtype=np.float64), "fir_pointer": 0,
            "path_output_history": np.zeros(self.path_length - 1, dtype=np.float64),
            "history_x": np.zeros(self.n_fft, dtype=np.float64),
            "history_e": np.zeros(self.n_fft, dtype=np.float64),
            "history_a": np.zeros((self.num_experts, self.n_fft), dtype=np.float64),
            "history_pointer": 0, "history_count": 0, "completed_feedback": 0,
            "pending_x": None, "pending_y": None, "block_x": [], "block_e": [], "block_y": [],
            "last_candidate_anti_block": None, "trace": [],
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
        length = ring.shape[-1]; count = values.shape[-1]
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
        stream = self._stream; assert stream is not None
        x = np.asarray(stream["block_x"], dtype=np.float64)
        e = np.asarray(stream["block_e"], dtype=np.float64)
        y = np.asarray(stream["block_y"], dtype=np.float64)
        overlap = np.concatenate((stream["path_output_history"], y))
        spectrum_y = np.fft.rfft(overlap, n=self.n_fft)
        convolution = np.fft.irfft(
            stream["secondary_spectrum"] * spectrum_y[None, :], n=self.n_fft, axis=-1,
        )
        first = self.path_length - 1
        anti = convolution[:, first:first + self.block_size]
        stream["last_candidate_anti_block"] = anti.copy()
        stream["path_output_history"] = overlap[-first:].copy() if first else overlap[:0]
        pointer = int(stream["history_pointer"])
        self._ring_append(stream["history_x"], pointer, x)
        self._ring_append(stream["history_e"], pointer, e)
        stream["history_pointer"] = self._ring_append(stream["history_a"], pointer, anti)
        stream["history_count"] = min(self.n_fft, stream["history_count"] + self.block_size)
        stream["completed_feedback"] += self.block_size
        stream["block_x"].clear(); stream["block_e"].clear(); stream["block_y"].clear()
        self._update_route_and_generator()

    def _update_route_and_generator(self) -> None:
        stream = self._stream; assert stream is not None
        if stream["history_count"] < self.n_fft:
            return
        pointer = int(stream["history_pointer"])
        x = self._ordered(stream["history_x"], pointer, self.n_fft)
        e = self._ordered(stream["history_e"], pointer, self.n_fft)
        anti = self._ordered(stream["history_a"], pointer, self.n_fft)
        band, window = stream["band"], stream["window"]
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
        mask = stream["mask"]
        if not np.all(np.isfinite(score[mask])) or np.any(denominator[mask] <= 1e-20):
            return
        log_j = np.log(score + 1e-12)
        previous = stream["ewma_log_j"]
        ewma = log_j if previous is None else self.ewma_lambda * previous + (1-self.ewma_lambda) * log_j
        minimum = float(np.min(ewma[mask]))
        centered = ewma - minimum
        logits = -centered / self.temperature
        logits[~mask] = -np.inf
        proposal = np.zeros(self.num_experts, dtype=np.float64)
        visible = logits[mask]
        proposal[mask] = np.exp(visible - np.max(visible))
        proposal /= proposal.sum()
        alpha = (1-self.alpha_update) * stream["alpha"] + self.alpha_update * proposal
        alpha[~mask] = 0.0; alpha /= alpha.sum()
        ordered = np.sort(ewma[mask])
        confidence = np.asarray([
            -np.sum(proposal[mask] * np.log(proposal[mask] + 1e-12)) / max(np.log(mask.sum()), 1e-12),
            ordered[1] - ordered[0] if ordered.size > 1 else 0.0,
            np.log(reference_power + 1e-12), np.log(float(np.min(score[mask])) + 1e-12),
        ])
        raw_feature = np.concatenate((centered, proposal, alpha, confidence, mask.astype(float), stream["z"]))
        normalized = np.clip(
            (raw_feature - stream["feature_mean"]) / np.maximum(stream["feature_std"], 1e-4), -5.0, 5.0,
        )
        with torch.inference_mode():
            feature_t = torch.from_numpy(normalized.astype(np.float32)).unsqueeze(0)
            hidden = self.gru(feature_t, stream["hidden"])
            proposal_z = torch.tanh(self.latent_head(hidden)).squeeze(0).cpu().numpy().astype(np.float64)
        z = (1-self.latent_update) * stream["z"] + self.latent_update * proposal_z
        effective = alpha @ stream["experts"] + z @ stream["dictionary"]
        if not np.all(np.isfinite(effective)):
            return
        stream["ewma_log_j"], stream["alpha"], stream["hidden"], stream["z"] = ewma, alpha, hidden, z
        stream["effective_filter"] = effective
        order = np.argsort(np.where(mask, score, np.inf))
        stream["trace"].append({
            "completed_samples": int(stream["completed_feedback"]), "innovation": score.tolist(),
            "posterior": proposal.tolist(), "alpha": alpha.tolist(), "latent": z.tolist(),
            "winner_zero_based": int(np.argmax(alpha)),
            "minimum_to_second_ratio": float(score[order[0]] / max(score[order[1]], 1e-20)),
            "generated_residual_norm": float(np.linalg.norm(z @ stream["dictionary"])),
        })

    def process_sample(self, reference_sample: float, previous_error_sample: float) -> float:
        if self._stream is None:
            self.reset_streaming_state()
        stream = self._stream; assert stream is not None
        if stream["pending_x"] is not None:
            self._append_completed_feedback(float(previous_error_sample))
        pointer = (int(stream["fir_pointer"]) - 1) % self.fir_length
        stream["fir_pointer"] = pointer
        stream["fir_ring"][pointer] = float(reference_sample)
        stream["fir_ring"][pointer + self.fir_length] = float(reference_sample)
        history = stream["fir_ring"][pointer:pointer + self.fir_length]
        raw = float(np.dot(stream["effective_filter"], history))
        safe_limit = self.output_limit - 1e-6
        output = float(safe_limit * np.tanh(raw / safe_limit))
        if not np.isfinite(output):
            raise FloatingPointError("GenerativeInnovationFIRController produced NaN or Inf.")
        stream["pending_x"], stream["pending_y"] = float(reference_sample), output
        return output

    def route_diagnostics(self) -> list[dict[str, Any]]:
        return [] if self._stream is None else list(self._stream["trace"])

    def current_alpha(self) -> np.ndarray:
        if self._stream is None:
            mask = self.candidate_mask.detach().cpu().numpy().astype(float)
            return mask / mask.sum()
        return self._stream["alpha"].copy()

    def current_latent(self) -> np.ndarray:
        return np.zeros(self.latent_size) if self._stream is None else self._stream["z"].copy()

    def get_complexity(self) -> dict[str, int]:
        trainable = sum(value.numel() for value in self.parameters() if value.requires_grad)
        parameters = sum(value.numel() for value in self.parameters())
        generator_macs = self.feature_dim * self.hidden_size * 3 + self.hidden_size**2 * 3
        generator_macs += self.hidden_size * self.latent_size + self.latent_size * self.fir_length
        fft_macs = int((self.num_experts + 1) * 5 * self.n_fft * np.log2(self.n_fft))
        boundary = fft_macs + self.num_experts * (self.n_fft // 2 + 1) * 14 + generator_macs
        return {
            "parameter_count": int(parameters), "trainable_parameter_count": int(trainable),
            "buffer_scalar_count": int(sum(value.numel() for value in self.buffers())),
            "steady_state_macs_per_sample": self.fir_length,
            "startup_macs": 0,
            "average_macs_per_sample": int(self.fir_length + boundary / self.block_size),
            "peak_macs_in_one_sample_event": int(self.fir_length + boundary),
        }
