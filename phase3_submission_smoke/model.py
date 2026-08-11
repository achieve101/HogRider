"""Streaming feedback FIR-expert controller for DEEPANC Phase 3."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


FEATURE_DIM = 10
X_ERROR_LAGS = (0, 8, 32)
Y_ERROR_LAGS = (0, 4, 8, 16)
MAX_FEATURE_LAG = 32


def _lagged_block(block: torch.Tensor, context: torch.Tensor, lag: int) -> torch.Tensor:
    if lag == 0:
        return block
    joined = torch.cat((context, block), dim=-1)
    start = context.shape[-1] - lag
    return joined[:, start:start + block.shape[-1]]


def extract_feedback_features(
    reference_block: torch.Tensor,
    error_block: torch.Tensor,
    output_block: torch.Tensor,
    reference_context: torch.Tensor,
    output_context: torch.Tensor,
    eps: float = 1e-8,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return ten causal block statistics and updated 32-sample contexts."""
    tensors = (reference_block, error_block, output_block)
    if any(value.ndim != 2 for value in tensors):
        raise ValueError("reference, error, and output blocks must have shape [B,T].")
    if not (reference_block.shape == error_block.shape == output_block.shape):
        raise ValueError("reference, error, and output blocks must have equal shape.")
    batch = reference_block.shape[0]
    expected_context = (batch, MAX_FEATURE_LAG)
    if reference_context.shape != expected_context or output_context.shape != expected_context:
        raise ValueError(f"feature contexts must have shape {expected_context}.")

    def rms(value: torch.Tensor) -> torch.Tensor:
        return value.square().mean(dim=-1).add(eps).sqrt()

    x_rms, e_rms, y_rms = (rms(value) for value in tensors)
    features = [torch.log10(x_rms), torch.log10(e_rms), torch.log10(y_rms)]
    for lag in X_ERROR_LAGS:
        lagged = _lagged_block(reference_block, reference_context, lag)
        denominator = rms(lagged) * e_rms + eps
        features.append((lagged * error_block).mean(dim=-1) / denominator)
    for lag in Y_ERROR_LAGS:
        lagged = _lagged_block(output_block, output_context, lag)
        denominator = rms(lagged) * e_rms + eps
        features.append((lagged * error_block).mean(dim=-1) / denominator)

    next_x_context = torch.cat((reference_context, reference_block), dim=-1)[:, -MAX_FEATURE_LAG:]
    next_y_context = torch.cat((output_context, output_block), dim=-1)[:, -MAX_FEATURE_LAG:]
    return torch.stack(features, dim=-1), next_x_context, next_y_context


@dataclass
class FeedbackState:
    hidden: torch.Tensor
    alpha: torch.Tensor
    reference_context: torch.Tensor
    output_context: torch.Tensor


class FeedbackFIRController(nn.Module):
    """Eight causal FIR experts selected by delayed-error block feedback."""

    requires_error = True
    sample_rate = 48_000

    def __init__(
        self,
        num_experts: int = 8,
        fir_length: int = 2048,
        hidden_size: int = 24,
        block_size: int = 240,
        alpha_update: float = 0.2,
        output_limit: float = 0.98,
    ) -> None:
        super().__init__()
        if num_experts <= 0 or fir_length <= 0 or hidden_size <= 0 or block_size <= 0:
            raise ValueError("expert, FIR, hidden, and block sizes must be positive.")
        if not 0.0 < alpha_update <= 1.0 or not 0.0 < output_limit < 1.0:
            raise ValueError("alpha_update and output_limit must lie in (0,1].")
        self.num_experts = int(num_experts)
        self.fir_length = int(fir_length)
        self.hidden_size = int(hidden_size)
        self.block_size = int(block_size)
        self.alpha_update = float(alpha_update)
        self.output_limit = float(output_limit)

        self.expert_filters = nn.Parameter(torch.zeros(num_experts, fir_length))
        self.gru = nn.GRUCell(FEATURE_DIM, hidden_size)
        self.route_head = nn.Linear(hidden_size, num_experts)
        nn.init.zeros_(self.route_head.weight)
        nn.init.zeros_(self.route_head.bias)
        self.register_buffer("feature_mean", torch.zeros(FEATURE_DIM))
        self.register_buffer("feature_std", torch.ones(FEATURE_DIM))
        self._stream: dict[str, Any] | None = None

    def initial_feedback_state(
        self, batch_size: int, device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> FeedbackState:
        device = device or self.expert_filters.device
        dtype = dtype or self.expert_filters.dtype
        return FeedbackState(
            hidden=torch.zeros(batch_size, self.hidden_size, device=device, dtype=dtype),
            alpha=torch.full(
                (batch_size, self.num_experts), 1.0 / self.num_experts,
                device=device, dtype=dtype,
            ),
            reference_context=torch.zeros(batch_size, MAX_FEATURE_LAG, device=device, dtype=dtype),
            output_context=torch.zeros(batch_size, MAX_FEATURE_LAG, device=device, dtype=dtype),
        )

    def normalize_features(self, raw: torch.Tensor) -> torch.Tensor:
        return ((raw - self.feature_mean) / self.feature_std.clamp_min(1e-4)).clamp(-5.0, 5.0)

    def update_feedback(
        self, raw_features: torch.Tensor, state: FeedbackState,
    ) -> tuple[FeedbackState, torch.Tensor, torch.Tensor]:
        normalized = self.normalize_features(raw_features)
        hidden = self.gru(normalized, state.hidden)
        logits = self.route_head(hidden)
        proposal = torch.softmax(logits, dim=-1)
        alpha = (1.0 - self.alpha_update) * state.alpha + self.alpha_update * proposal
        return FeedbackState(
            hidden=hidden, alpha=alpha,
            reference_context=state.reference_context,
            output_context=state.output_context,
        ), logits, proposal

    def causal_expert_outputs(self, reference: torch.Tensor) -> torch.Tensor:
        """Return [B,M,T] outputs with expert filters stored in lag order."""
        if reference.ndim != 2:
            raise ValueError("reference must have shape [B,T].")
        padded = F.pad(reference.unsqueeze(1), (self.fir_length - 1, 0))
        weights = torch.flip(self.expert_filters, dims=(-1,)).unsqueeze(1)
        return F.conv1d(padded, weights)

    def oracle_forward(self, reference: torch.Tensor, expert_index: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        expert_outputs = self.causal_expert_outputs(reference)
        if expert_index.shape != (reference.shape[0],):
            raise ValueError("expert_index must have shape [B].")
        selected = expert_outputs[torch.arange(reference.shape[0], device=reference.device), expert_index]
        return self.soft_limit(selected), selected

    def soft_limit(self, raw_output: torch.Tensor) -> torch.Tensor:
        safe_limit = self.output_limit - 1e-6
        return safe_limit * torch.tanh(raw_output / safe_limit)

    def set_feature_statistics(self, mean: torch.Tensor, std: torch.Tensor) -> None:
        if mean.shape != (FEATURE_DIM,) or std.shape != (FEATURE_DIM,):
            raise ValueError(f"feature statistics must have shape [{FEATURE_DIM}].")
        if not torch.isfinite(mean).all() or not torch.isfinite(std).all() or torch.any(std <= 0):
            raise ValueError("feature statistics must be finite with positive standard deviations.")
        self.feature_mean.copy_(mean.to(self.feature_mean))
        self.feature_std.copy_(std.to(self.feature_std))

    def reset_streaming_state(self) -> None:
        """Reset every cross-recording state used by process_sample()."""
        filters = self.expert_filters.detach().cpu().numpy().astype(np.float64, copy=True)
        state = self.initial_feedback_state(1, device=self.expert_filters.device, dtype=self.expert_filters.dtype)
        self._stream = {
            "filters": filters,
            "effective_filter": np.mean(filters, axis=0),
            "ring": np.zeros(2 * self.fir_length, dtype=np.float64),
            "pointer": 0,
            "state": state,
            "x_block": [], "e_block": [], "y_block": [],
            "context_x": np.zeros(MAX_FEATURE_LAG, dtype=np.float32),
            "context_y": np.zeros(MAX_FEATURE_LAG, dtype=np.float32),
            "pending_x": None, "pending_y": None,
        }

    def reset(self) -> None:
        self.reset_streaming_state()

    def _finalize_stream_block(self) -> None:
        stream = self._stream
        assert stream is not None
        device, dtype = self.expert_filters.device, self.expert_filters.dtype
        x = torch.tensor([stream["x_block"]], device=device, dtype=dtype)
        e = torch.tensor([stream["e_block"]], device=device, dtype=dtype)
        y = torch.tensor([stream["y_block"]], device=device, dtype=dtype)
        cx = torch.tensor(stream["context_x"], device=device, dtype=dtype).unsqueeze(0)
        cy = torch.tensor(stream["context_y"], device=device, dtype=dtype).unsqueeze(0)
        raw, next_cx, next_cy = extract_feedback_features(x, e, y, cx, cy)
        state: FeedbackState = stream["state"]
        state.reference_context = next_cx
        state.output_context = next_cy
        state, _, _ = self.update_feedback(raw, state)
        stream["state"] = state
        stream["context_x"] = next_cx[0].detach().cpu().numpy()
        stream["context_y"] = next_cy[0].detach().cpu().numpy()
        alpha = state.alpha[0].detach().cpu().numpy().astype(np.float64)
        stream["effective_filter"] = alpha @ stream["filters"]
        stream["x_block"].clear(); stream["e_block"].clear(); stream["y_block"].clear()

    def process_sample(self, reference_sample: float, previous_error_sample: float) -> float:
        if self._stream is None:
            self.reset_streaming_state()
        stream = self._stream
        assert stream is not None
        if stream["pending_x"] is not None:
            stream["x_block"].append(float(stream["pending_x"]))
            stream["y_block"].append(float(stream["pending_y"]))
            stream["e_block"].append(float(previous_error_sample))
            if len(stream["e_block"]) == self.block_size:
                self._finalize_stream_block()

        pointer = (int(stream["pointer"]) - 1) % self.fir_length
        stream["pointer"] = pointer
        stream["ring"][pointer] = float(reference_sample)
        stream["ring"][pointer + self.fir_length] = float(reference_sample)
        history = stream["ring"][pointer:pointer + self.fir_length]
        raw = float(np.dot(stream["effective_filter"], history))
        safe_limit = self.output_limit - 1e-6
        output = float(safe_limit * np.tanh(raw / safe_limit))
        if not np.isfinite(output):
            raise FloatingPointError("FeedbackFIRController produced NaN or Inf.")
        stream["pending_x"], stream["pending_y"] = float(reference_sample), output
        return output

    def get_complexity(self) -> dict[str, int]:
        parameter_count = sum(parameter.numel() for parameter in self.parameters())
        correlation_macs = len(X_ERROR_LAGS) + len(Y_ERROR_LAGS)
        steady = self.fir_length + correlation_macs
        gru_macs = 3 * self.hidden_size * (FEATURE_DIM + self.hidden_size)
        route_macs = self.hidden_size * self.num_experts
        filter_mix_macs = self.num_experts * self.fir_length
        peak = steady + gru_macs + route_macs + filter_mix_macs
        return {
            "parameter_count": int(parameter_count),
            "steady_state_macs_per_sample": int(steady),
            "startup_macs": 0,
            "peak_macs_in_one_sample_event": int(peak),
        }
