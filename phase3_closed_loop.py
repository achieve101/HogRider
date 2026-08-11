"""Differentiable blockwise closed-loop simulation for Phase 3 feedback models."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from phase3_model import FeedbackFIRController, FeedbackState, extract_feedback_features
from v6_metrics import INITIALIZATION_SAMPLES, compute_v6_metrics


@dataclass
class ClosedLoopResult:
    output: torch.Tensor
    raw_output: torch.Tensor
    residual: torch.Tensor
    route_logits: torch.Tensor
    alpha: torch.Tensor
    raw_features: torch.Tensor
    route_accuracy_after_initialization: torch.Tensor


def _path_block(
    output_block: torch.Tensor,
    output_history: torch.Tensor,
    path: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply a possibly batch-specific path to a block and preserve causal history."""
    batch, block = output_block.shape
    path_length = path.shape[-1]
    joined = torch.cat((output_history, output_block), dim=-1)
    signal = joined.reshape(1, batch, joined.shape[-1])
    weights = torch.flip(path, dims=(-1,)).reshape(batch, 1, path_length)
    propagated = F.conv1d(signal, weights, groups=batch).squeeze(0)
    if propagated.shape != (batch, block):
        raise RuntimeError("secondary-path block convolution returned an unexpected shape.")
    return propagated, joined[:, -(path_length - 1):] if path_length > 1 else joined[:, :0]


def rollout_feedback_closed_loop(
    model: FeedbackFIRController,
    reference: torch.Tensor,
    disturbance: torch.Tensor,
    secondary_paths: torch.Tensor,
    path_slots: torch.Tensor,
    route_labels: torch.Tensor,
    *,
    truncate_blocks: int = 100,
) -> ClosedLoopResult:
    """Roll out exact one-block-delayed feedback over complete 3.5-second signals."""
    if reference.ndim != 2 or disturbance.shape != reference.shape:
        raise ValueError("reference and disturbance must have identical [B,T] shapes.")
    if secondary_paths.ndim != 3 or secondary_paths.shape[0] != reference.shape[0]:
        raise ValueError("secondary_paths must have shape [B,path_slots,L].")
    if reference.shape[-1] % model.block_size:
        raise ValueError("signal length must be divisible by model.block_size.")
    blocks = reference.shape[-1] // model.block_size
    expected = (reference.shape[0], blocks)
    if path_slots.shape != expected or route_labels.shape != expected:
        raise ValueError(f"path_slots and route_labels must have shape {expected}.")
    if torch.any(path_slots < 0) or torch.any(path_slots >= secondary_paths.shape[1]):
        raise ValueError("path_slots contains an unavailable secondary-path slot.")
    if torch.any(route_labels < 0) or torch.any(route_labels >= model.num_experts):
        raise ValueError("route_labels contains an unavailable expert index.")

    batch = reference.shape[0]
    expert_outputs = model.causal_expert_outputs(reference)
    state = model.initial_feedback_state(batch, reference.device, reference.dtype)
    path_history = torch.zeros(
        batch, secondary_paths.shape[-1] - 1,
        device=reference.device, dtype=reference.dtype,
    )
    outputs, raw_outputs, residuals = [], [], []
    logits_history, alpha_history, feature_history = [], [], []

    for block_index in range(blocks):
        start = block_index * model.block_size
        end = start + model.block_size
        alpha_used = state.alpha
        experts = expert_outputs[:, :, start:end]
        raw_block = (experts * alpha_used.unsqueeze(-1)).sum(dim=1)
        output_block = model.soft_limit(raw_block)
        batch_indices = torch.arange(batch, device=reference.device)
        current_path = secondary_paths[batch_indices, path_slots[:, block_index]]
        anti_noise, path_history = _path_block(output_block, path_history, current_path)
        residual_block = disturbance[:, start:end] - anti_noise
        raw_features, next_x_context, next_y_context = extract_feedback_features(
            reference[:, start:end], residual_block, output_block,
            state.reference_context, state.output_context,
        )
        state, logits, _ = model.update_feedback(raw_features, state)
        state.reference_context = next_x_context
        state.output_context = next_y_context
        outputs.append(output_block); raw_outputs.append(raw_block); residuals.append(residual_block)
        logits_history.append(logits); alpha_history.append(alpha_used); feature_history.append(raw_features)

        if truncate_blocks > 0 and (block_index + 1) % truncate_blocks == 0:
            state = FeedbackState(
                hidden=state.hidden.detach(), alpha=state.alpha.detach(),
                reference_context=state.reference_context.detach(),
                output_context=state.output_context.detach(),
            )
            path_history = path_history.detach()

    logits_tensor = torch.stack(logits_history, dim=1)
    scored_block = INITIALIZATION_SAMPLES // model.block_size
    accuracy_start = scored_block if blocks > scored_block else 0
    predicted = logits_tensor[:, accuracy_start:].argmax(dim=-1)
    accuracy = (predicted == route_labels[:, accuracy_start:]).to(reference.dtype).mean()
    return ClosedLoopResult(
        output=torch.cat(outputs, dim=-1), raw_output=torch.cat(raw_outputs, dim=-1),
        residual=torch.cat(residuals, dim=-1), route_logits=logits_tensor,
        alpha=torch.stack(alpha_history, dim=1), raw_features=torch.stack(feature_history, dim=1),
        route_accuracy_after_initialization=accuracy,
    )


def compute_phase3_loss(
    disturbance: torch.Tensor,
    rollout: ClosedLoopResult,
    route_labels: torch.Tensor,
    *,
    primary_weight: float = 0.7,
    rebound_weight: float = 0.3,
    time_weight: float = 0.1,
    guard_weight: float = 1.0,
    route_weight: float = 0.05,
    gate_delta_weight: float = 0.01,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    metrics = compute_v6_metrics(disturbance, rollout.residual)
    primary_loss = -metrics["primary_score_db"] / 10.0
    rebound_loss = metrics["rebound_score_db"] / 10.0
    scored = slice(INITIALIZATION_SAMPLES, None)
    time_loss = rollout.residual[:, scored].square().mean() / (
        disturbance[:, scored].square().mean().detach() + 1e-12
    )
    violation = torch.relu(rollout.raw_output.abs() - 0.9).square()
    guard_loss = violation.mean() + violation.amax()
    route_loss = F.cross_entropy(
        rollout.route_logits.reshape(-1, rollout.route_logits.shape[-1]),
        route_labels.reshape(-1),
    )
    gate_delta = (rollout.alpha[:, 1:] - rollout.alpha[:, :-1]).square().mean()
    total = (
        primary_weight * primary_loss + rebound_weight * rebound_loss
        + time_weight * time_loss + guard_weight * guard_loss
        + route_weight * route_loss + gate_delta_weight * gate_delta
    )
    return total, {
        "total_loss": total.detach(), "primary_loss": primary_loss.detach(),
        "rebound_loss": rebound_loss.detach(), "time_loss": time_loss.detach(),
        "guard_loss": guard_loss.detach(), "route_loss": route_loss.detach(),
        "gate_delta_loss": gate_delta.detach(),
        "primary_score_db": metrics["primary_score_db"].detach(),
        "rebound_score_db": metrics["rebound_score_db"].detach(),
        "controller_peak_abs": rollout.output.detach().abs().amax(),
        "raw_controller_peak_abs": rollout.raw_output.detach().abs().amax(),
        "route_accuracy_after_initialization": rollout.route_accuracy_after_initialization.detach(),
    }
