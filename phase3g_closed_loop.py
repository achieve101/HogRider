"""Differentiable causal closed loop for the Phase-3G generator."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from phase3g_model import GenerativeInnovationFIRController
from v6_metrics import INITIALIZATION_SAMPLES, compute_v6_metrics


@dataclass
class Phase3GRollout:
    output: torch.Tensor
    raw_output: torch.Tensor
    residual: torch.Tensor
    alpha: torch.Tensor
    latent: torch.Tensor
    raw_features: torch.Tensor
    kernel_loss: torch.Tensor
    kernel_delta_loss: torch.Tensor
    distill_loss: torch.Tensor


def causal_filter_bank(reference: torch.Tensor, filters: torch.Tensor) -> torch.Tensor:
    """Apply lag-ordered fixed filters and return ``[B,K,T]``."""
    if reference.ndim != 2 or filters.ndim != 2:
        raise ValueError("reference and filters must have shapes [B,T] and [K,L].")
    padded = F.pad(reference.unsqueeze(1), (filters.shape[-1]-1, 0))
    return F.conv1d(padded, torch.flip(filters, dims=(-1,)).unsqueeze(1))


def _multi_path_block(
    output: torch.Tensor,
    history: torch.Tensor,
    paths: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply ``[B,M,L]`` paths to a shared ``[B,T]`` output block."""
    batch, block = output.shape
    _, candidates, length = paths.shape
    joined = torch.cat((history, output[:, None, :].expand(-1, candidates, -1)), dim=-1)
    signal = joined.reshape(1, batch*candidates, joined.shape[-1])
    weights = torch.flip(paths, dims=(-1,)).reshape(batch*candidates, 1, length)
    propagated = F.conv1d(signal, weights, groups=batch*candidates).reshape(batch, candidates, block)
    next_history = joined[..., -(length-1):] if length > 1 else joined[..., :0]
    return propagated, next_history


def _physical_path_block(
    output: torch.Tensor,
    history: torch.Tensor,
    path: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    return tuple(value.squeeze(1) for value in _multi_path_block(
        output, history.unsqueeze(1), path.unsqueeze(1),
    ))  # type: ignore[return-value]


def _innovation_update(
    model: GenerativeInnovationFIRController,
    history_x: torch.Tensor,
    history_e: torch.Tensor,
    history_a: torch.Tensor,
    candidate_mask: torch.Tensor,
    alpha: torch.Tensor,
    ewma: torch.Tensor | None,
    hidden: torch.Tensor,
    latent: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute detached physical features and a differentiable GRU update."""
    batch = history_x.shape[0]
    with torch.no_grad():
        window = model.hann_window.to(history_x)
        band = model.band_mask
        spectrum_x = torch.fft.rfft(history_x * window, dim=-1)
        selected_x=spectrum_x[..., band]
        reference_power=(selected_x.real.square()+selected_x.imag.square()).sum(dim=-1)
        disturbance = history_e[:, None, :] + history_a
        spectrum_d = torch.fft.rfft(disturbance * window, dim=-1)
        selected_d=spectrum_d[..., band]
        denominator=(selected_d.real.square()+selected_d.imag.square()).sum(dim=-1)
        primary = torch.complex(model.primary_real, model.primary_imag).to(spectrum_x.device)
        prediction = primary[None, ...] * spectrum_x[:, None, :]
        difference=spectrum_d[..., band]-prediction[..., band]
        numerator=(difference.real.square()+difference.imag.square()).sum(dim=-1)
        score = numerator / denominator.clamp_min(1e-20)
        log_j = torch.log(score+1e-12)
        next_ewma = log_j if ewma is None else model.ewma_lambda*ewma + (1-model.ewma_lambda)*log_j
        masked_ewma = next_ewma.masked_fill(~candidate_mask, torch.inf)
        minimum = masked_ewma.amin(dim=-1, keepdim=True)
        centered = next_ewma-minimum
        centered = centered.masked_fill(~candidate_mask, 0.0)
        logits = (-centered/model.temperature).masked_fill(~candidate_mask, -1e9)
        proposal = torch.softmax(logits, dim=-1)
        next_alpha = (1-model.alpha_update)*alpha + model.alpha_update*proposal
        next_alpha = next_alpha*candidate_mask.to(next_alpha.dtype)
        next_alpha = next_alpha/next_alpha.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        probability = proposal.clamp_min(1e-12)
        count = candidate_mask.sum(dim=-1).clamp_min(2).to(history_x.dtype)
        entropy = -(probability*torch.log(probability)*candidate_mask).sum(dim=-1)/torch.log(count)
        sorted_values = torch.sort(masked_ewma, dim=-1).values
        gap = sorted_values[:, 1]-sorted_values[:, 0]
        minimum_score = score.masked_fill(~candidate_mask, torch.inf).amin(dim=-1)
        confidence = torch.stack((
            entropy, gap, torch.log(reference_power+1e-12), torch.log(minimum_score+1e-12),
        ), dim=-1)
        raw = torch.cat((
            centered, proposal, next_alpha, confidence,
            candidate_mask.to(history_x.dtype), latent.detach(),
        ), dim=-1)
        valid = (
            (reference_power > 1e-12) & torch.isfinite(raw).all(dim=-1)
            & candidate_mask.any(dim=-1) & torch.isfinite(denominator.masked_fill(~candidate_mask, 1.0)).all(dim=-1)
        )
        normalized = ((raw-model.feature_mean.to(raw))/model.feature_std.to(raw).clamp_min(1e-4)).clamp(-5, 5)
        normalized = torch.where(valid[:, None], normalized, torch.zeros_like(normalized))
        next_alpha = torch.where(valid[:, None], next_alpha, alpha)
        if ewma is not None:
            next_ewma = torch.where(valid[:, None], next_ewma, ewma)
    proposed_hidden = model.gru(normalized, hidden)
    proposed_latent = torch.tanh(model.latent_head(proposed_hidden))
    next_hidden = torch.where(valid[:, None], proposed_hidden, hidden)
    smoothed_latent = (1-model.latent_update)*latent + model.latent_update*proposed_latent
    next_latent = torch.where(valid[:, None], smoothed_latent, latent)
    return next_alpha, next_ewma, next_hidden, next_latent, raw, valid


def rollout_phase3g_closed_loop(
    model: GenerativeInnovationFIRController,
    reference: torch.Tensor,
    disturbance: torch.Tensor,
    physical_paths: torch.Tensor,
    path_slots: torch.Tensor,
    candidate_masks: torch.Tensor,
    teacher_indices: torch.Tensor,
    measured_blocks: torch.Tensor,
    *,
    truncate_blocks: int = 100,
) -> Phase3GRollout:
    """Roll out the two-rate controller with exact delayed feedback."""
    if reference.ndim != 2 or disturbance.shape != reference.shape:
        raise ValueError("reference and disturbance must have shape [B,T].")
    batch, samples = reference.shape
    if samples % model.block_size:
        raise ValueError("signal length must be divisible by block_size.")
    blocks = samples//model.block_size
    if physical_paths.ndim != 3 or physical_paths.shape[0] != batch:
        raise ValueError("physical_paths must have shape [B,S,L].")
    if path_slots.shape != (batch, blocks):
        raise ValueError("path_slots has an invalid shape.")
    if candidate_masks.shape != (batch, blocks, model.num_experts):
        raise ValueError("candidate_masks has an invalid shape.")
    if teacher_indices.shape != (batch, blocks) or measured_blocks.shape != (batch, blocks):
        raise ValueError("teacher metadata has an invalid shape.")

    expert_outputs = causal_filter_bank(reference, model.expert_filters)
    dictionary_outputs = causal_filter_bank(reference, model.residual_dictionary)
    alpha = candidate_masks[:, 0].to(reference.dtype)
    alpha = alpha/alpha.sum(dim=-1, keepdim=True).clamp_min(1.0)
    latent = torch.zeros(batch, model.latent_size, device=reference.device, dtype=reference.dtype)
    hidden = torch.zeros(batch, model.hidden_size, device=reference.device, dtype=reference.dtype)
    ewma: torch.Tensor | None = None
    path_length = physical_paths.shape[-1]
    physical_history = torch.zeros(batch, path_length-1, device=reference.device, dtype=reference.dtype)
    candidate_paths = model.secondary_paths.to(reference)[None].expand(batch, -1, -1)
    candidate_history = torch.zeros(
        batch, model.num_experts, model.path_length-1,
        device=reference.device, dtype=reference.dtype,
    )
    history_x = reference[:, :0]
    history_e = disturbance[:, :0]
    history_a = torch.zeros(batch, model.num_experts, 0, device=reference.device, dtype=reference.dtype)
    outputs=[]; raw_outputs=[]; residuals=[]; alphas=[]; latents=[]; features=[]
    kernel_terms=[]; delta_terms=[]; distill_numerator=reference.new_zeros(()); distill_denominator=reference.new_zeros(())
    previous_filter: torch.Tensor | None = None
    batch_indices = torch.arange(batch, device=reference.device)
    for block_index in range(blocks):
        start=block_index*model.block_size; end=start+model.block_size
        base_block=(expert_outputs[:, :, start:end]*alpha.unsqueeze(-1)).sum(dim=1)
        residual_block_output=(dictionary_outputs[:, :, start:end]*latent.unsqueeze(-1)).sum(dim=1)
        raw_block=base_block+residual_block_output
        limit=model.output_limit-1e-6
        output_block=limit*torch.tanh(raw_block/limit)
        current_path=physical_paths[batch_indices, path_slots[:, block_index]]
        anti, physical_history=_physical_path_block(output_block, physical_history, current_path)
        error_block=disturbance[:, start:end]-anti
        candidate_anti, candidate_history=_multi_path_block(
            output_block, candidate_history, candidate_paths,
        )
        history_x=torch.cat((history_x, reference[:, start:end]), dim=-1)[..., -model.n_fft:]
        history_e=torch.cat((history_e, error_block), dim=-1)[..., -model.n_fft:]
        history_a=torch.cat((history_a, candidate_anti), dim=-1)[..., -model.n_fft:]

        base_filter=alpha@model.expert_filters
        generated_filter=latent@model.residual_dictionary
        effective_filter=base_filter+generated_filter
        kernel_terms.append(generated_filter.square().mean(dim=-1)/(base_filter.square().mean(dim=-1).detach()+1e-12))
        if previous_filter is not None:
            delta_terms.append((effective_filter-previous_filter).square().mean(dim=-1)/(previous_filter.square().mean(dim=-1).detach()+1e-12))
        previous_filter=effective_filter
        teacher=teacher_indices[:, block_index]
        valid_teacher=measured_blocks[:, block_index] & (teacher >= 0) & (teacher < model.num_experts)
        selected=expert_outputs[batch_indices, teacher.clamp(0, model.num_experts-1), start:end]
        if bool(valid_teacher.any()):
            difference=(raw_block-selected).square()*valid_teacher[:, None]
            target_energy=selected.square()*valid_teacher[:, None]
            distill_numerator=distill_numerator+difference.sum()
            distill_denominator=distill_denominator+target_energy.sum()
        outputs.append(output_block); raw_outputs.append(raw_block); residuals.append(error_block)
        alphas.append(alpha); latents.append(latent)

        if history_x.shape[-1] >= model.n_fft:
            alpha, ewma, hidden, latent, raw_feature, _ = _innovation_update(
                model, history_x, history_e, history_a, candidate_masks[:, block_index],
                alpha, ewma, hidden, latent,
            )
        else:
            raw_feature=reference.new_zeros(batch, model.feature_dim)
        features.append(raw_feature)
        if truncate_blocks > 0 and (block_index+1) % truncate_blocks == 0:
            alpha=alpha.detach(); latent=latent.detach(); hidden=hidden.detach()
            ewma=None if ewma is None else ewma.detach()
            physical_history=physical_history.detach(); candidate_history=candidate_history.detach()
            history_x=history_x.detach(); history_e=history_e.detach(); history_a=history_a.detach()
            previous_filter=None if previous_filter is None else previous_filter.detach()

    zero=reference.new_zeros(())
    return Phase3GRollout(
        output=torch.cat(outputs, dim=-1), raw_output=torch.cat(raw_outputs, dim=-1),
        residual=torch.cat(residuals, dim=-1), alpha=torch.stack(alphas, dim=1),
        latent=torch.stack(latents, dim=1), raw_features=torch.stack(features, dim=1),
        kernel_loss=torch.stack(kernel_terms).mean() if kernel_terms else zero,
        kernel_delta_loss=torch.stack(delta_terms).mean() if delta_terms else zero,
        distill_loss=distill_numerator/(distill_denominator.detach()+1e-12),
    )


def compute_phase3g_loss(
    disturbance: torch.Tensor,
    rollout: Phase3GRollout,
    *,
    primary_weight: float=0.7,
    rebound_weight: float=0.3,
    time_weight: float=0.1,
    guard_weight: float=1.0,
    kernel_weight: float=0.02,
    kernel_delta_weight: float=0.01,
    distill_weight: float=0.01,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    metrics=compute_v6_metrics(disturbance, rollout.residual)
    primary=-metrics["primary_score_db"]/10.0
    rebound=metrics["rebound_score_db"]/10.0
    scored=slice(INITIALIZATION_SAMPLES, None)
    time_loss=rollout.residual[:, scored].square().mean()/(disturbance[:, scored].square().mean().detach()+1e-12)
    violation=torch.relu(rollout.raw_output.abs()-0.9).square()
    guard=violation.mean()+violation.amax()
    total=(primary_weight*primary+rebound_weight*rebound+time_weight*time_loss+guard_weight*guard
           +kernel_weight*rollout.kernel_loss+kernel_delta_weight*rollout.kernel_delta_loss
           +distill_weight*rollout.distill_loss)
    return total, {
        "total_loss": total.detach(), "primary_loss": primary.detach(),
        "rebound_loss": rebound.detach(), "time_loss": time_loss.detach(),
        "guard_loss": guard.detach(), "kernel_loss": rollout.kernel_loss.detach(),
        "kernel_delta_loss": rollout.kernel_delta_loss.detach(),
        "distill_loss": rollout.distill_loss.detach(),
        "primary_score_db": metrics["primary_score_db"].detach(),
        "rebound_score_db": metrics["rebound_score_db"].detach(),
        "controller_peak_abs": rollout.output.detach().abs().amax(),
        "raw_controller_peak_abs": rollout.raw_output.detach().abs().amax(),
    }
