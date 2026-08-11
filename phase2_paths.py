"""Path-group sampling, conservative IR augmentation, and robust Phase-2 loss."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Sequence

import numpy as np
import soundfile as sf
import torch
from torch.utils.data import Dataset

from dataset import apply_dynamic_path
from v6_metrics import INITIALIZATION_SAMPLES, compute_v6_metrics


PHASE2_TRAIN_PATHS = tuple(range(8))
PHASE2_FINAL_PATHS = (8, 9)


def _shift_ir(ir: np.ndarray, delay_samples: int) -> np.ndarray:
    shifted = np.zeros_like(ir)
    if delay_samples > 0:
        shifted[delay_samples:] = ir[:-delay_samples]
    elif delay_samples < 0:
        advance = -delay_samples
        shifted[:-advance] = ir[advance:]
    else:
        shifted[:] = ir
    return shifted


def augment_secondary_path(
    path: np.ndarray,
    rng: np.random.Generator,
    *,
    gain_db: float | None = None,
    delay_samples: int | None = None,
    tail_energy_db: float | None = None,
    tail_start: int = 64,
) -> tuple[np.ndarray, dict[str, float | int]]:
    """Apply bounded gain/delay and a smooth, low-energy late-tail perturbation."""
    original = np.asarray(path, dtype=np.float32)
    if original.ndim != 1 or original.size <= tail_start:
        raise ValueError("path must be a 1-D IR longer than tail_start.")
    gain_db = float(rng.uniform(-1.5, 1.5) if gain_db is None else gain_db)
    delay_samples = int(rng.integers(-2, 3) if delay_samples is None else delay_samples)
    tail_energy_db = float(
        rng.uniform(-35.0, -30.0) if tail_energy_db is None else tail_energy_db
    )
    if not -1.5 <= gain_db <= 1.5:
        raise ValueError("gain_db must be within [-1.5, 1.5].")
    if not -2 <= delay_samples <= 2:
        raise ValueError("delay_samples must be within [-2, 2].")
    if not -35.0 <= tail_energy_db <= -30.0:
        raise ValueError("tail_energy_db must be within [-35, -30].")

    augmented = _shift_ir(original, delay_samples).astype(np.float64)
    augmented *= 10.0 ** (gain_db / 20.0)
    tail_length = original.size - tail_start
    noise = rng.standard_normal(tail_length)
    smoothing = np.hanning(17)
    smoothing /= smoothing.sum()
    smooth = np.convolve(noise, smoothing, mode="same")
    decay = np.exp(-np.linspace(0.0, 5.0, tail_length))
    perturbation = smooth * decay
    target_energy = (
        float(np.square(augmented).sum()) * 10.0 ** (tail_energy_db / 10.0)
    )
    perturbation *= math.sqrt(target_energy / (float(np.square(perturbation).sum()) + 1e-30))
    augmented[tail_start:] += perturbation
    return augmented.astype(np.float32), {
        "gain_db": gain_db,
        "delay_samples": delay_samples,
        "tail_energy_db": tail_energy_db,
        "tail_start": tail_start,
    }


class Phase2GroupedDataset(Dataset):
    """Return one reference and a physically aligned group of path/disturbance pairs."""

    def __init__(
        self,
        dataset_dir: str | Path,
        noise_names: Sequence[str],
        train_path_indices: Sequence[int] = PHASE2_TRAIN_PATHS,
        *,
        group_size: int = 4,
        use_augmentation: bool = True,
        augmentation_probability: float = 0.8,
        segment_duration: float = 3.5,
        sample_rate: int = 48_000,
        samples_per_epoch: int = 256,
        skip_seconds: float = 20.0,
        seed: int = 2026,
    ) -> None:
        self.dataset_dir = Path(dataset_dir)
        self.noise_names = [str(name) for name in noise_names]
        self.train_paths = tuple(int(index) for index in train_path_indices)
        self.group_size = int(group_size)
        self.use_augmentation = bool(use_augmentation)
        self.augmentation_probability = float(augmentation_probability)
        self.segment_length = int(round(segment_duration * sample_rate))
        self.sample_rate = int(sample_rate)
        self.samples_per_epoch = int(samples_per_epoch)
        self.skip_samples = int(round(skip_seconds * sample_rate))
        self.seed = int(seed)
        self.epoch = 0
        if not self.noise_names or not self.train_paths:
            raise ValueError("noise_names and train_path_indices must be non-empty.")
        if any(index not in PHASE2_TRAIN_PATHS for index in self.train_paths):
            raise ValueError("Phase 2 training may only use paths 1-8 (indices 0-7).")
        if len(set(self.train_paths)) != len(self.train_paths):
            raise ValueError("train_path_indices must be unique.")
        if self.group_size <= 0 or self.samples_per_epoch <= 0:
            raise ValueError("group_size and samples_per_epoch must be positive.")
        if not 0.0 <= self.augmentation_probability <= 1.0:
            raise ValueError("augmentation_probability must be in [0, 1].")

        self.paths = np.load(self.dataset_dir / "sh.npy", allow_pickle=False).T.astype(np.float32)
        if self.paths.ndim != 2 or self.paths.shape[0] < 10:
            raise ValueError("sh.npy must contain at least ten secondary paths.")
        self.raw_dir = self.dataset_dir / "NOISE"
        self.expected_dir = self.dataset_dir / "EXPECTED_NOISE"
        self.noises = [self._resolve_noise(name) for name in self.noise_names]
        self.expected: dict[tuple[str, int], Path] = {}
        for raw in self.noises:
            for path_index in self.train_paths:
                self.expected[(raw.name, path_index)] = self._resolve_expected(raw, path_index)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return self.samples_per_epoch

    def _resolve_noise(self, name: str) -> Path:
        candidates = [self.raw_dir / name]
        if Path(name).suffix.lower() != ".wav":
            candidates.extend((self.raw_dir / f"{name}.wav", self.raw_dir / f"{name}.WAV"))
        path = next((candidate for candidate in candidates if candidate.is_file()), None)
        if path is None:
            raise FileNotFoundError(f"Cannot resolve noise {name!r}.")
        if sf.info(str(path)).samplerate != self.sample_rate:
            raise ValueError(f"Unexpected sample rate for {path}.")
        return path

    def _resolve_expected(self, raw: Path, path_index: int) -> Path:
        suffix = f"_scene_{path_index + 1:02d}.wav"
        candidates = (
            self.expected_dir / f"{raw.stem}{suffix}",
            self.expected_dir / f"{raw.name}{suffix}",
        )
        path = next((candidate for candidate in candidates if candidate.is_file()), None)
        if path is None:
            raise FileNotFoundError(f"Missing expected noise for {raw.name}, path {path_index + 1}.")
        return path

    @staticmethod
    def _read(path: Path, start: int, frames: int) -> np.ndarray:
        audio, _ = sf.read(str(path), start=start, frames=frames, dtype="float32", always_2d=False)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        if audio.size < frames:
            audio = np.pad(audio, (0, frames - audio.size))
        return np.asarray(audio, dtype=np.float32)

    def _group_members(self, anchor: int, rng: np.random.Generator) -> list[tuple[int, bool]]:
        others = [index for index in self.train_paths if index != anchor]
        members: list[tuple[int, bool]] = [(anchor, False)]
        if self.group_size == 1:
            return members
        real_other = int(rng.choice(others if others else [anchor]))
        members.append((real_other, False))
        while len(members) < self.group_size:
            base = int(rng.choice(self.train_paths))
            augmented = self.use_augmentation and rng.random() < self.augmentation_probability
            members.append((base, augmented))
        return members

    def __getitem__(self, index: int):
        rng = np.random.default_rng(self.seed + self.epoch * 1_000_003 + int(index))
        anchor = self.train_paths[int(index) % len(self.train_paths)]
        members = self._group_members(anchor, rng)
        raw = self.noises[int(rng.integers(0, len(self.noises)))]
        total_frames = sf.info(str(raw)).frames
        max_start = total_frames - self.segment_length
        start = int(rng.integers(self.skip_samples, max_start + 1)) if max_start >= self.skip_samples else max(0, max_start)
        reference = self._read(raw, start, self.segment_length)

        paths, disturbances, base_indices, augmented_flags = [], [], [], []
        for base_index, do_augment in members:
            path = self.paths[base_index].copy()
            if do_augment:
                path, _ = augment_secondary_path(path, rng)
            paths.append(path)
            disturbances.append(self._read(self.expected[(raw.name, base_index)], start, self.segment_length))
            base_indices.append(base_index)
            augmented_flags.append(do_augment)
        return (
            torch.from_numpy(reference),
            torch.from_numpy(np.stack(paths)),
            torch.from_numpy(np.stack(disturbances)),
            torch.tensor(base_indices, dtype=torch.int64),
            torch.tensor(augmented_flags, dtype=torch.bool),
        )


def compute_phase2_group_loss(
    disturbance: torch.Tensor,
    controller_output: torch.Tensor,
    secondary_paths: torch.Tensor,
    *,
    beta: float = 0.25,
    primary_weight: float = 0.7,
    rebound_weight: float = 0.3,
    time_weight: float = 0.1,
    guard_weight: float = 1.0,
    guard_limit: float = 1.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compute mean-path plus top-quartile path loss for [B,K,T] targets."""
    if disturbance.ndim != 3 or secondary_paths.ndim != 3 or controller_output.ndim != 2:
        raise ValueError("Expected disturbance [B,K,T], paths [B,K,L], controller [B,T].")
    batch, group, samples = disturbance.shape
    if controller_output.shape != (batch, samples) or secondary_paths.shape[:2] != (batch, group):
        raise ValueError("Batch/group dimensions do not align.")
    if beta < 0 or guard_limit <= 0:
        raise ValueError("beta must be nonnegative and guard_limit positive.")
    expanded_controller = controller_output[:, None, :].expand(-1, group, -1).reshape(batch * group, samples)
    flat_paths = secondary_paths.reshape(batch * group, secondary_paths.shape[-1])
    flat_target = disturbance.reshape(batch * group, samples)
    residual = flat_target - apply_dynamic_path(expanded_controller, flat_paths)
    metrics = compute_v6_metrics(flat_target, residual)
    primary_per = -metrics["primary_window_db"].mean(dim=1).reshape(batch, group) / 10.0
    rebound_per = metrics["rebound_window_db"].mean(dim=1).reshape(batch, group) / 10.0
    target_scored = flat_target[:, INITIALIZATION_SAMPLES:]
    residual_scored = residual[:, INITIALIZATION_SAMPLES:]
    time_per = (
        residual_scored.square().mean(dim=1)
        / (target_scored.square().mean(dim=1).detach() + 1e-12)
    ).reshape(batch, group)
    path_loss = primary_weight * primary_per + rebound_weight * rebound_per + time_weight * time_per
    mean_term, worst_term, top_count = robust_path_reduce(path_loss)
    violation_squared = torch.relu(controller_output.abs() - guard_limit).square()
    guard_loss = violation_squared.mean() + violation_squared.amax()
    total = mean_term + beta * worst_term + guard_weight * guard_loss
    components = {
        "total_loss": total.detach(),
        "mean_path_loss": mean_term.detach(),
        "worst_quartile_loss": worst_term.detach(),
        "primary_loss": primary_per.mean().detach(),
        "rebound_loss": rebound_per.mean().detach(),
        "time_loss": time_per.mean().detach(),
        "guard_loss": guard_loss.detach(),
        "primary_score_db": (-10.0 * primary_per.mean()).detach(),
        "rebound_score_db": (10.0 * rebound_per.mean()).detach(),
        "controller_peak_abs": controller_output.detach().abs().amax(),
        "top_quartile_count": torch.tensor(top_count, device=total.device),
    }
    return total, components


def robust_path_reduce(path_loss: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Return global mean and per-sample top-quartile mean for a [B,K] loss."""
    if path_loss.ndim != 2 or path_loss.shape[1] < 1:
        raise ValueError("path_loss must have shape [batch, paths] with paths >= 1.")
    top_count = max(1, math.ceil(path_loss.shape[1] * 0.25))
    return path_loss.mean(), torch.topk(path_loss, k=top_count, dim=1).values.mean(), top_count
