"""Deterministic full-sequence data and path-switch sampling for Phase 3."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import soundfile as sf
import torch
from torch.utils.data import Dataset

from phase2_paths import augment_secondary_path
from v6_metrics import SAMPLE_RATE, TOTAL_SAMPLES


class Phase3SequenceDataset(Dataset):
    def __init__(
        self,
        dataset_dir: str | Path,
        noise_names: Sequence[str],
        train_paths: Sequence[int] = tuple(range(8)),
        *,
        samples_per_epoch: int = 128,
        block_size: int = 240,
        switch_probability: float = 0.0,
        augmentation_probability: float = 0.0,
        seed: int = 2026,
        skip_seconds: float = 20.0,
    ) -> None:
        self.dataset_dir = Path(dataset_dir)
        self.raw_dir = self.dataset_dir / "NOISE"
        self.expected_dir = self.dataset_dir / "EXPECTED_NOISE"
        self.train_paths = tuple(int(value) for value in train_paths)
        self.samples_per_epoch = int(samples_per_epoch)
        self.block_size = int(block_size)
        self.switch_probability = float(switch_probability)
        self.augmentation_probability = float(augmentation_probability)
        self.seed = int(seed)
        self.epoch = 0
        self.skip_samples = int(round(skip_seconds * SAMPLE_RATE))
        if TOTAL_SAMPLES % block_size or 96_000 % block_size:
            raise ValueError("block_size must divide 168000 and the 96000 path-switch sample.")
        if any(path not in range(8) for path in self.train_paths):
            raise ValueError("Phase 3 training is restricted to paths 1-8.")
        if not self.train_paths or not 0 <= switch_probability <= 1 or not 0 <= augmentation_probability <= 1:
            raise ValueError("invalid training paths or probabilities.")
        self.paths = np.load(self.dataset_dir / "sh.npy", allow_pickle=False).T.astype(np.float32)
        self.noises = [self._resolve_noise(str(name)) for name in noise_names]
        self.expected = {
            (raw.name, path): self._resolve_expected(raw, path)
            for raw in self.noises for path in self.train_paths
        }

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return self.samples_per_epoch

    def _resolve_noise(self, name: str) -> Path:
        candidates = [self.raw_dir / name]
        if Path(name).suffix.lower() != ".wav":
            candidates += [self.raw_dir / f"{name}.wav", self.raw_dir / f"{name}.WAV"]
        result = next((item for item in candidates if item.is_file()), None)
        if result is None:
            raise FileNotFoundError(f"Cannot resolve noise {name!r}.")
        return result

    def _resolve_expected(self, raw: Path, path: int) -> Path:
        suffix = f"_scene_{path + 1:02d}.wav"
        candidates = [self.expected_dir / f"{raw.stem}{suffix}", self.expected_dir / f"{raw.name}{suffix}"]
        result = next((item for item in candidates if item.is_file()), None)
        if result is None:
            raise FileNotFoundError(f"Missing expected signal for {raw.name}, path {path + 1}.")
        return result

    @staticmethod
    def _read(path: Path, start: int) -> np.ndarray:
        audio, _ = sf.read(str(path), start=start, frames=TOTAL_SAMPLES, dtype="float32", always_2d=False)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        if audio.size < TOTAL_SAMPLES:
            audio = np.pad(audio, (0, TOTAL_SAMPLES - audio.size))
        return np.asarray(audio, dtype=np.float32)

    def __getitem__(self, index: int):
        rng = np.random.default_rng(self.seed + self.epoch * 1_000_003 + int(index))
        first_path = self.train_paths[int(index) % len(self.train_paths)]
        do_switch = rng.random() < self.switch_probability and len(self.train_paths) > 1
        choices = [value for value in self.train_paths if value != first_path]
        second_path = int(rng.choice(choices)) if do_switch else first_path
        raw = self.noises[int(rng.integers(0, len(self.noises)))]
        max_start = sf.info(str(raw)).frames - TOTAL_SAMPLES
        start = int(rng.integers(self.skip_samples, max_start + 1)) if max_start >= self.skip_samples else max(0, max_start)
        reference = self._read(raw, start)
        target_a = self._read(self.expected[(raw.name, first_path)], start)
        target_b = self._read(self.expected[(raw.name, second_path)], start)
        switch_sample = 96_000 if do_switch else TOTAL_SAMPLES
        disturbance = np.concatenate((target_a[:switch_sample], target_b[switch_sample:]))

        path_values = []
        for base in (first_path, second_path):
            value = self.paths[base].copy()
            if rng.random() < self.augmentation_probability:
                gain = float(rng.uniform(-1.0, 1.0))
                delay = int(rng.integers(-1, 2))
                tail = float(rng.uniform(-35.0, -32.0))
                value, _ = augment_secondary_path(
                    value, rng, gain_db=gain, delay_samples=delay, tail_energy_db=tail,
                )
            path_values.append(value)

        blocks = TOTAL_SAMPLES // self.block_size
        switch_block = switch_sample // self.block_size
        slots = np.zeros(blocks, dtype=np.int64)
        labels = np.full(blocks, first_path, dtype=np.int64)
        if do_switch:
            slots[switch_block:] = 1
            labels[switch_block:] = second_path
        return (
            torch.from_numpy(reference), torch.from_numpy(disturbance),
            torch.from_numpy(np.stack(path_values)), torch.from_numpy(slots),
            torch.from_numpy(labels), torch.tensor(first_path), torch.tensor(second_path),
        )
