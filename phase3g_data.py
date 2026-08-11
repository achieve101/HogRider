"""Deterministic continuous-path synthesis for Phase 3G."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import soundfile as sf
import torch
from torch.utils.data import Dataset

from phase2_paths import augment_secondary_path
from phase3r_templates import _resolve_expected, sha256_file
from v6_metrics import SAMPLE_RATE, TOTAL_SAMPLES


SYNTHESIS_PROBABILITIES = {
    "measured": 0.50, "interpolate": 0.30, "extrapolate": 0.10, "augment": 0.10,
}
NEIGHBOR_POLICY_SINGLE = "single_nearest"
NEIGHBOR_POLICY_E10A = "three_neighbor_multispace"


def _shift_ir(path: np.ndarray, delay: int) -> np.ndarray:
    result = np.zeros_like(path)
    if delay >= 0:
        if delay < path.size:
            result[delay:] = path[:path.size-delay]
    elif -delay < path.size:
        result[:delay] = path[-delay:]
    return result


def globally_align_ir(reference: np.ndarray, other: np.ndarray) -> np.ndarray:
    """Align direct-arrival peaks without circular wraparound."""
    delay = int(np.argmax(np.abs(reference)) - np.argmax(np.abs(other)))
    return _shift_ir(np.asarray(other, dtype=np.float64), delay)


def dtw_align_ir(reference: np.ndarray, other: np.ndarray, radius: int = 16) -> np.ndarray:
    """Warp ``other`` onto ``reference`` with a bounded deterministic DTW path."""
    first = np.asarray(reference, dtype=np.float64)
    second = globally_align_ir(first, np.asarray(other, dtype=np.float64))
    if first.ndim != 1 or second.shape != first.shape or radius < 1:
        raise ValueError("DTW inputs must be equal-length one-dimensional arrays.")
    length = first.size
    width = 2 * radius + 1
    cost = np.full((length, width), np.inf, dtype=np.float64)
    parent = np.full((length, width), -1, dtype=np.int8)
    scale = max(float(np.sqrt(np.mean(np.square(first)))), float(np.sqrt(np.mean(np.square(second)))), 1e-12)
    a, b = first / scale, second / scale
    for i in range(length):
        lo, hi = max(0, i-radius), min(length, i+radius+1)
        for j in range(lo, hi):
            slot = j - i + radius
            local = (a[i] - b[j]) ** 2
            if i == 0 and j == 0:
                cost[i, slot] = local
                continue
            choices: list[tuple[float, int]] = []
            if i > 0 and abs(j-(i-1)) <= radius:
                choices.append((cost[i-1, j-(i-1)+radius], 0))
            if j > 0 and slot-1 >= 0:
                choices.append((cost[i, slot-1], 1))
            if i > 0 and j > 0 and abs((j-1)-(i-1)) <= radius:
                choices.append((cost[i-1, (j-1)-(i-1)+radius], 2))
            if choices:
                previous, direction = min(choices, key=lambda value: (value[0], value[1]))
                cost[i, slot] = local + previous
                parent[i, slot] = direction
    if not np.isfinite(cost[-1, radius]):
        return second
    mapping: list[list[int]] = [[] for _ in range(length)]
    i = j = length - 1
    while True:
        mapping[i].append(j)
        if i == 0 and j == 0:
            break
        direction = int(parent[i, j-i+radius])
        if direction == 0:
            i -= 1
        elif direction == 1:
            j -= 1
        elif direction == 2:
            i -= 1; j -= 1
        else:
            return second
    warped = np.empty_like(second)
    last = 0.0
    for index, matches in enumerate(mapping):
        if matches:
            last = float(np.mean(second[matches]))
        warped[index] = last
    return warped


def _band_energy(path: np.ndarray) -> float:
    spectrum = np.fft.rfft(path, n=4096)
    frequency = np.fft.rfftfreq(4096, 1.0 / SAMPLE_RATE)
    mask = (frequency >= 50.0) & (frequency <= 8000.0)
    return float(np.sum(np.abs(spectrum[mask]) ** 2))


def _scaled_nrmse(target: np.ndarray, candidate: np.ndarray) -> float:
    first = np.asarray(target, dtype=np.complex128).reshape(-1)
    second = np.asarray(candidate, dtype=np.complex128).reshape(-1)
    denominator = float(np.vdot(second, second).real)
    scale = float(np.vdot(second, first).real / max(denominator, 1e-20))
    return float(np.linalg.norm(first - scale * second) / max(np.linalg.norm(first), 1e-20))


def multispace_path_distances(
    secondary_paths: np.ndarray,
    primary_real: np.ndarray,
    primary_imag: np.ndarray,
    band_mask: np.ndarray,
    path_indices: Sequence[int],
) -> dict[int, dict[int, dict[str, float]]]:
    """Return retained-only distances used by the frozen E10-A sampler."""
    paths = tuple(int(value) for value in path_indices)
    if len(paths) < 4 or len(set(paths)) != len(paths) or any(value not in range(8) for value in paths):
        raise ValueError("E10-A requires at least four unique retained paths from paths 1-8.")
    secondary = np.asarray(secondary_paths, dtype=np.float64)
    primary = np.asarray(primary_real, dtype=np.float64) + 1j * np.asarray(primary_imag, dtype=np.float64)
    band = np.asarray(band_mask, dtype=bool)
    if secondary.ndim != 2 or primary.ndim != 2 or primary.shape[0] != secondary.shape[0]:
        raise ValueError("E10-A secondary paths and primary templates have incompatible shapes.")
    result: dict[int, dict[int, dict[str, float]]] = {}
    frequencies = np.fft.rfftfreq(4096, 1.0 / SAMPLE_RATE)
    response_band = (frequencies >= 50.0) & (frequencies <= 8000.0)
    for first in paths:
        first_response = np.fft.rfft(secondary[first], n=4096)[response_band]
        candidates = {}
        for second in paths:
            if second == first:
                continue
            aligned = globally_align_ir(secondary[first], secondary[second])
            second_response = np.fft.rfft(secondary[second], n=4096)[response_band]
            candidates[second] = {
                "aligned_ir_nrmse": _scaled_nrmse(secondary[first], aligned),
                "secondary_response_nrmse": _scaled_nrmse(first_response, second_response),
                "primary_template_nrmse": _scaled_nrmse(primary[first, band], primary[second, band]),
            }
        result[first] = candidates
    return result


def build_multispace_neighbor_table(
    secondary_paths: np.ndarray,
    primary_real: np.ndarray,
    primary_imag: np.ndarray,
    band_mask: np.ndarray,
    path_indices: Sequence[int],
    *,
    neighbor_count: int = 3,
) -> tuple[dict[int, tuple[int, ...]], dict[int, dict[int, dict[str, float]]]]:
    """Rank each retained path in three spaces and keep its closest peers."""
    distances = multispace_path_distances(
        secondary_paths, primary_real, primary_imag, band_mask, path_indices,
    )
    if neighbor_count < 1 or any(len(values) < neighbor_count for values in distances.values()):
        raise ValueError("neighbor_count exceeds the retained candidate count.")
    table = {}
    metric_names = ("aligned_ir_nrmse", "secondary_response_nrmse", "primary_template_nrmse")
    for first, candidates in distances.items():
        ranks: dict[str, dict[int, int]] = {}
        for metric in metric_names:
            ordered = sorted(candidates, key=lambda value: (candidates[value][metric], value))
            ranks[metric] = {path: index + 1 for index, path in enumerate(ordered)}
        for second, values in candidates.items():
            values["mean_rank"] = float(np.mean([ranks[metric][second] for metric in metric_names]))
        ordered = sorted(candidates, key=lambda value: (candidates[value]["mean_rank"], value))
        table[first] = tuple(ordered[:neighbor_count])
    return table, distances


def synthesize_path(
    first: np.ndarray,
    second: np.ndarray,
    *,
    mode: str,
    amount: float,
    radius: int = 16,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Create a bounded interpolation or extrapolation from two training IRs."""
    if mode not in {"interpolate", "extrapolate"}:
        raise ValueError("mode must be interpolate or extrapolate.")
    if mode == "interpolate" and not 0.2 <= amount <= 0.8:
        raise ValueError("interpolation amount must lie in [0.2, 0.8].")
    if mode == "extrapolate" and not 0.05 <= amount <= 0.20:
        raise ValueError("extrapolation amount must lie in [0.05, 0.20].")
    first = np.asarray(first, dtype=np.float64)
    aligned_dtw = dtw_align_ir(first, second, radius)
    aligned_global = globally_align_ir(first, second)
    if mode == "interpolate":
        generated = (1.0-amount) * first + amount * aligned_dtw
        fallback = (1.0-amount) * first + amount * aligned_global
    else:
        generated = first + amount * (first-aligned_dtw)
        fallback = first + amount * (first-aligned_global)
    peak = int(np.argmax(np.abs(generated)))
    endpoint_peaks = [int(np.argmax(np.abs(first))), int(np.argmax(np.abs(aligned_global)))]
    energies = [_band_energy(first), _band_energy(aligned_global)]
    energy = _band_energy(generated)
    low = min(energies) * 10 ** (-1.0/10.0)
    high = max(energies) * 10 ** (1.0/10.0)
    valid = (
        np.all(np.isfinite(generated))
        and min(endpoint_peaks)-1 <= peak <= max(endpoint_peaks)+1
        and low <= energy <= high
    )
    if not valid:
        generated = fallback
    return generated.astype(np.float32), {
        "mode": mode, "amount": float(amount), "dtw_radius": radius,
        "used_global_fallback": not valid,
    }


def build_phase3g_manifest(
    dataset_dir: str | Path,
    train_noises: Sequence[str],
    *,
    path_indices: Sequence[int] = tuple(range(8)),
    seed: int = 2026,
    stress_seed: int = 3030,
    neighbor_policy: str = NEIGHBOR_POLICY_SINGLE,
    neighbor_table: dict[int, Sequence[int]] | None = None,
    correction_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    paths = tuple(int(value) for value in path_indices)
    if not paths or any(value not in range(8) for value in paths):
        raise ValueError("Phase 3G manifests are restricted to paths 1-8.")
    root = Path(dataset_dir)
    inputs = [root / "sh.npy"]
    for name in train_noises:
        raw = root / "NOISE" / str(name)
        inputs.append(raw)
        inputs.extend(_resolve_expected(root / "EXPECTED_NOISE", raw, index) for index in paths)
    stress_rng = np.random.default_rng(stress_seed)
    stress = []
    for index in range(48):
        first = int(stress_rng.choice(paths))
        second = int(stress_rng.choice([value for value in paths if value != first]))
        mode = "interpolate" if index < 32 else "extrapolate"
        limits = (0.2, 0.8) if mode == "interpolate" else (0.05, 0.20)
        stress.append({
            "case": index, "first_path_zero_based": first, "second_path_zero_based": second,
            "mode": mode, "amount": float(stress_rng.uniform(*limits)),
        })
    if neighbor_policy not in {NEIGHBOR_POLICY_SINGLE, NEIGHBOR_POLICY_E10A}:
        raise ValueError(f"Unsupported Phase-3G neighbor policy: {neighbor_policy}")
    serialized_neighbors = None
    if neighbor_policy == NEIGHBOR_POLICY_E10A:
        if neighbor_table is None or set(neighbor_table) != set(paths):
            raise ValueError("E10-A requires a neighbor row for every retained path.")
        serialized_neighbors = {
            str(first + 1): [int(second) + 1 for second in neighbor_table[first]]
            for first in paths
        }
    return {
        "manifest_version": 1, "phase": "3G", "sample_rate": SAMPLE_RATE,
        "total_samples": TOTAL_SAMPLES, "seed": seed, "stress_seed": stress_seed,
        "train_noises": [str(value) for value in train_noises],
        "path_indices_zero_based": list(paths), "synthesis_probabilities": SYNTHESIS_PROBABILITIES,
        "dtw_radius": 16, "switch_probability": 0.25, "switch_sample": 96_000,
        "candidate_mask_probabilities": {"none": 0.4, "one_endpoint": 0.3, "two_endpoints": 0.3},
        "neighbor_policy": neighbor_policy,
        "neighbor_table_one_based": serialized_neighbors,
        "correction": correction_metadata,
        "stress_cases": stress,
        "input_sha256": {str(path.relative_to(root)): sha256_file(path) for path in sorted(set(inputs))},
        "sealed_paths_touched": False,
    }


class Phase3GSequenceDataset(Dataset):
    """Return full causal sequences with deterministic continuous-path synthesis."""

    def __init__(
        self,
        dataset_dir: str | Path,
        noise_names: Sequence[str],
        *,
        train_paths: Sequence[int] = tuple(range(8)),
        samples_per_epoch: int = 128,
        block_size: int = 240,
        synthesis_enabled: bool = True,
        switch_probability: float = 0.25,
        seed: int = 2026,
        neighbor_policy: str = NEIGHBOR_POLICY_SINGLE,
        neighbor_table: dict[int, Sequence[int]] | None = None,
    ) -> None:
        self.root = Path(dataset_dir)
        self.noise_names = tuple(str(value) for value in noise_names)
        self.train_paths = tuple(int(value) for value in train_paths)
        if not self.train_paths or any(value not in range(8) for value in self.train_paths):
            raise ValueError("Phase 3G training is restricted to paths 1-8.")
        if TOTAL_SAMPLES % block_size or 96_000 % block_size:
            raise ValueError("block_size must divide sequence and switch boundaries.")
        self.samples_per_epoch = int(samples_per_epoch); self.block_size = int(block_size)
        self.synthesis_enabled = bool(synthesis_enabled)
        self.switch_probability = float(switch_probability); self.seed = int(seed); self.epoch = 0
        if neighbor_policy not in {NEIGHBOR_POLICY_SINGLE, NEIGHBOR_POLICY_E10A}:
            raise ValueError(f"Unsupported Phase-3G neighbor policy: {neighbor_policy}")
        self.neighbor_policy = str(neighbor_policy)
        all_paths = np.load(self.root / "sh.npy", allow_pickle=False, mmap_mode="r").T
        self.paths = np.asarray(all_paths[list(self.train_paths)], dtype=np.float32).copy()
        self.index_to_local = {path: position for position, path in enumerate(self.train_paths)}
        self.raw_files = [self.root / "NOISE" / name for name in self.noise_names]
        if any(not path.is_file() for path in self.raw_files):
            raise FileNotFoundError("A Phase 3G training noise is missing.")
        self.expected = {
            (raw.name, path): _resolve_expected(self.root / "EXPECTED_NOISE", raw, path)
            for raw in self.raw_files for path in self.train_paths
        }
        self.nearest = self._nearest_paths()
        self.neighbor_table: dict[int, tuple[int, ...]] | None = None
        if self.neighbor_policy == NEIGHBOR_POLICY_E10A:
            if neighbor_table is None or set(int(value) for value in neighbor_table) != set(self.train_paths):
                raise ValueError("E10-A requires a frozen neighbor row for every retained path.")
            normalized = {
                int(first): tuple(int(second) for second in values)
                for first, values in neighbor_table.items()
            }
            for first, values in normalized.items():
                if len(values) != 3 or len(set(values)) != 3:
                    raise ValueError("Every E10-A path must have exactly three unique neighbors.")
                if first in values or any(value not in self.train_paths for value in values):
                    raise ValueError("E10-A neighbors must be retained paths distinct from their source.")
            self.neighbor_table = normalized

    def _nearest_paths(self) -> dict[int, int]:
        result = {}
        for path in self.train_paths:
            first = self.paths[self.index_to_local[path]].astype(np.float64)
            choices = []
            for other in self.train_paths:
                if other == path:
                    continue
                aligned = globally_align_ir(first, self.paths[self.index_to_local[other]])
                choices.append((float(np.mean((first-aligned)**2)), other))
            result[path] = min(choices)[1]
        return result

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return self.samples_per_epoch

    @staticmethod
    def _read(path: Path, start: int) -> np.ndarray:
        value, sr = sf.read(str(path), start=start, frames=TOTAL_SAMPLES, dtype="float32", always_2d=False)
        if sr != SAMPLE_RATE or len(value) != TOTAL_SAMPLES:
            raise ValueError(f"Invalid Phase 3G audio segment: {path}.")
        return np.asarray(value.mean(axis=1) if value.ndim == 2 else value, dtype=np.float32)

    def _sample_kind(self, rng: np.random.Generator) -> str:
        if not self.synthesis_enabled:
            return "measured"
        value = rng.random()
        if value < 0.50: return "measured"
        if value < 0.80: return "interpolate"
        if value < 0.90: return "extrapolate"
        return "augment"

    def _select_second(self, first: int, kind: str, rng: np.random.Generator) -> int:
        if kind in {"interpolate", "extrapolate"} and self.neighbor_policy == NEIGHBOR_POLICY_E10A:
            assert self.neighbor_table is not None
            return int(rng.choice(self.neighbor_table[first]))
        return int(self.nearest[first])

    def _environment(self, rng: np.random.Generator, raw: Path) -> dict[str, Any]:
        first = int(rng.choice(self.train_paths))
        kind = self._sample_kind(rng)
        second = self._select_second(first, kind, rng)
        path_a = self.paths[self.index_to_local[first]].copy()
        d_a = self._read(self.expected[(raw.name, first)], self._start)
        endpoints = [first]
        teacher = self.index_to_local[first]
        measured = kind in {"measured", "augment"}
        amount = 0.0
        if kind == "augment":
            path_a, _ = augment_secondary_path(
                path_a, rng, gain_db=float(rng.uniform(-1.0, 1.0)),
                delay_samples=int(rng.integers(-1, 2)),
                tail_energy_db=float(rng.uniform(-35.0, -32.0)),
            )
        elif kind in {"interpolate", "extrapolate"}:
            limits = (0.2, 0.8) if kind == "interpolate" else (0.05, 0.20)
            amount = float(rng.uniform(*limits))
            path_b = self.paths[self.index_to_local[second]]
            path_a, _ = synthesize_path(path_a, path_b, mode=kind, amount=amount)
            d_b = self._read(self.expected[(raw.name, second)], self._start)
            d_a = ((1-amount)*d_a + amount*d_b if kind == "interpolate" else d_a + amount*(d_a-d_b)).astype(np.float32)
            endpoints = [first, second]; teacher = -1; measured = False
        mask = np.ones(len(self.train_paths), dtype=bool)
        mask_draw = rng.random()
        if mask_draw >= 0.4:
            remove = endpoints[:1] if mask_draw < 0.7 else endpoints[:2]
            if len(remove) < 2 and mask_draw >= 0.7:
                remove.append(self.nearest[first])
            for path in remove:
                mask[self.index_to_local[path]] = False
            if not mask.any():
                mask[self.index_to_local[self.nearest[first]]] = True
        return {
            "path": path_a.astype(np.float32), "disturbance": d_a,
            "candidate_mask": mask, "teacher": teacher, "measured": measured,
            "kind": kind, "first": first, "second": second, "amount": amount,
        }

    def __getitem__(self, index: int):
        rng = np.random.default_rng(self.seed + self.epoch*1_000_003 + int(index))
        raw = self.raw_files[int(rng.integers(0, len(self.raw_files)))]
        info = sf.info(str(raw)); maximum = info.frames - TOTAL_SAMPLES
        minimum = 20*SAMPLE_RATE
        if maximum < minimum:
            raise ValueError(f"Training noise is too short: {raw}.")
        self._start = int(rng.integers(minimum, maximum+1))
        reference = self._read(raw, self._start)
        first = self._environment(rng, raw)
        switch = rng.random() < self.switch_probability
        second = self._environment(rng, raw) if switch else first
        switch_sample = 96_000 if switch else TOTAL_SAMPLES
        disturbance = np.concatenate((first["disturbance"][:switch_sample], second["disturbance"][switch_sample:]))
        paths = np.stack((first["path"], second["path"]))
        blocks = TOTAL_SAMPLES // self.block_size; switch_block = switch_sample // self.block_size
        slots = np.zeros(blocks, dtype=np.int64); slots[switch_block:] = 1
        masks = np.repeat(first["candidate_mask"][None, :], blocks, axis=0)
        masks[switch_block:] = second["candidate_mask"]
        teachers = np.full(blocks, first["teacher"], dtype=np.int64); teachers[switch_block:] = second["teacher"]
        measured = np.full(blocks, first["measured"], dtype=bool); measured[switch_block:] = second["measured"]
        return (
            torch.from_numpy(reference), torch.from_numpy(disturbance), torch.from_numpy(paths),
            torch.from_numpy(slots), torch.from_numpy(masks), torch.from_numpy(teachers),
            torch.from_numpy(measured),
        )


def save_phase3g_manifest(path: str | Path, manifest: dict[str, Any]) -> None:
    Path(path).write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
