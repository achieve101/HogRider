"""Fixed three-scene validation data for DEEPANC Phase 1."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterator, Tuple

import numpy as np
import soundfile as sf
import torch

from v6_metrics import SAMPLE_RATE, TOTAL_SAMPLES


VALIDATION_START_SECONDS = 20.0
TRANSITION_SAMPLE = 96_000
MANIFEST_VERSION = 1


def _scan_wavs(directory: Path) -> list[Path]:
    return sorted(
        [path for path in directory.iterdir()
         if path.is_file() and path.suffix.lower() == ".wav"],
        key=lambda path: path.name.casefold(),
    )


def _find_named_noise(noise_dir: Path, token: str) -> Path:
    matches = [path for path in _scan_wavs(noise_dir) if token in path.stem]
    if len(matches) != 1:
        raise FileNotFoundError(
            f"Expected exactly one noise containing {token!r} in {noise_dir}, "
            f"found {[path.name for path in matches]}."
        )
    return matches[0]


def _resolve_expected_path(
    expected_dir: Path, raw_path: Path, path_index: int,
) -> Path:
    suffix = f"_scene_{path_index + 1:02d}.wav"
    candidates = [
        expected_dir / f"{raw_path.stem}{suffix}",
        expected_dir / f"{raw_path.name}{suffix}",
    ]
    match = next((path for path in candidates if path.is_file()), None)
    if match is None:
        raise FileNotFoundError(
            f"Missing expected noise for {raw_path.name}, path {path_index + 1}."
        )
    return match


def build_validation_manifest(dataset_dir: str | Path) -> Dict[str, object]:
    dataset_dir = Path(dataset_dir)
    noise_dir = dataset_dir / "NOISE"
    vehicle = _find_named_noise(noise_dir, "车载")
    restaurant = _find_named_noise(noise_dir, "餐厅")
    secondary_paths = np.load(dataset_dir / "sh.npy", allow_pickle=False).T
    if secondary_paths.ndim != 2 or secondary_paths.shape[0] != 10:
        raise ValueError(
            "Phase 1 expects exactly ten secondary paths after transposing sh.npy."
        )
    return {
        "manifest_version": MANIFEST_VERSION,
        "sample_rate": SAMPLE_RATE,
        "total_samples": TOTAL_SAMPLES,
        "initialization_samples": 24_000,
        "scoring_window_samples": 24_000,
        "scoring_window_count": 6,
        "start_seconds": VALIDATION_START_SECONDS,
        "transition_sample": TRANSITION_SAMPLE,
        "path_indices_zero_based": list(range(10)),
        "sources": {
            "vehicle": vehicle.name,
            "restaurant": restaurant.name,
        },
        "scenes": [
            {"name": "vehicle_continuous", "first": "vehicle", "second": None},
            {"name": "restaurant_continuous", "first": "restaurant", "second": None},
            {
                "name": "vehicle_to_restaurant",
                "first": "vehicle",
                "second": "restaurant",
                "transition_sample": TRANSITION_SAMPLE,
            },
        ],
    }


def _read_exact(path: Path, start: int, frames: int) -> np.ndarray:
    info = sf.info(str(path))
    if info.samplerate != SAMPLE_RATE:
        raise ValueError(f"Unexpected sample rate for {path}: {info.samplerate}.")
    if start + frames > info.frames:
        raise ValueError(f"Not enough audio in {path} for fixed Phase-1 validation.")
    audio, _ = sf.read(
        str(path), start=start, frames=frames,
        dtype="float32", always_2d=False,
    )
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)
    return np.asarray(audio, dtype=np.float32)


def iter_validation_examples(
    dataset_dir: str | Path,
    manifest: Dict[str, object] | None = None,
) -> Iterator[Tuple[str, int, torch.Tensor, torch.Tensor, torch.Tensor]]:
    """Yield scene name, path index, reference, secondary path, disturbance."""
    dataset_dir = Path(dataset_dir)
    manifest = manifest or build_validation_manifest(dataset_dir)
    raw_dir = dataset_dir / "NOISE"
    expected_dir = dataset_dir / "EXPECTED_NOISE"
    sources = {
        key: raw_dir / filename
        for key, filename in manifest["sources"].items()
    }
    paths = np.load(dataset_dir / "sh.npy", allow_pickle=False).T.astype(np.float32)
    start = int(round(float(manifest["start_seconds"]) * SAMPLE_RATE))

    for scene in manifest["scenes"]:
        scene_name = str(scene["name"])
        first_key = str(scene["first"])
        second_key = scene.get("second")
        for path_index in manifest["path_indices_zero_based"]:
            path_index = int(path_index)
            first_raw = sources[first_key]
            first_expected = _resolve_expected_path(
                expected_dir, first_raw, path_index,
            )
            if second_key is None:
                reference = _read_exact(first_raw, start, TOTAL_SAMPLES)
                disturbance = _read_exact(first_expected, start, TOTAL_SAMPLES)
            else:
                transition = int(scene["transition_sample"])
                second_raw = sources[str(second_key)]
                second_expected = _resolve_expected_path(
                    expected_dir, second_raw, path_index,
                )
                reference = np.concatenate([
                    _read_exact(first_raw, start, transition),
                    _read_exact(second_raw, start, TOTAL_SAMPLES - transition),
                ])
                disturbance = np.concatenate([
                    _read_exact(first_expected, start, transition),
                    _read_exact(second_expected, start, TOTAL_SAMPLES - transition),
                ])

            yield (
                scene_name,
                path_index,
                torch.from_numpy(reference.copy()),
                torch.from_numpy(paths[path_index].copy()),
                torch.from_numpy(disturbance.copy()),
            )

