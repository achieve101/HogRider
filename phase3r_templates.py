"""Deterministic Phase-3R primary/secondary-path template generation."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from v6_metrics import SAMPLE_RATE


DEFAULT_TRAIN_NOISES = (
    "KTV.wav", "鍏氦.wav", "鍘ㄦ埧.wav", "鍦伴搧.wav", "姝ヨ琛?wav", "鐏溅.wav",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _resolve_expected(expected_dir: Path, raw: Path, path_index: int) -> Path:
    suffix = f"_scene_{path_index + 1:02d}.wav"
    candidates = (expected_dir / f"{raw.stem}{suffix}", expected_dir / f"{raw.name}{suffix}")
    found = next((path for path in candidates if path.is_file()), None)
    if found is None:
        raise FileNotFoundError(f"Missing expected signal for {raw.name}, path {path_index + 1}.")
    return found


def _read_mono_tail(path: Path, start: int, frames: int | None = None) -> np.ndarray:
    info = sf.info(str(path))
    if info.samplerate != SAMPLE_RATE:
        raise ValueError(f"Unexpected sample rate for {path}: {info.samplerate}.")
    audio, _ = sf.read(str(path), start=start, frames=-1 if frames is None else frames,
                       dtype="float64", always_2d=False)
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    return np.asarray(audio, dtype=np.float64)


def _accumulate_multi_path_files(
    raw_path: Path, expected_paths: list[Path], start: int, available: int,
    window: np.ndarray, hop: int, device: torch.device,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Stream files in bounded memory and reuse each reference FFT across paths."""
    length = len(window)
    frame_count = 1 + (available - length) // hop
    if frame_count <= 0:
        raise ValueError("Training source is shorter than one template FFT window.")
    cross = torch.zeros((len(expected_paths), length // 2 + 1), dtype=torch.complex64, device=device)
    power = torch.zeros(length // 2 + 1, dtype=torch.float32, device=device)
    window_t = torch.from_numpy(window.astype(np.float32)).to(device)
    with sf.SoundFile(str(raw_path)) as raw_handle:
        expected_handles = [sf.SoundFile(str(path)) for path in expected_paths]
        try:
            for first in range(0, frame_count, 512):
                count = min(512, frame_count - first)
                read_length = (count - 1) * hop + length
                offset = start + first * hop
                raw_handle.seek(offset)
                reference = raw_handle.read(read_length, dtype="float64", always_2d=True).mean(axis=1)
                disturbances = []
                for handle in expected_handles:
                    handle.seek(offset)
                    disturbances.append(handle.read(read_length, dtype="float64", always_2d=True).mean(axis=1))
                disturbance_array = np.stack(disturbances)
                x_frames = np.lib.stride_tricks.sliding_window_view(reference, length)[::hop][:count]
                d_frames = np.lib.stride_tricks.sliding_window_view(disturbance_array, length, axis=-1)[..., ::hop, :][:, :count]
                x_tensor = torch.from_numpy(np.array(x_frames, dtype=np.float32, copy=True)).to(device)
                d_tensor = torch.from_numpy(np.array(d_frames, dtype=np.float32, copy=True)).to(device)
                spectrum_x = torch.fft.rfft(x_tensor * window_t, dim=-1)
                spectrum_d = torch.fft.rfft(d_tensor * window_t, dim=-1)
                cross += torch.sum(spectrum_d * spectrum_x.unsqueeze(0).conj(), dim=1)
                power += torch.sum(spectrum_x.real.square() + spectrum_x.imag.square(), dim=0)
        finally:
            for handle in expected_handles:
                handle.close()
    return cross.cpu().numpy().astype(np.complex128), power.cpu().numpy().astype(np.float64), frame_count


def build_innovation_templates(
    dataset_dir: str | Path,
    output_path: str | Path,
    *,
    train_noises: tuple[str, ...] = DEFAULT_TRAIN_NOISES,
    path_indices: tuple[int, ...] = tuple(range(8)),
    start_seconds: float = 20.0,
    n_fft: int = 4096,
    hop_length: int = 240,
    generator_device: str = "auto",
) -> dict:
    """Build P_i and S_i using only the explicitly supplied training split."""
    if tuple(path_indices) != tuple(range(8)):
        raise ValueError("Phase 3R template generation is restricted to paths 1-8.")
    root = Path(dataset_dir)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    raw_dir, expected_dir = root / "NOISE", root / "EXPECTED_NOISE"
    sources = [raw_dir / name for name in train_noises]
    missing = [str(path) for path in sources if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing Phase-3R training noises: {missing}")
    secondary = np.load(root / "sh.npy", allow_pickle=False).T.astype(np.float64)
    if secondary.shape[0] < 8:
        raise ValueError("sh.npy contains fewer than eight secondary paths.")

    window = np.hanning(n_fft).astype(np.float64)
    frequencies = np.fft.rfftfreq(n_fft, 1.0 / SAMPLE_RATE)
    band_mask = (frequencies >= 50.0) & (frequencies <= 8000.0)
    start = int(round(start_seconds * SAMPLE_RATE))
    device = torch.device("cuda" if generator_device == "auto" and torch.cuda.is_available() else
                          "cpu" if generator_device == "auto" else generator_device)
    cross_total = np.zeros((8, n_fft // 2 + 1), dtype=np.complex128)
    power_total = np.zeros(n_fft // 2 + 1, dtype=np.float64)
    frame_counts: dict[str, int] = {}
    input_files: list[Path] = [root / "sh.npy"]
    for raw in sources:
        expected_paths = [_resolve_expected(expected_dir, raw, path_index) for path_index in path_indices]
        available = min([sf.info(str(raw)).frames, *[sf.info(str(path)).frames for path in expected_paths]]) - start
        cross, power, count = _accumulate_multi_path_files(
            raw, expected_paths, start, available, window, hop_length, device,
        )
        cross_total += cross
        power_total += power
        for path_index, expected in zip(path_indices, expected_paths):
            frame_counts[f"{raw.name}:path_{path_index + 1}"] = count
            input_files.append(expected)
        input_files.append(raw)
    median_power = float(np.median(power_total[band_mask]))
    regularizer = max(1e-30, 1e-6 * median_power)
    primary = cross_total / (power_total[None, :] + regularizer)

    np.savez_compressed(
        output,
        primary_real=primary.real.astype(np.float32),
        primary_imag=primary.imag.astype(np.float32),
        secondary_paths=secondary[:8].astype(np.float32),
        frequencies_hz=frequencies.astype(np.float32),
        band_mask=band_mask,
        hann_window=window.astype(np.float32),
    )
    unique_inputs = sorted(set(input_files), key=lambda value: str(value).casefold())
    manifest = {
        "manifest_version": 1,
        "split": "six_training_noises_paths_1_to_8",
        "sample_rate": SAMPLE_RATE,
        "start_seconds": start_seconds,
        "sample_range": "20 seconds to end of each aligned pair",
        "n_fft": n_fft,
        "hop_length": hop_length,
        "center": False,
        "window": "symmetric Hann (numpy.hanning)",
        "frequency_band_hz": [50.0, 8000.0],
        "regularization": "1e-6 * median in-band accumulated reference power",
        "generator_device": str(device),
        "train_noises": [path.name for path in sources],
        "path_indices_zero_based": list(path_indices),
        "frame_counts": frame_counts,
        "input_sha256": {str(path.relative_to(root)): sha256_file(path) for path in unique_inputs},
        "artifact": output.name,
        "artifact_sha256": sha256_file(output),
    }
    manifest_path = output.with_suffix(".manifest.json")
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


def validate_template_artifact(path: str | Path) -> dict:
    """Fail closed if the fixed artifact includes forbidden data or is modified."""
    artifact = Path(path)
    manifest_path = artifact.with_suffix(".manifest.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if len(manifest["train_noises"]) != 6:
        raise ValueError("Phase-3R templates must contain exactly six training noises.")
    if manifest["path_indices_zero_based"] != list(range(8)):
        raise ValueError("Phase-3R templates must contain paths 1-8 only.")
    forbidden = ("scene_09", "scene_10")
    if any(token in name for name in manifest["input_sha256"] for token in forbidden):
        raise ValueError("Template manifest touched sealed paths 9/10.")
    if manifest["artifact_sha256"] != sha256_file(artifact):
        raise ValueError("Template SHA-256 does not match its manifest.")
    with np.load(artifact, allow_pickle=False) as values:
        if values["primary_real"].shape[0] != 8 or values["secondary_paths"].shape[0] != 8:
            raise ValueError("Template artifact does not contain exactly eight candidates.")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", default="dataset")
    parser.add_argument("--output", default="artifacts/phase3r_innovation_templates.npz")
    parser.add_argument("--phase3-config", default="runs/phase3_suite_seed2026_v2/P3-E1/config.json")
    args = parser.parse_args()
    config_path = Path(args.phase3_config)
    noises = DEFAULT_TRAIN_NOISES
    if config_path.is_file():
        noises = tuple(json.loads(config_path.read_text(encoding="utf-8"))["train_noises"])
    print(json.dumps(build_innovation_templates(args.dataset_dir, args.output, train_noises=noises),
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
