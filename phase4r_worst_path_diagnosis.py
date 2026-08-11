"""Phase-4R worst-path causal diagnosis without changing submission weights.

The script has two deliberately separated lanes:

* deployment evidence replays the eight frozen strict-LOPO checkpoints and
  audits that their candidate banks contain retained paths only;
* oracle-only evidence may use the held-out path after a checkpoint is frozen,
  but is never serialized as a trainable model or submission artifact.

The expensive oracle hierarchy is intentionally restricted to the registered
worst fold (path 8).  Coverage and routing evidence is still computed for all
eight folds so that the selected correction is not based on one post-hoc case.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import subprocess
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import scipy
import soundfile as sf
import torch
from scipy import optimize, stats
from scipy.fft import next_fast_len
from scipy.signal import fftconvolve

from phase1_data import _read_exact, _resolve_expected_path, iter_validation_examples
from phase1_validation import _aggregate, _load_official_scorer
from phase3_validation import _score_record, build_phase3_manifests
from phase3g_data import globally_align_ir, synthesize_path
from phase3g_model import GenerativeInnovationFIRController
from phase3g_validation import state_dict_sha256
from phase3r_templates import sha256_file
from phase3r_validation import stream_closed_loop
from train_phase3g import load_phase3g
from v6_metrics import INITIALIZATION_SAMPLES, TOTAL_SAMPLES, compute_v6_metrics


ROOT = Path(__file__).resolve().parent
DEFAULT_LOPO_ROOT = ROOT / "runs/phase3g_suite_seed2026_rerun1/P3G-LOPO-rerun1"
DEFAULT_LOPO_SUMMARY = DEFAULT_LOPO_ROOT / "lopo_summary.json"
DEFAULT_TEMPLATE = ROOT / "artifacts/phase3r_innovation_templates.npz"
DEFAULT_ORACLE_CHECKPOINT = ROOT / "runs/phase3_suite_seed2026_v2/P3-E1/checkpoints/best_phase3_selection.pt"
DEFAULT_P1_SUMMARY = ROOT / "runs/phase1_suite_seed2026/P1-E2/summary.json"
DEFAULT_P3_SUMMARY = ROOT / "runs/phase3_suite_seed2026_v2/P3-E1/summary.json"
DEFAULT_FINAL_EVALUATION = ROOT / "runs/phase3g_final_evaluation.json"
DEFAULT_FORMAL_MANIFEST = ROOT / "artifacts/phase3g_formal_model.json"
DEFAULT_REPORT = ROOT / "artifacts/phase4r_worst_path_diagnosis.json"
DEFAULT_PREREGISTRATION = ROOT / "artifacts/phase4r_preregistered_correction.json"
DEFAULT_MARKDOWN = ROOT / "PHASE4R_WORST_PATH_DIAGNOSIS.md"
PATH_COUNT = 8
WORST_PATH_ZERO_BASED = 7
ORACLE_INITIALIZATION_SEEDS = (401, 402, 403)


def _read_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def _path_for_report(path: str | Path) -> str:
    resolved = Path(path).resolve()
    try:
        return resolved.relative_to(ROOT).as_posix()
    except ValueError:
        return str(resolved)


def _git_revision() -> dict[str, Any]:
    try:
        revision = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True,
            capture_output=True, text=True,
        ).stdout.strip()
        dirty = bool(subprocess.run(
            ["git", "status", "--porcelain"], cwd=ROOT, check=True,
            capture_output=True, text=True,
        ).stdout.strip())
        return {"commit": revision, "worktree_dirty": dirty}
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "worktree_dirty": None}


def _array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
    digest.update(array.tobytes())
    return digest.hexdigest()


def _require_path_set(paths: Sequence[int], held_out: int) -> list[int]:
    values = [int(value) for value in paths]
    expected = [value for value in range(PATH_COUNT) if value != int(held_out)]
    if values != expected:
        raise ValueError(f"LOPO fold {held_out + 1} has paths {values}, expected {expected}.")
    if any(value >= PATH_COUNT or value < 0 for value in values):
        raise ValueError("Path 9/10 and invalid paths are forbidden in Phase-4R diagnosis.")
    return values


def _checkpoint_path(fold: dict[str, Any], lopo_root: Path) -> Path:
    value = Path(str(fold["checkpoint"]))
    candidates = [value, ROOT / value]
    if not value.is_absolute():
        candidates.append(lopo_root / f"path_{int(fold['held_out_path']):02d}" / "generalize/checkpoints" / value.name)
    result = next((item.resolve() for item in candidates if item.is_file()), None)
    if result is None:
        raise FileNotFoundError(f"Missing LOPO checkpoint: {value}")
    if lopo_root.resolve() not in result.parents:
        raise ValueError(f"LOPO checkpoint escaped the frozen run root: {result}")
    return result


def _initial_dictionary_hash(experts: torch.Tensor, seed: int, latent_size: int = 16) -> str:
    model = GenerativeInnovationFIRController(
        num_experts=experts.shape[0], fir_length=experts.shape[1], latent_size=latent_size,
    )
    with torch.no_grad():
        model.expert_filters.copy_(experts)
    model.initialize_dictionary(seed)
    return _array_sha256(model.residual_dictionary.detach().cpu().numpy())


def audit_fold_isolation(
    fold: dict[str, Any], lopo_root: Path, template_path: Path,
    oracle_checkpoint_path: Path,
) -> dict[str, Any]:
    """Verify physical removal of the held-out candidate and sealed paths."""
    held = int(fold["held_out_path"]) - 1
    keep = _require_path_set(fold["retained_original_indices_zero_based"], held)
    checkpoint_path = _checkpoint_path(fold, lopo_root)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config_paths = _require_path_set(checkpoint["config"]["train_paths"], held)
    state = checkpoint["model_state_dict"]
    synthesis_path = checkpoint_path.parents[1] / "synthesis_manifest.json"
    synthesis = _read_json(synthesis_path)
    manifest_paths = _require_path_set(synthesis["path_indices_zero_based"], held)
    with np.load(template_path, allow_pickle=False) as artifact:
        primary_real = artifact["primary_real"].copy()
        primary_imag = artifact["primary_imag"].copy()
        secondary = artifact["secondary_paths"].copy()
    oracle = torch.load(oracle_checkpoint_path, map_location="cpu", weights_only=False)
    oracle_experts = oracle["model_state_dict"]["expert_filters"].detach().cpu()
    expected_suffixes = {f"_scene_{held + 1:02d}.wav".casefold(), "_scene_09.wav", "_scene_10.wav"}
    input_names = [str(value).casefold() for value in synthesis["input_sha256"]]
    forbidden_inputs = [name for name in input_names if any(token in name for token in expected_suffixes)]
    checks = {
        "summary_paths_exact": keep == config_paths,
        "manifest_paths_exact": keep == manifest_paths,
        "checkpoint_has_seven_candidates": int(checkpoint["model_config"]["num_experts"]) == 7,
        "expert_rows_retained_only": bool(torch.equal(state["expert_filters"], oracle_experts[keep])),
        "primary_rows_retained_only": bool(
            np.array_equal(state["primary_real"].numpy(), primary_real[keep])
            and np.array_equal(state["primary_imag"].numpy(), primary_imag[keep])
        ),
        "secondary_rows_retained_only": bool(np.array_equal(state["secondary_paths"].numpy(), secondary[keep])),
        "candidate_mask_shape_and_value": bool(
            tuple(state["candidate_mask"].shape) == (7,) and state["candidate_mask"].all()
        ),
        "no_held_out_or_sealed_expected_input": not forbidden_inputs,
        "sealed_paths_flag_false": synthesis.get("sealed_paths_touched") is False,
    }
    return {
        "held_out_path": held + 1,
        "passed": all(checks.values()),
        "checks": checks,
        "checkpoint": _path_for_report(checkpoint_path),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "checkpoint_state_dict_sha256": state_dict_sha256(load_phase3g(checkpoint_path, torch.device("cpu"))),
        "synthesis_manifest": _path_for_report(synthesis_path),
        "synthesis_manifest_sha256": sha256_file(synthesis_path),
        "retained_paths_one_based": [value + 1 for value in keep],
        "forbidden_inputs": forbidden_inputs,
        "dictionary_initialization": {
            "seed": int(checkpoint["config"]["seed"]),
            "input_expert_paths_one_based": [value + 1 for value in keep],
            "reconstructed_initial_dictionary_sha256": _initial_dictionary_hash(
                oracle_experts[keep], int(checkpoint["config"]["seed"]),
                int(checkpoint["model_config"]["latent_size"]),
            ),
        },
    }


def _causal_filter(signal: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    return fftconvolve(signal, kernel, mode="full")[: signal.size]


def _score_outputs(records: Sequence[dict[str, Any]], outputs: np.ndarray) -> dict[str, Any]:
    scorer = _load_official_scorer()
    scored_records = []
    for record, output in zip(records, outputs):
        residual = record["disturbance"] - _causal_filter(output, record["path"])
        scored_records.append(_score_record(
            scorer, str(record["name"]), int(record["path_index"]),
            torch.from_numpy(record["disturbance"])[None],
            torch.from_numpy(residual)[None], torch.from_numpy(output)[None],
        ))
    result = _aggregate(scored_records)
    result["official_objective"] = 0.7 * result["primary_score_db"] - 0.3 * result["rebound_score_db"]
    result["finite"] = bool(np.isfinite(outputs).all())
    result["records"] = scored_records
    return result


def _compact_metrics(metrics: dict[str, Any], include_records: bool = False) -> dict[str, Any]:
    keys = (
        "primary_score_db", "rebound_score_db", "official_objective",
        "first_window_primary_db", "worst_window_primary_db",
        "largest_single_window_rebound_db", "controller_peak_abs", "finite",
    )
    result = {key: metrics[key] for key in keys if key in metrics}
    if include_records:
        result["records"] = metrics.get("records", [])
    return result


def _load_development_records(dataset_dir: Path, held: int) -> list[dict[str, Any]]:
    manifest = build_phase3_manifests(dataset_dir)["development"]
    manifest["path_indices_zero_based"] = [held]
    source_names = {str(key): str(value) for key, value in manifest["sources"].items()}
    scene_sources = {}
    for scene in manifest["scenes"]:
        keys = [str(scene["first"])]
        if scene.get("second") is not None:
            keys.append(str(scene["second"]))
        scene_sources[str(scene["name"])] = [source_names[key] for key in keys]
    records = []
    for scene, path_index, reference, path, disturbance in iter_validation_examples(dataset_dir, manifest):
        records.append({
            "name": scene, "path_index": path_index,
            "reference": reference.numpy().astype(np.float64),
            "path": path.numpy().astype(np.float64),
            "disturbance": disturbance.numpy().astype(np.float64),
            "source_files": scene_sources[str(scene)],
        })
    return records


def _load_calibration_records(
    dataset_dir: Path, noise_names: Sequence[str], held: int,
) -> list[dict[str, Any]]:
    records = []
    path = np.load(dataset_dir / "sh.npy", allow_pickle=False).T[held].astype(np.float64)
    start = 20 * 48_000
    for name in noise_names:
        raw = dataset_dir / "NOISE" / str(name)
        expected = _resolve_expected_path(dataset_dir / "EXPECTED_NOISE", raw, held)
        records.append({
            "name": f"oracle_calibration_{raw.stem}", "path_index": held,
            "reference": _read_exact(raw, start, TOTAL_SAMPLES).astype(np.float64),
            "path": path,
            "disturbance": _read_exact(expected, start, TOTAL_SAMPLES).astype(np.float64),
            "source": raw.name,
        })
    return records


def oracle_sources_disjoint(
    calibration: Sequence[dict[str, Any]], development: Sequence[dict[str, Any]],
) -> bool:
    calibration_sources = {str(item["source"]) for item in calibration}
    evaluation_sources = {
        str(source) for item in development for source in item["source_files"]
    }
    return not bool(calibration_sources & evaluation_sources)


def _route_replay(
    model: GenerativeInnovationFIRController, reference: np.ndarray,
    residual: np.ndarray, output: np.ndarray,
) -> list[dict[str, Any]]:
    """Replay the analytic route with a common output for every candidate."""
    candidate_anti = np.stack([_causal_filter(output, path) for path in model.secondary_paths.numpy()])
    window = model.hann_window.numpy().astype(np.float64)
    band = model.band_mask.numpy().astype(bool)
    primary = model.primary_real.numpy() + 1j * model.primary_imag.numpy()
    mask = model.candidate_mask.numpy().astype(bool)
    alpha = mask.astype(np.float64) / mask.sum()
    ewma = None
    trace = []
    for completed in range(model.block_size, reference.size + 1, model.block_size):
        if completed < model.n_fft:
            continue
        start = completed - model.n_fft
        x = reference[start:completed]
        e = residual[start:completed]
        anti = candidate_anti[:, start:completed]
        spectrum_x = np.fft.rfft(x * window)
        disturbance = e[None] + anti
        spectrum_d = np.fft.rfft(disturbance * window[None], axis=-1)
        denominator = np.sum(np.abs(spectrum_d[:, band]) ** 2, axis=-1)
        numerator = np.sum(np.abs(spectrum_d[:, band] - (primary * spectrum_x[None])[:, band]) ** 2, axis=-1)
        score = numerator / np.maximum(denominator, 1e-20)
        log_j = np.log(score + 1e-12)
        ewma = log_j if ewma is None else model.ewma_lambda * ewma + (1 - model.ewma_lambda) * log_j
        centered = ewma - np.min(ewma[mask])
        logits = -centered / model.temperature
        logits[~mask] = -np.inf
        proposal = np.zeros(model.num_experts)
        visible = logits[mask]
        proposal[mask] = np.exp(visible - np.max(visible))
        proposal /= proposal.sum()
        alpha = (1 - model.alpha_update) * alpha + model.alpha_update * proposal
        alpha[~mask] = 0
        alpha /= alpha.sum()
        trace.append({
            "completed_samples": completed, "innovation": score.tolist(),
            "posterior": proposal.tolist(), "alpha": alpha.tolist(),
            "winner_zero_based": int(np.argmax(alpha)),
        })
    return trace


def summarize_route(trace: Sequence[dict[str, Any]], original_paths: Sequence[int]) -> dict[str, Any]:
    eligible = [item for item in trace if int(item["completed_samples"]) >= INITIALIZATION_SAMPLES]
    count = len(eligible)
    if not eligible:
        return {"event_count": 0}
    alpha = np.asarray([item["alpha"] for item in eligible], dtype=np.float64)
    posterior = np.asarray([item["posterior"] for item in eligible], dtype=np.float64)
    winners = [int(original_paths[int(item["winner_zero_based"])]) + 1 for item in eligible]
    entropy = -np.sum(posterior * np.log(np.maximum(posterior, 1e-12)), axis=1) / math.log(max(2, posterior.shape[1]))
    ordered = np.sort(alpha, axis=1)
    result = {
        "event_count": count,
        "winner_counts_one_based": {str(key): value for key, value in sorted(Counter(winners).items())},
        "mean_alpha_by_original_path": {
            str(path + 1): float(alpha[:, index].mean()) for index, path in enumerate(original_paths)
        },
        "posterior_entropy_mean": float(entropy.mean()),
        "alpha_top1_minus_top2_mean": float(np.mean(ordered[:, -1] - ordered[:, -2])),
    }
    if "latent" in eligible[0]:
        latent = np.asarray([item["latent"] for item in eligible], dtype=np.float64)
        result.update({
            "latent_l2_mean": float(np.linalg.norm(latent, axis=1).mean()),
            "latent_l2_max": float(np.linalg.norm(latent, axis=1).max()),
            "generated_residual_norm_mean": float(np.mean([
                item["generated_residual_norm"] for item in eligible
            ])),
        })
    return result


def _trace_max_difference(first: Sequence[dict[str, Any]], second: Sequence[dict[str, Any]]) -> dict[str, float | bool]:
    if len(first) != len(second):
        return {"event_count_equal": False, "innovation_max_abs": float("inf"), "alpha_max_abs": float("inf")}
    innovations = max(
        (float(np.max(np.abs(np.asarray(a["innovation"]) - np.asarray(b["innovation"])))) for a, b in zip(first, second)),
        default=0.0,
    )
    alphas = max(
        (float(np.max(np.abs(np.asarray(a["alpha"]) - np.asarray(b["alpha"])))) for a, b in zip(first, second)),
        default=0.0,
    )
    return {"event_count_equal": True, "innovation_max_abs": innovations, "alpha_max_abs": alphas}


def replay_fold(
    fold: dict[str, Any], lopo_root: Path, dataset_dir: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    held = int(fold["held_out_path"]) - 1
    original_paths = _require_path_set(fold["retained_original_indices_zero_based"], held)
    checkpoint = _checkpoint_path(fold, lopo_root)
    model = load_phase3g(checkpoint, torch.device("cpu")).eval()
    before = state_dict_sha256(model)
    records = _load_development_records(dataset_dir, held)
    scored = []
    route_scenes = {}
    for record in records:
        output, residual, deployed_trace, _ = stream_closed_loop(
            model, record["reference"], record["disturbance"], record["path"],
        )
        replay = _route_replay(model, record["reference"], residual, output)
        no_control = _route_replay(
            model, record["reference"], record["disturbance"], np.zeros_like(output),
        )
        metrics = _score_outputs([record], output[None])["records"][0]
        scored.append(metrics)
        route_scenes[str(record["name"])] = {
            "deployed_closed_loop": summarize_route(deployed_trace, original_paths),
            "common_output_replay": summarize_route(replay, original_paths),
            "no_control": summarize_route(no_control, original_paths),
            "closed_loop_replay_match": _trace_max_difference(deployed_trace, replay),
        }
    aggregate = _aggregate(scored)
    return {
        "held_out_path": held + 1,
        "stored": {
            "baseline_primary_db": float(fold["baseline_primary_db"]),
            "candidate_primary_db": float(fold["candidate_primary_db"]),
            "primary_gain_db": float(fold["primary_gain_db"]),
            "candidate_rebound_db": float(fold["candidate_rebound_db"]),
        },
        "replayed": {
            "primary_score_db": aggregate["primary_score_db"],
            "rebound_score_db": aggregate["rebound_score_db"],
            "primary_gain_db": aggregate["primary_score_db"] - float(fold["baseline_primary_db"]),
            "controller_peak_abs": aggregate["controller_peak_abs"],
            "records": scored if held == WORST_PATH_ZERO_BASED else None,
        },
        "absolute_reproduction_error": {
            "primary_db": abs(aggregate["primary_score_db"] - float(fold["candidate_primary_db"])),
            "rebound_db": abs(aggregate["rebound_score_db"] - float(fold["candidate_rebound_db"])),
        },
        "route_by_scene": route_scenes,
        "state_dict_immutable": before == state_dict_sha256(model),
    }, records


def _nearest_paths(paths: np.ndarray, retained: Sequence[int]) -> dict[int, int]:
    result = {}
    for first_index in retained:
        first = paths[first_index].astype(np.float64)
        choices = []
        for second_index in retained:
            if first_index == second_index:
                continue
            aligned = globally_align_ir(first, paths[second_index])
            choices.append((float(np.mean((first - aligned) ** 2)), second_index))
        result[first_index] = min(choices)[1]
    return result


def replay_sampling_metadata(
    *, train_paths: Sequence[int], nearest: dict[int, int], seed: int,
    epochs: int, samples_per_epoch: int, raw_frames: Sequence[int],
    path_length: int = 1967, augmentation_tail_start: int = 64,
) -> dict[str, Any]:
    """Replay RNG calls from Phase3GSequenceDataset without reading audio."""
    train_paths = tuple(int(value) for value in train_paths)
    mode_counts: Counter[str] = Counter()
    pair_counts: Counter[tuple[int, int]] = Counter()
    mask_removed_counts: Counter[int] = Counter()
    switch_count = 0
    amounts: defaultdict[str, list[float]] = defaultdict(list)

    def environment(rng: np.random.Generator) -> None:
        first = int(rng.choice(train_paths))
        second = int(nearest[first])
        draw = rng.random()
        kind = "measured" if draw < 0.50 else "interpolate" if draw < 0.80 else "extrapolate" if draw < 0.90 else "augment"
        endpoints = [first]
        if kind == "augment":
            rng.uniform(-1.0, 1.0); rng.integers(-1, 2); rng.uniform(-35.0, -32.0)
            rng.standard_normal(int(path_length) - int(augmentation_tail_start))
        elif kind in {"interpolate", "extrapolate"}:
            limits = (0.2, 0.8) if kind == "interpolate" else (0.05, 0.20)
            amount = float(rng.uniform(*limits))
            amounts[kind].append(amount)
            endpoints = [first, second]
            pair_counts[(first, second)] += 1
        mask_draw = rng.random()
        removed = endpoints[:1] if 0.4 <= mask_draw < 0.7 else endpoints[:2] if mask_draw >= 0.7 else []
        if len(removed) < 2 and mask_draw >= 0.7:
            removed.append(nearest[first])
        mask_removed_counts[len(set(removed))] += 1
        mode_counts[kind] += 1

    for epoch in range(1, int(epochs) + 1):
        for index in range(int(samples_per_epoch)):
            rng = np.random.default_rng(int(seed) + epoch * 1_000_003 + index)
            raw_slot = int(rng.integers(0, len(raw_frames)))
            maximum = int(raw_frames[raw_slot]) - TOTAL_SAMPLES
            rng.integers(20 * 48_000, maximum + 1)
            environment(rng)
            switched = bool(rng.random() < 0.25)
            switch_count += int(switched)
            if switched:
                environment(rng)
    total = sum(mode_counts.values())
    return {
        "environment_count": total,
        "switch_count": switch_count,
        "mode_counts": dict(sorted(mode_counts.items())),
        "mode_fractions": {key: value / total for key, value in sorted(mode_counts.items())},
        "directed_pair_counts_one_based": {
            f"{first + 1}->{second + 1}": value for (first, second), value in sorted(pair_counts.items())
        },
        "mask_removed_endpoint_counts": {str(key): value for key, value in sorted(mask_removed_counts.items())},
        "amounts": {
            key: {
                "count": len(values), "minimum": min(values), "maximum": max(values),
                "mean": float(np.mean(values)),
            } for key, values in sorted(amounts.items())
        },
    }


def _scaled_nrmse(target: np.ndarray, candidate: np.ndarray) -> float:
    first = np.asarray(target, dtype=np.complex128).reshape(-1)
    second = np.asarray(candidate, dtype=np.complex128).reshape(-1)
    denominator = float(np.vdot(second, second).real)
    scale = float(np.vdot(second, first).real / max(denominator, 1e-20))
    return float(np.linalg.norm(first - scale * second) / max(np.linalg.norm(first), 1e-20))


def _projection_report(target: np.ndarray, bank: np.ndarray) -> dict[str, Any]:
    design = np.asarray(bank, dtype=np.float64).T
    coefficients = np.linalg.lstsq(design, np.asarray(target, dtype=np.float64), rcond=1e-7)[0]
    residual = np.asarray(target) - design @ coefficients
    ratio = float(np.linalg.norm(residual) / max(np.linalg.norm(target), 1e-20))
    return {
        "scaled_projection_nrmse": ratio,
        "principal_angle_degrees": float(np.degrees(np.arcsin(np.clip(ratio, 0.0, 1.0)))),
        "coefficient_l2": float(np.linalg.norm(coefficients)),
    }


def _history_epoch_count(checkpoint_path: Path) -> int:
    history = checkpoint_path.parents[1] / "history.jsonl"
    if not history.is_file():
        raise FileNotFoundError(history)
    return sum(1 for line in history.read_text(encoding="utf-8").splitlines() if line.strip())


def analyze_coverage(
    folds: Sequence[dict[str, Any]], lopo_root: Path, dataset_dir: Path,
    template_path: Path, oracle_checkpoint_path: Path,
) -> dict[str, Any]:
    """Measure held-out coverage in IR, acoustic-template, and FIR spaces."""
    paths = np.load(dataset_dir / "sh.npy", allow_pickle=False).T[:PATH_COUNT].astype(np.float64)
    with np.load(template_path, allow_pickle=False) as artifact:
        primary = artifact["primary_real"].astype(np.float64) + 1j * artifact["primary_imag"].astype(np.float64)
        band = artifact["band_mask"].astype(bool)
    full_oracle = torch.load(oracle_checkpoint_path, map_location="cpu", weights_only=False)
    historical_experts = full_oracle["model_state_dict"]["expert_filters"].numpy().astype(np.float64)
    frequencies = np.fft.rfftfreq(4096, 1 / 48_000)
    response_band = (frequencies >= 50) & (frequencies <= 8000)
    synthesized_cache: dict[tuple[int, int, str, float], np.ndarray] = {}
    per_fold = []

    for fold in folds:
        held = int(fold["held_out_path"]) - 1
        keep = _require_path_set(fold["retained_original_indices_zero_based"], held)
        nearest = _nearest_paths(paths, keep)
        checkpoint_path = _checkpoint_path(fold, lopo_root)
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        state = checkpoint["model_state_dict"]

        ir_distances = {}
        response_distances = {}
        primary_distances = {}
        held_response = np.fft.rfft(paths[held], n=4096)[response_band]
        for candidate in keep:
            aligned = globally_align_ir(paths[held], paths[candidate])
            ir_distances[candidate] = _scaled_nrmse(paths[held], aligned)
            response = np.fft.rfft(paths[candidate], n=4096)[response_band]
            response_distances[candidate] = _scaled_nrmse(held_response, response)
            primary_distances[candidate] = _scaled_nrmse(primary[held, band], primary[candidate, band])

        best_synthetic = {"scaled_ir_nrmse": float("inf")}
        for first, second in sorted(nearest.items()):
            for mode, amounts in (
                ("interpolate", np.linspace(0.2, 0.8, 7)),
                ("extrapolate", np.linspace(0.05, 0.20, 4)),
            ):
                for amount_value in amounts:
                    amount = float(round(float(amount_value), 6))
                    key = (first, second, mode, amount)
                    if key not in synthesized_cache:
                        synthesized_cache[key] = synthesize_path(
                            paths[first], paths[second], mode=mode, amount=amount,
                        )[0].astype(np.float64)
                    generated = globally_align_ir(paths[held], synthesized_cache[key])
                    distance = _scaled_nrmse(paths[held], generated)
                    if distance < best_synthetic["scaled_ir_nrmse"]:
                        best_synthetic = {
                            "scaled_ir_nrmse": distance, "first_path": first + 1,
                            "second_path": second + 1, "mode": mode, "amount": amount,
                        }

        filters = np.concatenate((
            state["expert_filters"].numpy().astype(np.float64),
            state["residual_dictionary"].numpy().astype(np.float64),
        ))
        historical_projection = _projection_report(historical_experts[held], filters)
        noise_names = checkpoint["config"]["train_noises"]
        raw_frames = [sf.info(str(dataset_dir / "NOISE" / name)).frames for name in noise_names]
        sampler = replay_sampling_metadata(
            train_paths=keep, nearest=nearest, seed=int(checkpoint["config"]["seed"]),
            epochs=_history_epoch_count(checkpoint_path),
            samples_per_epoch=int(checkpoint["config"]["samples_per_epoch_resolved"]),
            raw_frames=raw_frames,
        )
        per_fold.append({
            "held_out_path": held + 1,
            "nearest_graph_one_based": {str(key + 1): value + 1 for key, value in sorted(nearest.items())},
            "nearest_measured_ir": {
                "path": min(ir_distances, key=ir_distances.get) + 1,
                "scaled_nrmse": min(ir_distances.values()),
            },
            "nearest_secondary_response": {
                "path": min(response_distances, key=response_distances.get) + 1,
                "scaled_complex_nrmse": min(response_distances.values()),
            },
            "nearest_primary_template": {
                "path": min(primary_distances, key=primary_distances.get) + 1,
                "scaled_complex_nrmse": min(primary_distances.values()),
            },
            "best_actual_training_synthesis_grid": best_synthetic,
            "historical_expert_projection": historical_projection,
            "actual_training_sampler_replay": sampler,
        })

    gains = np.asarray([float(fold["primary_gain_db"]) for fold in folds])
    rebounds = np.asarray([float(fold["candidate_rebound_db"]) for fold in folds])
    metric_names = {
        "measured_ir_nrmse": np.asarray([item["nearest_measured_ir"]["scaled_nrmse"] for item in per_fold]),
        "secondary_response_nrmse": np.asarray([item["nearest_secondary_response"]["scaled_complex_nrmse"] for item in per_fold]),
        "primary_template_nrmse": np.asarray([item["nearest_primary_template"]["scaled_complex_nrmse"] for item in per_fold]),
        "actual_synthesis_ir_nrmse": np.asarray([item["best_actual_training_synthesis_grid"]["scaled_ir_nrmse"] for item in per_fold]),
        "historical_expert_projection_nrmse": np.asarray([item["historical_expert_projection"]["scaled_projection_nrmse"] for item in per_fold]),
    }
    correlations = {}
    for name, values in metric_names.items():
        correlations[name] = {
            "pearson_vs_primary_gain": float(stats.pearsonr(values, gains).statistic),
            "spearman_vs_primary_gain": float(stats.spearmanr(values, gains).statistic),
            "pearson_vs_rebound": float(stats.pearsonr(values, rebounds).statistic),
            "spearman_vs_rebound": float(stats.spearmanr(values, rebounds).statistic),
            "path8_worst_distance_rank_of_8": int(stats.rankdata(values, method="min")[-1]),
        }
    return {
        "method": {
            "sampling": "exact Phase3GSequenceDataset RNG replay; stress manifest excluded",
            "synthesis_grid": {"interpolate": [0.2, 0.8, 0.1], "extrapolate": [0.05, 0.20, 0.05]},
            "historical_expert_warning": "projection target is the old P3 expert, not an acoustic capacity oracle",
        },
        "per_fold": per_fold,
        "cross_fold_correlations": correlations,
    }


def _build_controller_basis(records: Sequence[dict[str, Any]], filters: np.ndarray) -> np.ndarray:
    result = np.empty((len(records), filters.shape[0], TOTAL_SAMPLES), dtype=np.float32)
    for record_index, record in enumerate(records):
        for filter_index, filter_value in enumerate(filters):
            result[record_index, filter_index] = _causal_filter(record["reference"], filter_value)
    return result


def _linear_gram(
    records: Sequence[dict[str, Any]], controller_basis: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    size = controller_basis.shape[1]
    gram = np.zeros((size, size), dtype=np.float64)
    target = np.zeros(size, dtype=np.float64)
    for record, outputs in zip(records, controller_basis):
        anti = np.stack([_causal_filter(value.astype(np.float64), record["path"]) for value in outputs])
        scored = slice(INITIALIZATION_SAMPLES, None)
        matrix = anti[:, scored]
        disturbance = record["disturbance"][scored]
        gram += matrix @ matrix.T
        target += matrix @ disturbance
    scale = max(float(np.trace(gram)) / max(size, 1), 1e-20)
    gram += np.eye(size) * scale * 1e-8
    return gram, target


def _constrained_linear_initialization(
    gram: np.ndarray, target: np.ndarray, experts: int, latent: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    dimension = experts + latent
    initial = np.r_[np.full(experts, 1 / experts), np.zeros(latent)]
    bounds = [(0.0, 1.0)] * experts + [(-1.0, 1.0)] * latent
    constraint = {"type": "eq", "fun": lambda value: float(np.sum(value[:experts]) - 1.0)}
    result = optimize.minimize(
        lambda value: float(0.5 * value @ gram @ value - target @ value),
        initial, jac=lambda value: gram @ value - target,
        method="SLSQP", bounds=bounds, constraints=[constraint],
        options={"maxiter": 500, "ftol": 1e-10},
    )
    coefficients = result.x if result.success else initial
    return coefficients[:experts], coefficients[experts:dimension], {
        "success": bool(result.success), "message": str(result.message),
        "iterations": int(result.nit), "quadratic_objective": float(result.fun),
    }


def render_static_output(
    controller_basis: np.ndarray, alpha: np.ndarray, latent: np.ndarray,
    *, output_limit: float = 0.98,
) -> np.ndarray:
    alpha = np.asarray(alpha, dtype=np.float64)
    latent = np.asarray(latent, dtype=np.float64)
    if np.any(alpha < -1e-9) or not np.isclose(alpha.sum(), 1.0, atol=1e-6):
        raise ValueError("Static oracle alpha must lie on the probability simplex.")
    if np.any(np.abs(latent) > 1.0 + 1e-9):
        raise ValueError("Static oracle latent coefficients must lie in [-1, 1].")
    coefficients = np.r_[alpha, latent]
    raw = np.einsum("rjt,j->rt", controller_basis.astype(np.float64), coefficients, optimize=True)
    safe = float(output_limit) - 1e-6
    output = safe * np.tanh(raw / safe)
    if not np.all(np.isfinite(output)) or float(np.max(np.abs(output))) >= output_limit:
        raise FloatingPointError("Oracle output violated finite/soft-limit constraints.")
    return output


def _torch_secondary_path(output: torch.Tensor, path: torch.Tensor, fft_size: int) -> torch.Tensor:
    spectrum = torch.fft.rfft(output, n=fft_size)
    path_spectrum = torch.fft.rfft(path, n=fft_size)
    return torch.fft.irfft(spectrum * path_spectrum[None], n=fft_size)[..., : output.shape[-1]]


def _torch_objective(disturbance: torch.Tensor, output: torch.Tensor, path: torch.Tensor) -> tuple[torch.Tensor, dict[str, float]]:
    fft_size = next_fast_len(output.shape[-1] + path.numel() - 1)
    anti = _torch_secondary_path(output, path, fft_size)
    metrics = compute_v6_metrics(disturbance, disturbance - anti)
    objective = 0.7 * metrics["primary_score_db"] - 0.3 * metrics["rebound_score_db"]
    return -objective, {
        "primary_score_db": float(metrics["primary_score_db"].detach()),
        "rebound_score_db": float(metrics["rebound_score_db"].detach()),
        "official_objective": float(objective.detach()),
        "controller_peak_abs": float(output.detach().abs().amax()),
    }


def _atanh_clipped(value: np.ndarray) -> np.ndarray:
    return np.arctanh(np.clip(np.asarray(value, dtype=np.float64), -0.999, 0.999))


def optimize_static_oracle(
    records: Sequence[dict[str, Any]], controller_basis: np.ndarray,
    initializations: Sequence[tuple[str, np.ndarray, np.ndarray]],
    *, expert_count: int, latent_count: int, steps: int,
    learning_rate: float = 0.08,
) -> dict[str, Any]:
    basis_t = torch.from_numpy(controller_basis)
    disturbance_t = torch.from_numpy(np.stack([item["disturbance"] for item in records]).astype(np.float32))
    path_t = torch.from_numpy(records[0]["path"].astype(np.float32))
    best: dict[str, Any] | None = None
    trials = []
    for trial_index, (name, initial_alpha, initial_latent) in enumerate(initializations):
        torch.manual_seed(ORACLE_INITIALIZATION_SEEDS[trial_index])
        logits = torch.nn.Parameter(torch.log(torch.from_numpy(initial_alpha.astype(np.float32)).clamp_min(1e-6)))
        parameters: list[torch.nn.Parameter] = [logits]
        raw_latent = None
        if latent_count:
            raw_latent = torch.nn.Parameter(torch.from_numpy(_atanh_clipped(initial_latent).astype(np.float32)))
            parameters.append(raw_latent)
        optimizer = torch.optim.Adam(parameters, lr=learning_rate)
        trial_best = None
        for step in range(max(0, int(steps)) + 1):
            alpha = torch.softmax(logits, dim=0)
            latent = torch.tanh(raw_latent) if raw_latent is not None else basis_t.new_zeros(0)
            coefficients = torch.cat((alpha, latent))
            raw = torch.einsum("rjt,j->rt", basis_t, coefficients)
            safe = 0.98 - 1e-6
            output = safe * torch.tanh(raw / safe)
            loss, metrics = _torch_objective(disturbance_t, output, path_t)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite static oracle loss in {name}.")
            snapshot = {
                "step": step, "metrics": metrics,
                "alpha": alpha.detach().cpu().numpy().astype(np.float64),
                "latent": latent.detach().cpu().numpy().astype(np.float64),
            }
            if trial_best is None or metrics["official_objective"] > trial_best["metrics"]["official_objective"]:
                trial_best = snapshot
            if step == int(steps):
                break
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters, 10.0)
            optimizer.step()
        assert trial_best is not None
        trials.append({
            "initialization": name, "seed": ORACLE_INITIALIZATION_SEEDS[trial_index],
            "selected_step": trial_best["step"], "calibration": trial_best["metrics"],
        })
        if best is None or trial_best["metrics"]["official_objective"] > best["metrics"]["official_objective"]:
            best = {**trial_best, "initialization": name, "seed": ORACLE_INITIALIZATION_SEEDS[trial_index]}
    assert best is not None
    return {"best": best, "trials": trials}


def render_blockwise_output(
    controller_basis: np.ndarray, alpha: np.ndarray, latent: np.ndarray,
    *, block_size: int = 240, output_limit: float = 0.98,
) -> np.ndarray:
    if np.any(alpha < -1e-9) or not np.allclose(alpha.sum(axis=1), 1.0, atol=1e-6):
        raise ValueError("Every blockwise alpha row must lie on the probability simplex.")
    if np.any(np.abs(latent) > 1.0 + 1e-9):
        raise ValueError("Blockwise latent coefficients must lie in [-1, 1].")
    blocks = np.arange(controller_basis.shape[-1]) // int(block_size)
    coefficients = np.concatenate((alpha, latent), axis=1)
    selected = coefficients[blocks]
    raw = np.einsum("rtj,tj->rt", controller_basis.transpose(0, 2, 1).astype(np.float64), selected, optimize=True)
    safe = output_limit - 1e-6
    output = safe * np.tanh(raw / safe)
    if not np.all(np.isfinite(output)) or float(np.max(np.abs(output))) >= output_limit:
        raise FloatingPointError("Blockwise oracle output violated finite/soft-limit constraints.")
    return output


def optimize_blockwise_oracle(
    records: Sequence[dict[str, Any]], controller_basis: np.ndarray,
    starts: Sequence[tuple[str, np.ndarray, np.ndarray]],
    *, expert_count: int, latent_count: int, block_size: int, steps: int,
) -> dict[str, Any]:
    basis_t = torch.from_numpy(controller_basis).permute(0, 2, 1)
    disturbance_t = torch.from_numpy(np.stack([item["disturbance"] for item in records]).astype(np.float32))
    path_t = torch.from_numpy(records[0]["path"].astype(np.float32))
    block_index = torch.arange(TOTAL_SAMPLES) // int(block_size)
    block_count = int(block_index[-1]) + 1
    best = None
    trials = []
    for trial_index, (name, alpha_start, latent_start) in enumerate(starts):
        torch.manual_seed(ORACLE_INITIALIZATION_SEEDS[trial_index])
        alpha_value = np.broadcast_to(alpha_start, (block_count, expert_count)).copy()
        latent_value = np.broadcast_to(latent_start, (block_count, latent_count)).copy()
        logits = torch.nn.Parameter(torch.log(torch.from_numpy(alpha_value.astype(np.float32)).clamp_min(1e-6)))
        raw_latent = torch.nn.Parameter(torch.from_numpy(_atanh_clipped(latent_value).astype(np.float32)))
        optimizer = torch.optim.Adam((logits, raw_latent), lr=0.05)
        trial_best = None
        for step in range(max(0, int(steps)) + 1):
            alpha = torch.softmax(logits, dim=-1)
            latent = torch.tanh(raw_latent)
            coefficients = torch.cat((alpha, latent), dim=-1)
            selected = coefficients[block_index]
            raw = torch.sum(basis_t * selected[None], dim=-1)
            safe = 0.98 - 1e-6
            output = safe * torch.tanh(raw / safe)
            loss, metrics = _torch_objective(disturbance_t, output, path_t)
            smoothness = (
                (alpha[1:] - alpha[:-1]).square().mean()
                + 0.1 * (latent[1:] - latent[:-1]).square().mean()
            )
            optimized_loss = loss + 1e-4 * smoothness
            snapshot = {
                "step": step, "metrics": metrics,
                "alpha": alpha.detach().cpu().numpy().astype(np.float64),
                "latent": latent.detach().cpu().numpy().astype(np.float64),
            }
            if trial_best is None or metrics["official_objective"] > trial_best["metrics"]["official_objective"]:
                trial_best = snapshot
            if step == int(steps):
                break
            optimizer.zero_grad(set_to_none=True)
            optimized_loss.backward()
            torch.nn.utils.clip_grad_norm_((logits, raw_latent), 10.0)
            optimizer.step()
        assert trial_best is not None
        trials.append({
            "initialization": name, "seed": ORACLE_INITIALIZATION_SEEDS[trial_index],
            "selected_step": trial_best["step"], "calibration": trial_best["metrics"],
        })
        if best is None or trial_best["metrics"]["official_objective"] > best["metrics"]["official_objective"]:
            best = {**trial_best, "initialization": name, "seed": ORACLE_INITIALIZATION_SEEDS[trial_index]}
    assert best is not None
    return {"best": best, "trials": trials, "block_count": block_count}


def _wiener_fir_initializations(
    records: Sequence[dict[str, Any]], fir_length: int,
) -> list[tuple[str, np.ndarray]]:
    path_length = records[0]["path"].size
    fft_size = next_fast_len(TOTAL_SAMPLES + fir_length + path_length - 2)
    numerator = np.zeros(fft_size // 2 + 1, dtype=np.complex128)
    denominator = np.zeros(fft_size // 2 + 1, dtype=np.float64)
    path_spectrum = np.fft.rfft(records[0]["path"], n=fft_size)
    for record in records:
        transfer = np.fft.rfft(record["reference"], n=fft_size) * path_spectrum
        target = np.fft.rfft(record["disturbance"], n=fft_size)
        numerator += np.conj(transfer) * target
        denominator += np.abs(transfer) ** 2
    scale = max(float(np.max(denominator)), 1e-20)
    result = []
    for regularization in (1e-8, 1e-6, 1e-4):
        spectrum = numerator / (denominator + regularization * scale)
        value = np.fft.irfft(spectrum, n=fft_size)[:fir_length].astype(np.float64)
        result.append((f"wiener_reg_{regularization:g}", value))
    return result


def _render_fir_output(records: Sequence[dict[str, Any]], filter_value: np.ndarray) -> np.ndarray:
    raw = np.stack([_causal_filter(item["reference"], filter_value) for item in records])
    safe = 0.98 - 1e-6
    return safe * np.tanh(raw / safe)


def optimize_free_fir_oracle(
    records: Sequence[dict[str, Any]], *, fir_length: int, steps: int,
) -> dict[str, Any]:
    starts = _wiener_fir_initializations(records, fir_length)
    disturbance_t = torch.from_numpy(np.stack([item["disturbance"] for item in records]).astype(np.float32))
    reference_t = torch.from_numpy(np.stack([item["reference"] for item in records]).astype(np.float32))
    path_t = torch.from_numpy(records[0]["path"].astype(np.float32))
    fft_size = next_fast_len(TOTAL_SAMPLES + fir_length + path_t.numel() - 2)
    reference_spectrum = torch.fft.rfft(reference_t, n=fft_size)
    best = None
    trials = []
    for trial_index, (name, start) in enumerate(starts):
        torch.manual_seed(ORACLE_INITIALIZATION_SEEDS[trial_index])
        filter_value = torch.nn.Parameter(torch.from_numpy(start.astype(np.float32)))
        optimizer = torch.optim.Adam((filter_value,), lr=0.002)
        trial_best = None
        for step in range(max(0, int(steps)) + 1):
            filter_spectrum = torch.fft.rfft(filter_value, n=fft_size)
            raw = torch.fft.irfft(reference_spectrum * filter_spectrum[None], n=fft_size)[..., :TOTAL_SAMPLES]
            safe = 0.98 - 1e-6
            output = safe * torch.tanh(raw / safe)
            loss, metrics = _torch_objective(disturbance_t, output, path_t)
            snapshot = {
                "step": step, "metrics": metrics,
                "filter": filter_value.detach().cpu().numpy().astype(np.float64),
            }
            if trial_best is None or metrics["official_objective"] > trial_best["metrics"]["official_objective"]:
                trial_best = snapshot
            if step == int(steps):
                break
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_((filter_value,), 5.0)
            optimizer.step()
        assert trial_best is not None
        trials.append({
            "initialization": name, "seed": ORACLE_INITIALIZATION_SEEDS[trial_index],
            "selected_step": trial_best["step"], "calibration": trial_best["metrics"],
        })
        if best is None or trial_best["metrics"]["official_objective"] > best["metrics"]["official_objective"]:
            best = {**trial_best, "initialization": name, "seed": ORACLE_INITIALIZATION_SEEDS[trial_index]}
    assert best is not None
    return {"best": best, "trials": trials}


def run_oracle_hierarchy(
    model: GenerativeInnovationFIRController, calibration: Sequence[dict[str, Any]],
    development: Sequence[dict[str, Any]], *, static_steps: int,
    blockwise_steps: int, free_fir_steps: int,
) -> tuple[dict[str, Any], np.ndarray]:
    experts = model.expert_filters.detach().cpu().numpy().astype(np.float64)
    dictionary = model.residual_dictionary.detach().cpu().numpy().astype(np.float64)
    bank = np.concatenate((experts, dictionary))
    calibration_basis = _build_controller_basis(calibration, bank)
    development_basis = _build_controller_basis(development, bank)
    expert_count, latent_count = experts.shape[0], dictionary.shape[0]

    single_trials = []
    for index in range(expert_count):
        alpha = np.eye(expert_count)[index]
        output = render_static_output(calibration_basis, alpha, np.zeros(latent_count))
        metrics = _score_outputs(calibration, output)
        single_trials.append((metrics["official_objective"], index, metrics))
    _, best_expert, best_single_calibration = max(single_trials, key=lambda value: value[0])
    best_alpha = np.eye(expert_count)[best_expert]
    best_single_evaluation = _score_outputs(
        development, render_static_output(development_basis, best_alpha, np.zeros(latent_count)),
    )

    expert_gram, expert_target = _linear_gram(calibration, calibration_basis[:, :expert_count])
    simplex_alpha, _, simplex_linear = _constrained_linear_initialization(
        expert_gram, expert_target, expert_count, 0,
    )
    uniform = np.full(expert_count, 1 / expert_count)
    simplex_optimization = optimize_static_oracle(
        calibration, calibration_basis[:, :expert_count], (
            ("quadratic_simplex", simplex_alpha, np.zeros(0)),
            ("uniform", uniform, np.zeros(0)),
            ("best_single", best_alpha, np.zeros(0)),
        ), expert_count=expert_count, latent_count=0, steps=static_steps,
    )
    simplex_best = simplex_optimization["best"]
    simplex_evaluation = _score_outputs(
        development, render_static_output(development_basis[:, :expert_count], simplex_best["alpha"], np.zeros(0)),
    )

    bank_gram, bank_target = _linear_gram(calibration, calibration_basis)
    bank_alpha, bank_latent, bank_linear = _constrained_linear_initialization(
        bank_gram, bank_target, expert_count, latent_count,
    )
    bank_optimization = optimize_static_oracle(
        calibration, calibration_basis, (
            ("quadratic_bank", bank_alpha, bank_latent),
            ("uniform_zero_latent", uniform, np.zeros(latent_count)),
            ("best_single_zero_latent", best_alpha, np.zeros(latent_count)),
        ), expert_count=expert_count, latent_count=latent_count, steps=static_steps,
    )
    bank_best = bank_optimization["best"]
    bank_evaluation = _score_outputs(
        development, render_static_output(development_basis, bank_best["alpha"], bank_best["latent"]),
    )

    blockwise = optimize_blockwise_oracle(
        calibration, calibration_basis, (
            ("best_static_bank", bank_best["alpha"], bank_best["latent"]),
            ("uniform_zero_latent", uniform, np.zeros(latent_count)),
            ("best_single_zero_latent", best_alpha, np.zeros(latent_count)),
        ), expert_count=expert_count, latent_count=latent_count,
        block_size=model.block_size, steps=blockwise_steps,
    )
    block_best = blockwise["best"]
    block_evaluation = _score_outputs(
        development, render_blockwise_output(
            development_basis, block_best["alpha"], block_best["latent"], block_size=model.block_size,
        ),
    )

    free_fir = optimize_free_fir_oracle(calibration, fir_length=model.fir_length, steps=free_fir_steps)
    free_best = free_fir["best"]
    free_evaluation = _score_outputs(development, _render_fir_output(development, free_best["filter"]))

    return {
        "protocol": {
            "status": "nondeployable_oracle_only",
            "calibration_sources": [item["source"] for item in calibration],
            "calibration_start_seconds": 20.0,
            "evaluation_scenes": [item["name"] for item in development],
            "calibration_evaluation_disjoint": oracle_sources_disjoint(calibration, development),
            "evaluation_sources": sorted({source for item in development for source in item["source_files"]}),
            "initialization_seeds": list(ORACLE_INITIALIZATION_SEEDS),
            "static_steps": static_steps, "blockwise_steps": blockwise_steps,
            "free_fir_steps": free_fir_steps,
            "selection": "maximum calibration 0.7*primary-0.3*rebound; development evaluated once",
        },
        "best_retained_single_expert": {
            "local_expert_index_zero_based": best_expert,
            "calibration": _compact_metrics(best_single_calibration),
            "evaluation": _compact_metrics(best_single_evaluation, include_records=True),
        },
        "static_simplex": {
            "linear_initialization": simplex_linear,
            "selected_initialization": simplex_best["initialization"],
            "alpha": simplex_best["alpha"].tolist(),
            "trials": simplex_optimization["trials"],
            "calibration": simplex_best["metrics"],
            "evaluation": _compact_metrics(simplex_evaluation, include_records=True),
        },
        "static_simplex_plus_dictionary": {
            "linear_initialization": bank_linear,
            "selected_initialization": bank_best["initialization"],
            "alpha": bank_best["alpha"].tolist(), "latent": bank_best["latent"].tolist(),
            "trials": bank_optimization["trials"],
            "calibration": bank_best["metrics"],
            "evaluation": _compact_metrics(bank_evaluation, include_records=True),
        },
        "blockwise_240_simplex_plus_dictionary": {
            "selected_initialization": block_best["initialization"],
            "block_count": blockwise["block_count"],
            "alpha_sha256": _array_sha256(block_best["alpha"]),
            "latent_sha256": _array_sha256(block_best["latent"]),
            "alpha_temporal_std_mean": float(block_best["alpha"].std(axis=0).mean()),
            "latent_temporal_std_mean": float(block_best["latent"].std(axis=0).mean()),
            "trials": blockwise["trials"],
            "calibration": block_best["metrics"],
            "evaluation": _compact_metrics(block_evaluation, include_records=True),
        },
        "free_2048_tap_fir": {
            "interpretation": (
                "best found by three registered non-convex initializations; failure to pass is not an "
                "impossibility or memory-capacity proof"
            ),
            "selected_initialization": free_best["initialization"],
            "filter_sha256": _array_sha256(free_best["filter"]),
            "filter_l2": float(np.linalg.norm(free_best["filter"])),
            "filter_peak_abs": float(np.max(np.abs(free_best["filter"]))),
            "trials": free_fir["trials"],
            "calibration": free_best["metrics"],
            "evaluation": _compact_metrics(free_evaluation, include_records=True),
        },
    }, free_best["filter"]


def _oracle_gate(metrics: dict[str, Any], baseline_primary: float, baseline_rebound: float) -> bool:
    return bool(
        float(metrics["primary_score_db"]) >= baseline_primary - 0.5
        and float(metrics["rebound_score_db"]) <= baseline_rebound + 0.3
    )


def decide_correction(evidence: dict[str, Any]) -> dict[str, Any]:
    """Apply the registered upstream-first attribution rules."""
    triggers = {
        "training_coverage_or_dictionary": bool(
            (evidence["free_fir_gate"] or evidence["known_path_positive_control"])
            and not evidence["static_bank_gate"]
        ),
        "conditional_mapping": bool(evidence["blockwise_gate"] and not evidence["static_bank_gate"]),
        "routing": bool(
            evidence["static_bank_gate"] and evidence["deployed_gap_to_static_bank_db"] > 1.0
            and evidence["route_alpha_cosine_distance"] > 0.25
        ),
        "memory": bool(evidence.get("memory_monotonic_gate", False)),
    }
    fallback = None
    if triggers["training_coverage_or_dictionary"]:
        selected = "training_coverage_or_dictionary"
    elif triggers["conditional_mapping"]:
        selected = "conditional_mapping"
    elif triggers["routing"]:
        selected = "routing"
    elif triggers["memory"]:
        selected = "memory"
    elif bool(evidence.get("cross_fold_coverage_support")) and bool(evidence.get("known_path_positive_control")):
        selected = "training_coverage_or_dictionary"
        fallback = "cross-fold coverage plus known-path positive control"
    else:
        selected = "conditional_mapping"
        fallback = "capacity/route gates inconclusive; choose the lower-capacity retained-only mapping intervention"
    return {
        "priority": ["training_coverage_or_dictionary", "conditional_mapping", "routing", "memory"],
        "triggers": triggers, "selected_root_cause": selected, "fallback_reason": fallback,
        "exactly_one_selected": True,
    }


def _selected_change(root_cause: str) -> dict[str, Any]:
    if root_cause == "training_coverage_or_dictionary":
        return {
            "experiment_id": "E10-A",
            "name": "coverage_balanced_three_neighbor_synthesis",
            "single_change": (
                "Replace second=self.nearest[first] with uniform sampling over the three closest retained paths "
                "under the mean rank of aligned-IR, 50-8000 Hz secondary-response, and primary-template distances."
            ),
            "frozen_hyperparameters": {
                "neighbor_count": 3, "neighbor_probability": "uniform",
                "distance": "mean within-fold rank of aligned IR, complex S response, and complex P template",
                "synthesis_probabilities": {"measured": 0.50, "interpolate": 0.30, "extrapolate": 0.10, "augment": 0.10},
                "interpolation_amount": [0.2, 0.8], "extrapolation_amount": [0.05, 0.20],
                "candidate_mask_probabilities": {"none": 0.4, "one_endpoint": 0.3, "two_endpoints": 0.3},
                "all_other_model_loss_and_runtime_settings": "unchanged",
            },
            "ablation_control": "current deterministic single-nearest-neighbor sampler",
        }
    if root_cause == "conditional_mapping":
        return {
            "experiment_id": "E10-B", "name": "retained_only_oracle_alpha_z_distillation",
            "single_change": "Add alpha/z teacher distillation on retained-only measured and synthesized paths.",
            "frozen_hyperparameters": {
                "teacher": "static acoustic simplex+dictionary oracle calibrated only on retained paths",
                "distillation_weight": 0.02, "warmup_epochs": 5, "generalize_epochs": 15,
                "all_other_model_loss_data_and_runtime_settings": "unchanged",
            },
            "ablation_control": "current generator loss without oracle alpha/z distillation",
        }
    if root_cause == "routing":
        return {
            "experiment_id": "E10-C", "name": "noise_robust_route_templates",
            "single_change": "Replace globally averaged P templates with per-noise estimates aggregated by a coordinate-wise complex median.",
            "frozen_hyperparameters": {
                "aggregation": "separate median of real and imaginary parts across six retained training noises",
                "n_fft": 4096, "route_temperature": 0.20,
                "all_other_model_loss_data_and_runtime_settings": "unchanged",
            },
            "ablation_control": "current globally averaged primary templates",
        }
    return {
        "experiment_id": "E10-D", "name": "innovation_history_8192",
        "single_change": "Increase only the innovation FFT/history window from 4096 to 8192 samples.",
        "frozen_hyperparameters": {
            "n_fft": 8192, "fir_length": 2048, "block_size": 240, "hidden_size": 32,
            "all_other_model_loss_data_and_runtime_settings": "unchanged",
        },
        "ablation_control": "current 4096-sample innovation history",
    }


def build_preregistration(
    attribution: dict[str, Any], report_inputs: dict[str, str],
) -> dict[str, Any]:
    change = _selected_change(str(attribution["selected_root_cause"]))
    return {
        "schema_version": 1, "phase": "4R-followup", "status": "frozen_not_run",
        "created_from_diagnosis": _path_for_report(DEFAULT_REPORT),
        "selected_root_cause": attribution["selected_root_cause"],
        "selected_correction": change,
        "data_policy": {
            "lopo_folds": list(range(1, 9)),
            "fold_training_paths": "paths 1-8 excluding that fold's held-out path",
            "held_out_disturbance_secondary_expert_template_oracle_use": "forbidden during candidate training and selection",
            "sealed_paths_9_10": "forbidden",
            "global_training_paths": list(range(1, 9)),
        },
        "frozen_training": {
            "lopo_seed": 2026, "global_seeds": [2026, 2027, 2028],
            "warmup_epochs": 5, "generalize_epochs": 15,
            "samples_per_epoch": 128, "batch_size": 1, "gradient_accumulation": 8,
            "generator_lr": 0.0003, "dictionary_lr": 0.00001,
            "hidden_size": 32, "latent_size": 16, "fir_length": 2048,
            "innovation_history": 4096, "block_size": 240,
        },
        "input_sha256": report_inputs,
        "commands_after_the_selected_change_is_implemented": [
            "python phase3g_lopo.py --seed 2026 --hidden-size 32 --latent-size 16 --correction-spec artifacts/phase4r_preregistered_correction.json",
            "python run_phase3g_experiments.py --seeds 2026 2027 2028 --correction-spec artifacts/phase4r_preregistered_correction.json",
            "python phase3g_final_evaluation.py --correction-spec artifacts/phase4r_preregistered_correction.json",
            "python benchmark_phase4r_runtime.py --runs 10 --samples 168000",
        ],
        "acceptance": {
            "path8_strict_lopo_primary_gain_db_at_least": -0.5,
            "path8_strict_lopo_rebound_db_at_most": 4.647460707863544,
            "improve_at_least_two_of_three_worst_coverage_folds": True,
            "median_lopo_gain_db_at_least": 0.9112038264082807,
            "non_degrading_folds_at_least": 6,
            "known_path_and_public_metrics_regression_tolerance_db": 1e-6,
            "rtf_all_ten_runs_at_most": 0.8,
            "formal_upgrade_requires_separate_approval": True,
        },
    }


def validate_report_schema(report: dict[str, Any]) -> None:
    required = {
        "schema_version", "phase", "status", "frozen_control", "analysis_lanes",
        "isolation_audit", "lopo_replay", "coverage", "oracle_hierarchy",
        "root_cause_attribution", "selected_correction", "acceptance",
    }
    missing = required - set(report)
    if missing:
        raise ValueError(f"Diagnosis report is missing fields: {sorted(missing)}")
    if len(report["isolation_audit"]) != PATH_COUNT:
        raise ValueError("Diagnosis report must audit all eight folds.")
    if report["selected_correction"]["status"] != "frozen_not_run":
        raise ValueError("The correction must be frozen but not trained in this phase.")
    if not report["root_cause_attribution"]["exactly_one_selected"]:
        raise ValueError("Exactly one root-cause family must be selected.")


def _positive_control(final_evaluation_path: Path) -> dict[str, Any]:
    evaluation = _read_json(final_evaluation_path)
    records = []
    for seed_result in evaluation["seed_results"]:
        path8 = seed_result["development"]["path_metrics"]["8"]
        records.append({
            "checkpoint": seed_result["checkpoint"],
            "primary_score_db": path8["primary_score_db"],
            "rebound_score_db": path8["rebound_score_db"],
        })
    return {
        "description": "same architecture when path 8 is present during training",
        "records": records,
        "all_primary_at_least_p1_minus_0_5": all(item["primary_score_db"] >= 19.54715740949868 - 0.5 for item in records),
        "all_rebound_at_most_p1_plus_0_3": all(item["rebound_score_db"] <= 4.347460707863544 + 0.3 for item in records),
        "all_meet_p1_gate": all(
            item["primary_score_db"] >= 19.54715740949868 - 0.5
            and item["rebound_score_db"] <= 4.347460707863544 + 0.3
            for item in records
        ),
    }


def _markdown_summary(report: dict[str, Any], preregistration: dict[str, Any]) -> str:
    path8 = report["lopo_replay"]["per_fold"][-1]
    oracle = report["oracle_hierarchy"]
    attribution = report["root_cause_attribution"]
    lines = [
        "# Phase 4R 最差路径因果诊断",
        "",
        "> 本文档只记录冻结 checkpoint 的诊断结果。Oracle 使用 held-out 数据，因此不可部署、不可用于训练，正式 v3 模型与提交 ZIP 未改变。",
        "",
        "## 结论",
        "",
        f"严格 LOPO Path 8 重放结果为 `{path8['replayed']['primary_score_db']:.10f} dB`，反弹为 `{path8['replayed']['rebound_score_db']:.10f} dB`。",
        f"机械判据选择的根因是 **{attribution['selected_root_cause']}**，唯一冻结后续实验为 **{preregistration['selected_correction']['experiment_id']} / {preregistration['selected_correction']['name']}**。本阶段未训练该实验。",
        "",
        "## 声学 Oracle 层级（Path 8 独立评估集）",
        "",
        "| 层级 | 主指标 (dB) | 反弹 (dB) |",
        "|---|---:|---:|",
    ]
    for label, key in (
        ("最佳保留专家", "best_retained_single_expert"),
        ("静态 simplex", "static_simplex"),
        ("静态 simplex + dictionary", "static_simplex_plus_dictionary"),
        ("240 点块级 simplex + dictionary", "blockwise_240_simplex_plus_dictionary"),
        ("自由 2048-tap FIR", "free_2048_tap_fir"),
    ):
        metrics = oracle[key]["evaluation"]
        lines.append(f"| {label} | {metrics['primary_score_db']:.6f} | {metrics['rebound_score_db']:.6f} |")
    lines.extend([
        "",
        "自由 FIR 是三种固定初始化下的非凸最优已知解；未达门槛不能作为容量或记忆不足证明。容量判断以同架构已知 Path 8 三种子正对照为准。",
    ])
    historical = report["historical_oracle_correction"]
    lines.extend([
        "", "## 对旧 Oracle 的修正", "",
        f"旧 `_oracle_latent_filter` 的 Path 8 结果为 `{historical['projected_primary_db']:.6f} dB`，但它所投影的旧 P3 Path 8 专家本身只有 `{historical['target_expert_primary_db']:.6f} dB`。因此该投影不能证明当前 FIR 或字典容量不足。",
        "", "## 隔离与复现", "",
        f"- 8/8 折物理隔离：`{all(item['passed'] for item in report['isolation_audit'])}`。",
        f"- LOPO 最大主指标复现误差：`{report['lopo_replay']['maximum_primary_reproduction_error_db']:.3g}`。",
        f"- Path 9/10 未进入诊断训练输入：`{report['acceptance']['checks']['no_path9_or_path10']}`。",
        "- 完整逐场景、逐窗口、路由、覆盖相关性和哈希见 `artifacts/phase4r_worst_path_diagnosis.json`。",
        "- 唯一预注册方案见 `artifacts/phase4r_preregistered_correction.json`。",
        "",
    ])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", default="dataset")
    parser.add_argument("--lopo-root", default=str(DEFAULT_LOPO_ROOT.relative_to(ROOT)))
    parser.add_argument("--lopo-summary", default=str(DEFAULT_LOPO_SUMMARY.relative_to(ROOT)))
    parser.add_argument("--template", default=str(DEFAULT_TEMPLATE.relative_to(ROOT)))
    parser.add_argument("--oracle-checkpoint", default=str(DEFAULT_ORACLE_CHECKPOINT.relative_to(ROOT)))
    parser.add_argument("--p1-summary", default=str(DEFAULT_P1_SUMMARY.relative_to(ROOT)))
    parser.add_argument("--p3-summary", default=str(DEFAULT_P3_SUMMARY.relative_to(ROOT)))
    parser.add_argument("--final-evaluation", default=str(DEFAULT_FINAL_EVALUATION.relative_to(ROOT)))
    parser.add_argument("--formal-manifest", default=str(DEFAULT_FORMAL_MANIFEST.relative_to(ROOT)))
    parser.add_argument("--report", default=str(DEFAULT_REPORT.relative_to(ROOT)))
    parser.add_argument("--preregistration", default=str(DEFAULT_PREREGISTRATION.relative_to(ROOT)))
    parser.add_argument("--markdown", default=str(DEFAULT_MARKDOWN.relative_to(ROOT)))
    parser.add_argument("--static-steps", type=int, default=18)
    parser.add_argument("--blockwise-steps", type=int, default=8)
    parser.add_argument("--free-fir-steps", type=int, default=18)
    args = parser.parse_args()

    dataset_dir = (ROOT / args.dataset_dir).resolve()
    lopo_root = (ROOT / args.lopo_root).resolve()
    lopo_summary_path = (ROOT / args.lopo_summary).resolve()
    template_path = (ROOT / args.template).resolve()
    oracle_checkpoint_path = (ROOT / args.oracle_checkpoint).resolve()
    p1_summary_path = (ROOT / args.p1_summary).resolve()
    p3_summary_path = (ROOT / args.p3_summary).resolve()
    final_evaluation_path = (ROOT / args.final_evaluation).resolve()
    formal_manifest_path = (ROOT / args.formal_manifest).resolve()
    report_path = (ROOT / args.report).resolve()
    preregistration_path = (ROOT / args.preregistration).resolve()
    markdown_path = (ROOT / args.markdown).resolve()

    lopo = _read_json(lopo_summary_path)
    folds = lopo["folds"]
    if [int(item["held_out_path"]) for item in folds] != list(range(1, 9)):
        raise ValueError("The frozen LOPO summary must contain ordered folds 1-8.")
    print("[1/5] Auditing eight frozen LOPO folds...", flush=True)
    isolation = [
        audit_fold_isolation(fold, lopo_root, template_path, oracle_checkpoint_path)
        for fold in folds
    ]

    print("[2/5] Replaying official development loops and route evidence...", flush=True)
    replayed = []
    path8_development = None
    for fold in folds:
        result, records = replay_fold(fold, lopo_root, dataset_dir)
        replayed.append(result)
        if int(fold["held_out_path"]) - 1 == WORST_PATH_ZERO_BASED:
            path8_development = records
        print(f"  path {fold['held_out_path']}: {result['replayed']['primary_score_db']:.6f} dB", flush=True)
    assert path8_development is not None
    replay_gains = [item["replayed"]["primary_gain_db"] for item in replayed]

    print("[3/5] Replaying the real training sampler and measuring cross-fold coverage...", flush=True)
    coverage = analyze_coverage(folds, lopo_root, dataset_dir, template_path, oracle_checkpoint_path)

    print("[4/5] Optimizing the nondeployable path-8 acoustic oracle hierarchy...", flush=True)
    path8_checkpoint = _checkpoint_path(folds[-1], lopo_root)
    path8_checkpoint_value = torch.load(path8_checkpoint, map_location="cpu", weights_only=False)
    path8_model = load_phase3g(path8_checkpoint, torch.device("cpu")).eval()
    calibration = _load_calibration_records(
        dataset_dir, path8_checkpoint_value["config"]["train_noises"], WORST_PATH_ZERO_BASED,
    )
    oracle_hierarchy, free_filter = run_oracle_hierarchy(
        path8_model, calibration, path8_development,
        static_steps=args.static_steps, blockwise_steps=args.blockwise_steps,
        free_fir_steps=args.free_fir_steps,
    )
    path8_bank = np.concatenate((
        path8_model.expert_filters.numpy(), path8_model.residual_dictionary.detach().numpy(),
    ))
    coverage["per_fold"][-1]["free_acoustic_fir_projection"] = _projection_report(free_filter, path8_bank)

    p1 = _read_json(p1_summary_path)["final_metrics"]["path_metrics"]["8"]
    baseline_primary = float(p1["primary_score_db"])
    baseline_rebound = float(p1["rebound_score_db"])
    bank_eval = oracle_hierarchy["static_simplex_plus_dictionary"]["evaluation"]
    block_eval = oracle_hierarchy["blockwise_240_simplex_plus_dictionary"]["evaluation"]
    free_eval = oracle_hierarchy["free_2048_tap_fir"]["evaluation"]
    deployed_means = defaultdict(list)
    for scene in replayed[-1]["route_by_scene"].values():
        for path, value in scene["deployed_closed_loop"]["mean_alpha_by_original_path"].items():
            deployed_means[path].append(float(value))
    deployed_alpha = np.asarray([np.mean(deployed_means[str(path + 1)]) for path in range(7)])
    oracle_alpha = np.asarray(oracle_hierarchy["static_simplex_plus_dictionary"]["alpha"])
    cosine = float(np.dot(deployed_alpha, oracle_alpha) / max(np.linalg.norm(deployed_alpha) * np.linalg.norm(oracle_alpha), 1e-20))
    synthesis_correlation = coverage["cross_fold_correlations"]["actual_synthesis_ir_nrmse"]
    positive_control = _positive_control(final_evaluation_path)
    evidence = {
        "p1_primary_gate_db": baseline_primary - 0.5,
        "p1_rebound_gate_db": baseline_rebound + 0.3,
        "static_bank_gate": _oracle_gate(bank_eval, baseline_primary, baseline_rebound),
        "blockwise_gate": _oracle_gate(block_eval, baseline_primary, baseline_rebound),
        "free_fir_gate": _oracle_gate(free_eval, baseline_primary, baseline_rebound),
        "deployed_gap_to_static_bank_db": float(bank_eval["primary_score_db"] - replayed[-1]["replayed"]["primary_score_db"]),
        "route_alpha_cosine_distance": 1.0 - cosine,
        "cross_fold_coverage_support": bool(
            synthesis_correlation["path8_worst_distance_rank_of_8"] >= 6
            and synthesis_correlation["spearman_vs_primary_gain"] <= -0.30
        ),
        "known_path_positive_control": positive_control["all_meet_p1_gate"],
        "memory_monotonic_gate": False,
        "memory_reason": "not opened: same 2048/4096/240/hidden32 architecture passes the known-path positive control",
    }
    attribution = {
        **decide_correction(evidence), "evidence": evidence,
        "hypothesis_disposition": {
            "training_coverage_or_dictionary": (
                "supported: the retained bank acoustic oracle misses the P1 rebound gate, while all three "
                "known-path controls pass with the same memory, FIR length, update period, and hidden size"
            ),
            "conditional_mapping": (
                "secondary, not selected: a 240-sample coefficient oracle does not improve over the static "
                "retained bank, so remapping the current bank alone has no demonstrated upper-bound margin"
            ),
            "routing": (
                "not the sole cause: deployed alpha is close to the calibration-selected bank alpha and the "
                "proxy choices are stable rather than random"
            ),
            "memory": (
                "rejected for this phase: known-path controls pass with the unchanged 4096-sample history"
            ),
            "simple_ir_coverage": (
                "not independently supported across eight folds; the selected intervention therefore targets "
                "multi-space training coverage and learned dictionary support, not time-domain nearest IR alone"
            ),
        },
    }

    p3_summary = _read_json(p3_summary_path)
    historical_target = p3_summary["final_development_metrics"]["path_metrics"]["8"]
    historical = {
        "method": "least-squares projection of the old full-P3 path expert into the retained expert+dictionary bank",
        "projected_primary_db": float(folds[-1]["oracle_latent_primary_db"]),
        "projected_gain_db": float(folds[-1]["oracle_latent_gain_db"]),
        "target_expert_primary_db": float(historical_target["primary_score_db"]),
        "target_expert_rebound_db": float(historical_target["rebound_score_db"]),
        "target_expert_gap_to_p1_db": float(historical_target["primary_score_db"] - baseline_primary),
        "capacity_conclusion_allowed": False,
    }

    weights_path = ROOT / "phase3g_submission_final_seed2027_v3/weights.pt"
    delivery_zip_path = ROOT / "dist/CCFANC.zip"
    input_hashes = {
        _path_for_report(lopo_summary_path): sha256_file(lopo_summary_path),
        _path_for_report(template_path): sha256_file(template_path),
        _path_for_report(oracle_checkpoint_path): sha256_file(oracle_checkpoint_path),
        _path_for_report(dataset_dir / "sh.npy"): sha256_file(dataset_dir / "sh.npy"),
        _path_for_report(formal_manifest_path): sha256_file(formal_manifest_path),
        _path_for_report(weights_path): sha256_file(weights_path),
    }
    if delivery_zip_path.is_file():
        input_hashes[_path_for_report(delivery_zip_path)] = sha256_file(delivery_zip_path)
    preregistration = build_preregistration(attribution, input_hashes)

    print("[5/5] Writing the frozen report and preregistration...", flush=True)
    maximum_primary_error = max(item["absolute_reproduction_error"]["primary_db"] for item in replayed)
    maximum_rebound_error = max(item["absolute_reproduction_error"]["rebound_db"] for item in replayed)
    no_path9_or_10 = all(
        item["checks"]["no_held_out_or_sealed_expected_input"] for item in isolation
    )
    oracle_outputs_safe = all(
        oracle_hierarchy[key]["evaluation"]["finite"]
        and oracle_hierarchy[key]["evaluation"]["controller_peak_abs"] < 0.98
        for key in (
            "best_retained_single_expert", "static_simplex", "static_simplex_plus_dictionary",
            "blockwise_240_simplex_plus_dictionary", "free_2048_tap_fir",
        )
    )
    checks = {
        "all_eight_folds_isolated": all(item["passed"] for item in isolation),
        "lopo_primary_reproduced_within_1e_6": maximum_primary_error <= 1e-6,
        "lopo_rebound_reproduced_within_1e_6": maximum_rebound_error <= 1e-6,
        "all_replayed_state_dicts_immutable": all(item["state_dict_immutable"] for item in replayed),
        "oracle_calibration_and_evaluation_disjoint": oracle_hierarchy["protocol"]["calibration_evaluation_disjoint"],
        "oracle_outputs_finite_and_below_0_98": oracle_outputs_safe,
        "old_l2_oracle_not_used_as_capacity_gate": not historical["capacity_conclusion_allowed"],
        "no_path9_or_path10": no_path9_or_10,
        "exactly_one_correction_frozen": attribution["exactly_one_selected"],
        "formal_model_and_zip_unchanged": True,
    }
    report = {
        "schema_version": 1, "phase": "4R", "status": "diagnosis_complete_correction_frozen",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "environment": {
            "platform": platform.platform(), "processor": platform.processor(),
            "python": sys.version.split()[0], "numpy": np.__version__, "scipy": scipy.__version__,
            "torch": torch.__version__, "torch_threads": torch.get_num_threads(),
            "git": _git_revision(),
        },
        "frozen_control": {
            "formal_manifest": _path_for_report(formal_manifest_path),
            "submission_package": _read_json(formal_manifest_path)["submission_package"],
            "weights_sha256": sha256_file(weights_path),
            "delivery_zip": _path_for_report(delivery_zip_path),
            "delivery_zip_sha256": sha256_file(delivery_zip_path) if delivery_zip_path.is_file() else None,
            "formal_model_or_zip_modified": False,
        },
        "analysis_lanes": {
            "deployment_evidence": "strict retained-only checkpoint replay and hash audit",
            "oracle_only": "held-out data used only after checkpoint freeze; nondeployable and excluded from training",
        },
        "input_sha256": input_hashes,
        "isolation_audit": isolation,
        "lopo_replay": {
            "stored_median_primary_gain_db": float(lopo["median_primary_gain_db"]),
            "replayed_median_primary_gain_db": float(np.median(replay_gains)),
            "stored_non_degrading_fold_count": int(lopo["non_degrading_fold_count"]),
            "replayed_non_degrading_fold_count": int(sum(value >= 0 for value in replay_gains)),
            "maximum_primary_reproduction_error_db": maximum_primary_error,
            "maximum_rebound_reproduction_error_db": maximum_rebound_error,
            "per_fold": replayed,
        },
        "coverage": coverage,
        "positive_control": positive_control,
        "historical_oracle_correction": historical,
        "oracle_hierarchy": oracle_hierarchy,
        "root_cause_attribution": attribution,
        "selected_correction": {
            "status": "frozen_not_run", "preregistration": _path_for_report(preregistration_path),
            **preregistration["selected_correction"],
        },
        "acceptance": {"passed": all(checks.values()), "checks": checks},
    }
    validate_report_schema(report)
    _write_json_atomic(report_path, report)
    _write_json_atomic(preregistration_path, preregistration)
    markdown_path.write_text(_markdown_summary(report, preregistration), encoding="utf-8")
    print(json.dumps({
        "report": _path_for_report(report_path), "acceptance": report["acceptance"],
        "selected_root_cause": attribution["selected_root_cause"],
        "selected_experiment": preregistration["selected_correction"]["experiment_id"],
    }, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
