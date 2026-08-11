"""Leakage-safe development, stress, and final validation for Phase 2."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import numpy as np
import torch

from dataset import apply_dynamic_path
from phase1_data import build_validation_manifest, iter_validation_examples
from phase1_validation import _aggregate, _load_official_scorer, evaluate_v6_model
from phase2_paths import PHASE2_FINAL_PATHS, PHASE2_TRAIN_PATHS, augment_secondary_path
from v6_metrics import INITIALIZATION_SAMPLES, SCORING_WINDOW_SAMPLES


def build_phase2_manifests(dataset_dir: str | Path, seed: int = 2026) -> dict[str, Any]:
    base = build_validation_manifest(dataset_dir)
    dev = copy.deepcopy(base)
    dev["manifest_version"] = 2
    dev["split"] = "development_paths_1_to_8"
    dev["path_indices_zero_based"] = list(PHASE2_TRAIN_PATHS)
    final = copy.deepcopy(base)
    final["manifest_version"] = 2
    final["split"] = "final_unseen_paths_9_and_10"
    final["path_indices_zero_based"] = list(PHASE2_FINAL_PATHS)
    rng = np.random.default_rng(seed + 2000)
    variants = []
    for path_index in PHASE2_TRAIN_PATHS:
        variants.append({
            "base_path_index_zero_based": path_index,
            "gain_db": float(rng.uniform(-1.5, 1.5)),
            "delay_samples": int(rng.integers(-2, 3)),
            "tail_energy_db": float(rng.uniform(-35.0, -30.0)),
            "seed": int(seed + 10_000 + path_index),
        })
    stress = {
        "manifest_version": 1,
        "split": "fixed_synthetic_dev_stress",
        "source_manifest": dev,
        "variants": variants,
    }
    return {"development": dev, "stress": stress, "final": final}


def robust_development_score(metrics: dict[str, Any], path_quantile: float = 0.25) -> float:
    values = np.asarray([
        item["primary_score_db"] for item in metrics["path_metrics"].values()
    ], dtype=np.float64)
    return float(metrics["selection_score"] + path_quantile * np.percentile(values, 25.0))


def development_gate(baseline: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    checks = {
        "primary_drop_at_most_0_5_db": candidate["primary_score_db"] >= baseline["primary_score_db"] - 0.5,
        "rebound_increase_at_most_0_3_db": candidate["rebound_score_db"] <= baseline["rebound_score_db"] + 0.3,
        "controller_peak_at_most_1": candidate["controller_peak_abs"] <= 1.0,
    }
    return {"passed": all(checks.values()), "checks": checks}


def phase2_acceptance(
    baseline_dev: dict[str, Any], baseline_final: dict[str, Any],
    candidate_dev: dict[str, Any], candidate_final: dict[str, Any],
) -> dict[str, Any]:
    """Apply the Phase-2 hard gates, weighted by the number of evaluated paths."""
    dev_count = len(candidate_dev["path_metrics"])
    final_count = len(candidate_final["path_metrics"])
    total_count = dev_count + final_count

    def combined(first: dict[str, Any], second: dict[str, Any], key: str) -> float:
        return float((dev_count * first[key] + final_count * second[key]) / total_count)

    baseline_s = combined(baseline_dev, baseline_final, "primary_score_db")
    candidate_s = combined(candidate_dev, candidate_final, "primary_score_db")
    baseline_r = combined(baseline_dev, baseline_final, "rebound_score_db")
    candidate_r = combined(candidate_dev, candidate_final, "rebound_score_db")
    baseline_c = 0.7 * baseline_s - 0.3 * baseline_r
    candidate_c = 0.7 * candidate_s - 0.3 * candidate_r
    worst_path = min(candidate_dev["worst_path_primary_db"], candidate_final["worst_path_primary_db"])
    path10 = float(candidate_final["path_metrics"]["10"]["primary_score_db"])
    unseen_gain = float(candidate_final["primary_score_db"] - baseline_final["primary_score_db"])
    peak = max(candidate_dev["controller_peak_abs"], candidate_final["controller_peak_abs"])
    checks = {
        "global_primary_drop_at_most_0_5_db": candidate_s >= baseline_s - 0.5,
        "global_rebound_increase_at_most_0_3_db": candidate_r <= baseline_r + 0.3,
        "global_composite_drop_at_most_0_2": candidate_c >= baseline_c - 0.2,
        "worst_path_primary_at_least_2_db": worst_path >= 2.0,
        "path10_primary_at_least_2_5_db": path10 >= 2.5,
        "paths9_10_average_gain_at_least_0_75_db": unseen_gain >= 0.75,
        "controller_peak_at_most_1": peak <= 1.0,
    }
    return {
        "passed": all(checks.values()), "checks": checks,
        "baseline_global": {"primary_score_db": baseline_s, "rebound_score_db": baseline_r, "composite": baseline_c},
        "candidate_global": {"primary_score_db": candidate_s, "rebound_score_db": candidate_r, "composite": candidate_c},
        "worst_path_primary_db": worst_path, "path10_primary_db": path10,
        "unseen_average_primary_gain_db": unseen_gain, "controller_peak_abs": peak,
    }


def evaluate_phase2_development(
    model: torch.nn.Module,
    dataset_dir: str | Path,
    device: torch.device,
    manifest: dict[str, Any],
    *,
    include_records: bool = False,
) -> dict[str, Any]:
    result = evaluate_v6_model(model, dataset_dir, device, manifest, include_records=include_records)
    result["robust_development_score"] = robust_development_score(result)
    return result


def evaluate_phase2_stress(
    model: torch.nn.Module,
    dataset_dir: str | Path,
    device: torch.device,
    stress_manifest: dict[str, Any],
    *,
    include_records: bool = False,
) -> dict[str, Any]:
    scorer = _load_official_scorer()
    variants = {
        int(item["base_path_index_zero_based"]): item for item in stress_manifest["variants"]
    }
    records = []
    was_training = model.training
    model.eval()
    with torch.inference_mode():
        for scene_name, path_index, reference, path, target in iter_validation_examples(
            dataset_dir, stress_manifest["source_manifest"]
        ):
            spec = variants[path_index]
            augmented, _ = augment_secondary_path(
                path.numpy(), np.random.default_rng(int(spec["seed"])),
                gain_db=float(spec["gain_db"]), delay_samples=int(spec["delay_samples"]),
                tail_energy_db=float(spec["tail_energy_db"]),
            )
            reference = reference.unsqueeze(0).to(device)
            target = target.unsqueeze(0).to(device)
            aug_path = torch.from_numpy(augmented).unsqueeze(0).to(device)
            controller = model(reference)
            residual = target - apply_dynamic_path(controller, aug_path)
            scored = slice(INITIALIZATION_SAMPLES, None)
            scored_metrics = scorer(
                target[0, scored].cpu().numpy(), residual[0, scored].cpu().numpy(),
                controller[0, scored].cpu().numpy(), sample_rate=48_000,
                window_samples=SCORING_WINDOW_SAMPLES,
            )
            records.append({
                "scene_name": scene_name, "path_index_zero_based": path_index,
                "path_number": path_index + 1,
                "full_controller_peak_abs": float(controller.abs().amax().cpu()),
                "stress_variant": spec, **scored_metrics,
            })
    if was_training:
        model.train()
    result = _aggregate(records)
    result["robust_development_score"] = robust_development_score(result)
    if include_records:
        result["records"] = records
    return result
