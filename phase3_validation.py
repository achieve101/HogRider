"""Oracle, feedback, and path-switch validation for Phase 3."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

from dataset import apply_dynamic_path
from phase1_data import build_validation_manifest, iter_validation_examples
from phase1_validation import _aggregate, _load_official_scorer
from phase3_closed_loop import rollout_feedback_closed_loop
from phase3_model import FeedbackFIRController
from v6_metrics import INITIALIZATION_SAMPLES, SCORING_WINDOW_SAMPLES, TOTAL_SAMPLES


PATH_SWITCH_PAIRS = ((0, 6), (6, 0), (3, 5), (5, 7))


def phase3_selection_score(metrics: dict[str, Any]) -> float:
    return float(
        metrics["selection_score"]
        + 0.5 * metrics["worst_path_primary_db"]
        + 0.2 * metrics["first_window_primary_db"]
    )


def build_phase3_manifests(dataset_dir: str | Path, seed: int = 2026) -> dict[str, Any]:
    base = build_validation_manifest(dataset_dir)
    development = dict(base)
    development["manifest_version"] = 3
    development["split"] = "phase3_development_paths_1_to_8"
    development["path_indices_zero_based"] = list(range(8))
    final = dict(base)
    final["manifest_version"] = 3
    final["split"] = "phase3_final_paths_9_and_10"
    final["path_indices_zero_based"] = [8, 9]
    return {
        "seed": seed,
        "development": development,
        "path_switch": {
            "manifest_version": 1, "source_manifest": development,
            "switch_sample": 96_000,
            "pairs_zero_based": [list(pair) for pair in PATH_SWITCH_PAIRS],
            "scenes": ["vehicle_continuous", "restaurant_continuous"],
        },
        "final": final,
    }


def _score_record(
    scorer, scene_name: str, path_index: int, disturbance: torch.Tensor,
    residual: torch.Tensor, output: torch.Tensor,
) -> dict[str, Any]:
    scored = slice(INITIALIZATION_SAMPLES, None)
    metrics = scorer(
        disturbance[0, scored].detach().cpu().numpy(),
        residual[0, scored].detach().cpu().numpy(),
        output[0, scored].detach().cpu().numpy(),
        sample_rate=48_000, window_samples=SCORING_WINDOW_SAMPLES,
    )
    return {
        "scene_name": scene_name, "path_index_zero_based": path_index,
        "path_number": path_index + 1,
        "full_controller_peak_abs": float(output.abs().amax().detach().cpu()),
        **metrics,
    }


def evaluate_phase3_oracle(
    model: FeedbackFIRController,
    dataset_dir: str | Path,
    device: torch.device,
    manifest: dict[str, Any],
    *, include_records: bool = False,
) -> dict[str, Any]:
    scorer = _load_official_scorer()
    records = []
    was_training = model.training
    model.eval()
    with torch.inference_mode():
        for scene, path_index, reference, path, disturbance in iter_validation_examples(dataset_dir, manifest):
            reference = reference.unsqueeze(0).to(device)
            path = path.unsqueeze(0).to(device)
            disturbance = disturbance.unsqueeze(0).to(device)
            output, _ = model.oracle_forward(reference, torch.tensor([path_index], device=device))
            residual = disturbance - apply_dynamic_path(output, path)
            records.append(_score_record(scorer, scene, path_index, disturbance, residual, output))
    if was_training:
        model.train()
    result = _aggregate(records)
    result["phase3_selection_score"] = phase3_selection_score(result)
    if include_records:
        result["records"] = records
    return result


def evaluate_phase3_feedback(
    model: FeedbackFIRController,
    dataset_dir: str | Path,
    device: torch.device,
    manifest: dict[str, Any],
    *, include_records: bool = False,
) -> dict[str, Any]:
    scorer = _load_official_scorer()
    records, accuracies = [], []
    blocks = TOTAL_SAMPLES // model.block_size
    was_training = model.training
    model.eval()
    with torch.inference_mode():
        for scene, path_index, reference, path, disturbance in iter_validation_examples(dataset_dir, manifest):
            reference = reference.unsqueeze(0).to(device)
            disturbance = disturbance.unsqueeze(0).to(device)
            paths = path.unsqueeze(0).unsqueeze(0).to(device)
            slots = torch.zeros(1, blocks, dtype=torch.long, device=device)
            route_label = path_index if path_index < model.num_experts else 0
            labels = torch.full((1, blocks), route_label, dtype=torch.long, device=device)
            rollout = rollout_feedback_closed_loop(
                model, reference, disturbance, paths, slots, labels, truncate_blocks=0,
            )
            record = _score_record(
                scorer, scene, path_index, disturbance, rollout.residual, rollout.output,
            )
            record["route_accuracy_after_initialization"] = (
                float(rollout.route_accuracy_after_initialization.cpu())
                if path_index < model.num_experts else None
            )
            records.append(record)
            if record["route_accuracy_after_initialization"] is not None:
                accuracies.append(record["route_accuracy_after_initialization"])
    if was_training:
        model.train()
    result = _aggregate(records)
    result["phase3_selection_score"] = phase3_selection_score(result)
    result["route_accuracy_after_initialization"] = float(np.mean(accuracies)) if accuracies else None
    if include_records:
        result["records"] = records
    return result


def evaluate_path_switch_stress(
    model: FeedbackFIRController,
    dataset_dir: str | Path,
    device: torch.device,
    switch_manifest: dict[str, Any],
    *, include_records: bool = False,
) -> dict[str, Any]:
    scorer = _load_official_scorer()
    source_manifest = switch_manifest["source_manifest"]
    wanted_scenes = set(switch_manifest["scenes"])
    examples = {
        (scene, path): (reference, secondary, target)
        for scene, path, reference, secondary, target in iter_validation_examples(dataset_dir, source_manifest)
        if scene in wanted_scenes
    }
    switch_sample = int(switch_manifest["switch_sample"])
    switch_block = switch_sample // model.block_size
    blocks = TOTAL_SAMPLES // model.block_size
    records, post_switch = [], []
    was_training = model.training
    model.eval()
    with torch.inference_mode():
        for first, second in (tuple(pair) for pair in switch_manifest["pairs_zero_based"]):
            for scene in sorted(wanted_scenes):
                reference, path_a, target_a = examples[(scene, first)]
                _, path_b, target_b = examples[(scene, second)]
                disturbance = torch.cat((target_a[:switch_sample], target_b[switch_sample:])).unsqueeze(0).to(device)
                reference = reference.unsqueeze(0).to(device)
                paths = torch.stack((path_a, path_b)).unsqueeze(0).to(device)
                slots = torch.zeros(1, blocks, dtype=torch.long, device=device)
                slots[:, switch_block:] = 1
                labels = torch.full((1, blocks), first, dtype=torch.long, device=device)
                labels[:, switch_block:] = second
                rollout = rollout_feedback_closed_loop(
                    model, reference, disturbance, paths, slots, labels, truncate_blocks=0,
                )
                name = f"{scene}_path_{first + 1}_to_{second + 1}"
                record = _score_record(
                    scorer, name, second, disturbance, rollout.residual, rollout.output,
                )
                record["first_path_number"] = first + 1
                record["second_path_number"] = second + 1
                record["post_switch_first_window_primary_db"] = float(
                    record["window_results"][3]["primary_score_db"]
                )
                post_switch.append(record["post_switch_first_window_primary_db"])
                records.append(record)
    if was_training:
        model.train()
    result = _aggregate(records)
    result["phase3_selection_score"] = phase3_selection_score(result)
    result["post_switch_first_window_primary_db"] = float(np.mean(post_switch))
    result["worst_post_switch_first_window_primary_db"] = float(min(post_switch))
    if include_records:
        result["records"] = records
    return result


def phase3_development_gate(baseline: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    checks = {
        "primary_drop_at_most_0_5_db": candidate["primary_score_db"] >= baseline["primary_score_db"] - 0.5,
        "rebound_increase_at_most_0_3_db": candidate["rebound_score_db"] <= baseline["rebound_score_db"] + 0.3,
        "worst_path_at_least_1_5_db": candidate["worst_path_primary_db"] >= 1.5,
        "first_window_drop_at_most_0_25_db": candidate["first_window_primary_db"] >= baseline["first_window_primary_db"] - 0.25,
        "controller_peak_below_0_98": candidate["controller_peak_abs"] < 0.98,
    }
    return {"passed": all(checks.values()), "checks": checks}


def phase3_oracle_gate(baseline: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    checks = {
        "primary_drop_at_most_0_5_db": candidate["primary_score_db"] >= baseline["primary_score_db"] - 0.5,
        "rebound_increase_at_most_0_3_db": candidate["rebound_score_db"] <= baseline["rebound_score_db"] + 0.3,
        "worst_path_at_least_2_db": candidate["worst_path_primary_db"] >= 2.0,
        "controller_peak_below_0_98": candidate["controller_peak_abs"] < 0.98,
    }
    return {"passed": all(checks.values()), "checks": checks}


def phase3_final_acceptance(
    baseline_dev: dict[str, Any], baseline_final: dict[str, Any],
    candidate_dev: dict[str, Any], candidate_final: dict[str, Any],
    real_time_factor: float,
) -> dict[str, Any]:
    def combine(first: dict[str, Any], second: dict[str, Any], key: str) -> float:
        return float((8.0 * first[key] + 2.0 * second[key]) / 10.0)
    baseline_s = combine(baseline_dev, baseline_final, "primary_score_db")
    baseline_r = combine(baseline_dev, baseline_final, "rebound_score_db")
    candidate_s = combine(candidate_dev, candidate_final, "primary_score_db")
    candidate_r = combine(candidate_dev, candidate_final, "rebound_score_db")
    baseline_c = 0.7 * baseline_s - 0.3 * baseline_r
    candidate_c = 0.7 * candidate_s - 0.3 * candidate_r
    worst = min(candidate_dev["worst_path_primary_db"], candidate_final["worst_path_primary_db"])
    path10 = float(candidate_final["path_metrics"]["10"]["primary_score_db"])
    unseen_gain = float(candidate_final["primary_score_db"] - baseline_final["primary_score_db"])
    peak = max(candidate_dev["controller_peak_abs"], candidate_final["controller_peak_abs"])
    checks = {
        "global_primary_drop_at_most_0_5_db": candidate_s >= baseline_s - 0.5,
        "global_rebound_increase_at_most_0_3_db": candidate_r <= baseline_r + 0.3,
        "global_composite_drop_at_most_0_2": candidate_c >= baseline_c - 0.2,
        "worst_path_at_least_2_db": worst >= 2.0,
        "path10_at_least_2_5_db": path10 >= 2.5,
        "unseen_average_gain_at_least_0_75_db": unseen_gain >= 0.75,
        "controller_peak_below_0_98": peak < 0.98,
        "cpu_real_time_factor_at_most_1": real_time_factor <= 1.0,
    }
    return {
        "passed": all(checks.values()), "checks": checks,
        "baseline_global": {"primary_score_db": baseline_s, "rebound_score_db": baseline_r, "composite": baseline_c},
        "candidate_global": {"primary_score_db": candidate_s, "rebound_score_db": candidate_r, "composite": candidate_c},
        "worst_path_primary_db": worst, "path10_primary_db": path10,
        "unseen_average_gain_db": unseen_gain, "controller_peak_abs": peak,
        "cpu_real_time_factor": real_time_factor,
    }
