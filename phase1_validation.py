"""Official protocol-v6 validation for Phase 1 models."""

from __future__ import annotations

import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict

import numpy as np
import torch

from dataset import apply_dynamic_path
from phase1_data import build_validation_manifest, iter_validation_examples
from v6_metrics import INITIALIZATION_SAMPLES, SCORING_WINDOW_SAMPLES


def _load_official_scorer():
    kit_dir = Path(__file__).resolve().parent / "DEEPANC_PARTICIPANT_KIT"
    if not (kit_dir / "public_demo_scoring.py").is_file():
        raise FileNotFoundError(
            "DEEPANC_PARTICIPANT_KIT/public_demo_scoring.py is required."
        )
    if str(kit_dir) not in sys.path:
        sys.path.insert(0, str(kit_dir))
    from public_demo_scoring import score_windowed_signals
    return score_windowed_signals


def _aggregate(records: list[Dict[str, object]]) -> Dict[str, object]:
    primary = np.asarray([record["primary_score_db"] for record in records])
    rebound = np.asarray([
        record["third_octave_rebound_peak_1000_8000_db"] for record in records
    ])
    if not np.all(np.isfinite(primary)) or not np.all(np.isfinite(rebound)):
        raise FloatingPointError("Official validation produced NaN or Inf metrics.")

    by_scene = defaultdict(list)
    by_path = defaultdict(list)
    all_windows = []
    for record in records:
        by_scene[str(record["scene_name"])].append(record)
        by_path[int(record["path_index_zero_based"])].append(record)
        all_windows.extend(record["window_results"])

    def summarize(group):
        return {
            "primary_score_db": float(np.mean([
                item["primary_score_db"] for item in group
            ])),
            "rebound_score_db": float(np.mean([
                item["third_octave_rebound_peak_1000_8000_db"] for item in group
            ])),
        }

    path_metrics = {
        str(path_index + 1): summarize(group)
        for path_index, group in sorted(by_path.items())
    }
    mean_primary = float(primary.mean())
    mean_rebound = float(rebound.mean())
    return {
        "primary_score_db": mean_primary,
        "rebound_score_db": mean_rebound,
        "selection_score": 0.7 * mean_primary - 0.3 * mean_rebound,
        "first_window_primary_db": float(np.mean([
            record["window_results"][0]["primary_score_db"] for record in records
        ])),
        "worst_window_primary_db": float(min(
            window["primary_score_db"] for window in all_windows
        )),
        "worst_path_primary_db": float(min(
            value["primary_score_db"] for value in path_metrics.values()
        )),
        "largest_single_window_rebound_db": float(max(
            window["third_octave_rebound_peak_1000_8000_db"]
            for window in all_windows
        )),
        "controller_peak_abs": float(max(
            record["full_controller_peak_abs"] for record in records
        )),
        "scene_metrics": {
            name: summarize(group) for name, group in sorted(by_scene.items())
        },
        "path_metrics": path_metrics,
        "record_count": len(records),
    }


def evaluate_v6_model(
    model: torch.nn.Module,
    dataset_dir: str | Path,
    device: torch.device,
    manifest: Dict[str, object] | None = None,
    include_records: bool = True,
) -> Dict[str, object]:
    scorer = _load_official_scorer()
    manifest = manifest or build_validation_manifest(dataset_dir)
    was_training = model.training
    model.eval()
    records = []

    with torch.inference_mode():
        for scene_name, path_index, x, secondary_path, disturbance in (
            iter_validation_examples(dataset_dir, manifest)
        ):
            x = x.unsqueeze(0).to(device)
            secondary_path = secondary_path.unsqueeze(0).to(device)
            disturbance = disturbance.unsqueeze(0).to(device)
            controller = model(x)
            residual = disturbance - apply_dynamic_path(controller, secondary_path)
            scored = slice(INITIALIZATION_SAMPLES, None)
            metrics = scorer(
                disturbance[0, scored].cpu().numpy(),
                residual[0, scored].cpu().numpy(),
                controller[0, scored].cpu().numpy(),
                sample_rate=48_000,
                window_samples=SCORING_WINDOW_SAMPLES,
            )
            records.append({
                "scene_name": scene_name,
                "path_index_zero_based": path_index,
                "path_number": path_index + 1,
                "full_controller_peak_abs": float(controller.abs().amax().cpu()),
                **metrics,
            })

    if was_training:
        model.train()
    result = _aggregate(records)
    if include_records:
        result["records"] = records
    return result

