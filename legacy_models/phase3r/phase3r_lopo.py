"""Eight-fold path-template/expert removal evaluation for Phase 3R."""

from __future__ import annotations

import argparse
import json
import statistics
from datetime import datetime
from pathlib import Path

import torch

from phase3_validation import build_phase3_manifests
from phase3r_model import InnovationRoutedFIRController
from phase3r_validation import evaluate_development


def remove_candidate(full: InnovationRoutedFIRController, held_out: int) -> InnovationRoutedFIRController:
    keep = [index for index in range(full.num_experts) if index != held_out]
    config = full.model_config
    config["num_experts"] = len(keep)
    reduced = InnovationRoutedFIRController(**config)
    with torch.no_grad():
        reduced.expert_filters.copy_(full.expert_filters[keep])
        reduced.primary_real.copy_(full.primary_real[keep])
        reduced.primary_imag.copy_(full.primary_imag[keep])
        reduced.secondary_paths.copy_(full.secondary_paths[keep])
        reduced.hann_window.copy_(full.hann_window)
        reduced.band_mask.copy_(full.band_mask)
    return reduced.eval()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default="runs/phase3r_suite_seed2026/P3R-E1c/candidate.pt")
    parser.add_argument("--dataset-dir", default="dataset")
    parser.add_argument("--p1-summary", default="runs/phase1_suite_seed2026/P1-E2/summary.json")
    parser.add_argument("--output-root", default=None)
    args = parser.parse_args()
    root = Path(args.output_root or f"runs/phase3r_lopo_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    root.mkdir(parents=True, exist_ok=False)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    full = InnovationRoutedFIRController(**checkpoint["model_config"])
    full.load_state_dict(checkpoint["model_state_dict"])
    p1 = json.loads(Path(args.p1_summary).read_text(encoding="utf-8"))["final_metrics"]["path_metrics"]
    base_manifest = build_phase3_manifests(args.dataset_dir)["development"]
    folds, gains = [], []
    for held_out in range(8):
        manifest = dict(base_manifest)
        manifest["path_indices_zero_based"] = [held_out]
        reduced = remove_candidate(full, held_out)
        metrics = evaluate_development(reduced, args.dataset_dir, manifest, include_records=False)
        baseline = float(p1[str(held_out + 1)]["primary_score_db"])
        gain = float(metrics["primary_score_db"] - baseline)
        gains.append(gain)
        folds.append({
            "held_out_path": held_out + 1,
            "removed_original_candidate_zero_based": held_out,
            "remaining_candidate_original_indices_zero_based": [index for index in range(8) if index != held_out],
            "baseline_primary_db": baseline,
            "candidate_primary_db": metrics["primary_score_db"],
            "primary_gain_db": gain,
            "candidate_rebound_db": metrics["rebound_score_db"],
            "cpu_real_time_factor": metrics["cpu_real_time_factor"],
        })
    checks = {
        "median_gain_at_least_0_5_db": statistics.median(gains) >= 0.5,
        "at_least_6_of_8_non_degrading": sum(value >= 0.0 for value in gains) >= 6,
    }
    summary = {
        "source_checkpoint": args.checkpoint, "folds": folds,
        "median_primary_gain_db": statistics.median(gains),
        "non_degrading_fold_count": sum(value >= 0.0 for value in gains),
        "acceptance": {"passed": all(checks.values()), "checks": checks},
        "fold_policy": "held-out expert, P_i, and S_i physically removed",
        "final_paths_touched": False,
    }
    (root / "lopo_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
