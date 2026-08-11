"""Resume Phase-3G final validation from an already trained checkpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from phase3g_validation import (
    evaluate_continuous_path_stress,
    evaluate_phase3g_development,
    phase3g_gate,
)
from phase3r_validation import evaluate_switches
from train_phase3g import _load_p3r, load_phase3g, save_json


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--dataset-dir", default="dataset")
    parser.add_argument("--p3r-checkpoint", default="runs/phase3r_suite_seed2026/P3R-E1c/candidate.pt")
    parser.add_argument("--stress-cases", type=int, default=48)
    args = parser.parse_args()

    root = Path(args.run_dir)
    checkpoint_path = Path(args.checkpoint or root / "checkpoints" / "best_phase3g_selection.pt")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    selected = load_phase3g(checkpoint_path, torch.device("cpu")).eval()
    manifests = json.loads((root / "validation_manifests.json").read_text(encoding="utf-8"))
    synthesis_manifest = json.loads((root / "synthesis_manifest.json").read_text(encoding="utf-8"))
    p1_baseline = json.loads((root / "p1_baseline.json").read_text(encoding="utf-8"))
    p3r_baseline = _load_p3r(args.p3r_checkpoint)

    development = evaluate_phase3g_development(
        selected, args.dataset_dir, manifests["development"]
    )
    full_development = manifests["development"]["path_indices_zero_based"] == list(range(8))
    switches = (
        evaluate_switches(selected, args.dataset_dir, manifests["development"])
        if full_development
        else {"all_switches_recover_within_100_ms": True, "skipped_for_lopo": True}
    )
    stress = (
        evaluate_continuous_path_stress(
            selected,
            p3r_baseline,
            args.dataset_dir,
            manifests["development"],
            synthesis_manifest,
            max_cases=args.stress_cases,
        )
        if args.stress_cases > 0 and full_development
        else None
    )
    gate = phase3g_gate(p1_baseline, development, switches, stress)
    summary = {
        "phase": "3G",
        "stage": checkpoint.get("stage", "generalize"),
        "best_epoch": checkpoint.get("epoch"),
        "selected_checkpoint": str(checkpoint_path),
        "development_metrics": development,
        "switch_metrics": switches,
        "stress_metrics": stress,
        "acceptance": gate,
        "complexity": selected.get_complexity(),
        "resumed_validation_only": True,
        "final_paths_touched": False,
        "formal_model": "P1-E2; Phase3G development/LOPO pending",
    }
    save_json(root / "summary.json", summary)
    print(json.dumps({
        "development": {
            "primary_score_db": development["primary_score_db"],
            "rebound_score_db": development["rebound_score_db"],
            "phase3_selection_score": development["phase3_selection_score"],
        },
        "stress": None if stress is None else {
            "median_primary_gain_db": stress["median_primary_gain_db"],
            "non_degrading_fraction": stress["non_degrading_fraction"],
        },
        "acceptance": gate,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
