"""Paired leave-one-path-out evaluation for the winning Phase-2 configuration."""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
from datetime import datetime
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--dataset-dir", default="dataset")
    parser.add_argument("--checkpoint", default="runs/phase1_suite_seed2026/P1-E2/checkpoints/best_official_composite.pt")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--samples-per-epoch", type=int, default=128)
    parser.add_argument("--beta", type=float, default=0.25)
    parser.add_argument("--augmentation-probability", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def run_one(args: argparse.Namespace, output: Path, holdout: int, experiment: str) -> dict:
    train_paths = ",".join(str(number) for number in range(1, 9) if number != holdout)
    command = [
        sys.executable, "-u", "-m", "legacy_models.phase2.train_phase2",
        "--dataset-dir", args.dataset_dir,
        "--checkpoint", args.checkpoint, "--output-dir", str(output),
        "--experiment", experiment, "--train-paths", train_paths, "--dev-paths", str(holdout),
        "--epochs", str(args.epochs), "--samples-per-epoch", str(args.samples_per_epoch),
        "--beta", str(args.beta), "--augmentation-probability", str(args.augmentation_probability),
        "--seed", str(args.seed), "--device", args.device,
    ]
    subprocess.run(command, check=True)
    return json.loads((output / "summary.json").read_text(encoding="utf-8"))


def main() -> None:
    args = parse_args()
    root = Path(args.output_root or f"runs/phase2_lopo_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    root.mkdir(parents=True, exist_ok=False)
    folds = []
    improvements = []
    rebound_deltas = []
    for holdout in range(1, 9):
        control = run_one(args, root / f"path_{holdout:02d}_control", holdout, "control")
        robust = run_one(args, root / f"path_{holdout:02d}_augment", holdout, "augment")
        control_metrics = control["final_development_metrics"]
        robust_metrics = robust["final_development_metrics"]
        primary_gain = robust_metrics["primary_score_db"] - control_metrics["primary_score_db"]
        rebound_delta = robust_metrics["rebound_score_db"] - control_metrics["rebound_score_db"]
        improvements.append(primary_gain)
        rebound_deltas.append(rebound_delta)
        folds.append({
            "held_out_path": holdout,
            "control_checkpoint": control["selected_checkpoint"],
            "robust_checkpoint": robust["selected_checkpoint"],
            "primary_gain_db": primary_gain, "rebound_change_db": rebound_delta,
        })
    checks = {
        "median_primary_gain_at_least_0_5_db": statistics.median(improvements) >= 0.5,
        "at_least_6_of_8_non_degrading": sum(value >= 0 for value in improvements) >= 6,
        "every_fold_rebound_increase_at_most_0_3_db": max(rebound_deltas) <= 0.3,
    }
    summary = {
        "folds": folds, "median_primary_gain_db": statistics.median(improvements),
        "non_degrading_fold_count": sum(value >= 0 for value in improvements),
        "worst_rebound_change_db": max(rebound_deltas),
        "acceptance": {"passed": all(checks.values()), "checks": checks},
        "final_paths_touched": False,
    }
    (root / "lopo_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
