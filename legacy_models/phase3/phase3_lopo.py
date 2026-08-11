"""Eight-fold leave-one-path-out evaluation for a fixed Phase-3 feedback configuration."""

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
    parser.add_argument("--common-checkpoint", required=True)
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--dataset-dir", default="dataset")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=2026)
    return parser.parse_args()


def run(output: Path, args: argparse.Namespace, stage: str, checkpoint: str, train_paths: str, dev_path: int) -> dict:
    command = [
        sys.executable, "-u", "-m", "legacy_models.phase3.train_phase3", "--stage", stage,
        "--dataset-dir", args.dataset_dir, "--checkpoint", checkpoint,
        "--output-dir", str(output), "--train-paths", train_paths,
        "--dev-paths", str(dev_path), "--device", args.device, "--seed", str(args.seed),
    ]
    subprocess.run(command, check=True)
    return json.loads((output / "summary.json").read_text(encoding="utf-8"))


def main() -> None:
    args = parse_args()
    root = Path(args.output_root or f"runs/phase3_lopo_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    root.mkdir(parents=True, exist_ok=False)
    p1 = json.loads(Path("runs/phase1_suite_seed2026/P1-E2/summary.json").read_text(encoding="utf-8"))
    p1_paths = p1["final_metrics"]["path_metrics"]
    folds, gains = [], []
    for held_out in range(1, 9):
        train_paths = ",".join(str(value) for value in range(1, 9) if value != held_out)
        expert = run(root / f"path_{held_out:02d}_expert", args, "expert", args.common_checkpoint, train_paths, held_out)
        gate = run(root / f"path_{held_out:02d}_gate", args, "gate", expert["selected_checkpoint"], train_paths, held_out)
        joint = run(root / f"path_{held_out:02d}_joint", args, "joint", gate["selected_checkpoint"], train_paths, held_out)
        candidate = joint["final_development_metrics"]
        baseline = float(p1_paths[str(held_out)]["primary_score_db"])
        gain = float(candidate["primary_score_db"] - baseline)
        gains.append(gain)
        folds.append({
            "held_out_path": held_out, "baseline_primary_db": baseline,
            "candidate_primary_db": candidate["primary_score_db"], "primary_gain_db": gain,
            "candidate_rebound_db": candidate["rebound_score_db"],
            "checkpoint": joint["selected_checkpoint"],
        })
    checks = {
        "median_gain_at_least_0_5_db": statistics.median(gains) >= 0.5,
        "at_least_6_of_8_non_degrading": sum(value >= 0 for value in gains) >= 6,
    }
    summary = {
        "folds": folds, "median_primary_gain_db": statistics.median(gains),
        "non_degrading_fold_count": sum(value >= 0 for value in gains),
        "acceptance": {"passed": all(checks.values()), "checks": checks},
        "final_paths_touched": False,
    }
    (root / "lopo_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
