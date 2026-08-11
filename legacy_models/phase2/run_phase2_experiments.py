"""Run P2-E0/E1/E2, the deterministic E3 correction, and optional P2-E4."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--dataset-dir", default="dataset")
    parser.add_argument("--checkpoint", default="runs/phase1_suite_seed2026/P1-E2/checkpoints/best_official_composite.pt")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--final-epochs", type=int, default=20)
    parser.add_argument("--samples-per-epoch", type=int, default=256)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--final-seeds", default="2026,2027,2028")
    parser.add_argument("--run-final", action="store_true", help="Train P2-E4 and touch final-only paths 9/10.")
    return parser.parse_args()


def run_training(common: list[str], output: Path, extra: list[str]) -> dict[str, Any]:
    command = [
        sys.executable, "-u", "-m", "legacy_models.phase2.train_phase2",
        *common, "--output-dir", str(output), *extra,
    ]
    subprocess.run(command, check=True)
    return json.loads((output / "summary.json").read_text(encoding="utf-8"))


def compact(summary: dict[str, Any]) -> dict[str, Any]:
    dev = summary["final_development_metrics"]
    stress = summary["final_stress_metrics"]
    result = {
        "mode": summary["mode"], "best_epoch": summary["best_epoch"],
        "selected_checkpoint": summary["selected_checkpoint"],
        "development": {key: dev[key] for key in (
            "primary_score_db", "rebound_score_db", "selection_score",
            "robust_development_score", "worst_path_primary_db", "controller_peak_abs",
        )},
        "stress": {key: stress[key] for key in (
            "primary_score_db", "rebound_score_db", "robust_development_score", "worst_path_primary_db",
        )},
        "gate": summary["development_gate"],
    }
    if "final_unseen_metrics" in summary:
        final = summary["final_unseen_metrics"]
        result["final_unseen"] = {key: final[key] for key in (
            "primary_score_db", "rebound_score_db", "selection_score",
            "worst_path_primary_db", "controller_peak_abs",
        )}
    return result


def main() -> None:
    args = parse_args()
    root = Path(args.output_root or f"runs/phase2_suite_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    root.mkdir(parents=True, exist_ok=False)
    common = [
        "--dataset-dir", args.dataset_dir, "--checkpoint", args.checkpoint,
        "--epochs", str(args.epochs), "--samples-per-epoch", str(args.samples_per_epoch),
        "--device", args.device, "--seed", str(args.seed),
    ]
    candidates: dict[str, dict[str, Any]] = {}
    candidates["P2-E0"] = run_training(common, root / "P2-E0", ["--experiment", "control"])
    candidates["P2-E1"] = run_training(common, root / "P2-E1", ["--experiment", "real"])
    candidates["P2-E2"] = run_training(common, root / "P2-E2", ["--experiment", "augment"])

    baseline = candidates["P2-E2"]["baseline_development_metrics"]
    e2 = candidates["P2-E2"]["final_development_metrics"]
    e3_reason = None
    e3_args: list[str] = []
    if e2["primary_score_db"] < baseline["primary_score_db"] - 0.5 or e2["rebound_score_db"] > baseline["rebound_score_db"] + 0.3:
        e3_reason = "E2 violated the primary/rebound development envelope"
        e3_args = ["--beta", "0.10", "--augmentation-probability", "0.5"]
    else:
        baseline_worst = baseline["worst_path_primary_db"]
        if e2["worst_path_primary_db"] - baseline_worst < 0.5:
            e3_reason = "E2 worst-path gain was below 0.5 dB"
            e3_args = ["--beta", "0.50"]
    if e3_reason:
        candidates["P2-E3"] = run_training(
            common, root / "P2-E3", ["--experiment", "augment", *e3_args],
        )

    eligible = {name: result for name, result in candidates.items() if result["development_gate"]["passed"]}
    winner = max(
        eligible,
        key=lambda name: eligible[name]["final_development_metrics"]["robust_development_score"],
        default=None,
    )
    robust_attempts = [candidates[name] for name in ("P2-E2", "P2-E3") if name in candidates]
    best_robust_attempt_worst_path = max(
        (result["final_development_metrics"]["worst_path_primary_db"] for result in robust_attempts),
        default=-float("inf"),
    )
    stopped_for_feedback = best_robust_attempt_worst_path < 1.5
    final_results = []
    if args.run_final and winner is not None and not stopped_for_feedback:
        winner_config = json.loads((root / winner / "config.json").read_text(encoding="utf-8"))
        final_seeds = [int(item.strip()) for item in args.final_seeds.split(",") if item.strip()]
        for final_seed in final_seeds:
            final_extra = [
                "--experiment", winner_config["experiment"], "--epochs", str(args.final_epochs),
                "--beta", str(winner_config["beta_resolved"]),
                "--augmentation-probability", str(winner_config["augmentation_probability"]),
                "--seed", str(final_seed), "--evaluate-final",
            ]
            final_results.append(run_training(common, root / f"P2-E4-seed{final_seed}", final_extra))

    final_passed = bool(final_results) and all(
        result["phase2_acceptance"]["passed"] for result in final_results
    )
    suite = {
        "initial_checkpoint": args.checkpoint,
        "candidate_results": {name: compact(value) for name, value in candidates.items()},
        "e3_trigger_reason": e3_reason, "selected_development_experiment": winner,
        "best_robust_attempt_worst_path_primary_db": best_robust_attempt_worst_path,
        "phase2_stopped_for_feedback": stopped_for_feedback,
        "phase2_passed": final_passed,
        "selected_checkpoint": (
            args.checkpoint if stopped_for_feedback or not final_passed
            else max(final_results, key=lambda result: result["final_development_metrics"]["robust_development_score"])["selected_checkpoint"]
        ),
        "p2_e4_ran": bool(final_results),
        "final_results": [compact(result) | {"acceptance": result["phase2_acceptance"]} for result in final_results],
        "all_final_seeds_passed": final_passed,
        "final_paths_touched": bool(final_results),
    }
    (root / "suite_summary.json").write_text(json.dumps(suite, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(suite, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
