"""Run the bounded Phase-3 common/oracle/feedback experiment decision tree."""

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
    parser.add_argument("--p1-checkpoint", default="runs/phase1_suite_seed2026/P1-E2/checkpoints/best_official_composite.pt")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--max-train-batches", type=int, default=None)
    return parser.parse_args()


def run_stage(common: list[str], output: Path, stage: str, checkpoint: str, extra: list[str] | None = None) -> dict[str, Any]:
    command = [
        sys.executable, "-u", "-m", "legacy_models.phase3.train_phase3", "--stage", stage,
        "--checkpoint", checkpoint, "--output-dir", str(output), *common,
        *(extra or []),
    ]
    subprocess.run(command, check=True)
    return json.loads((output / "summary.json").read_text(encoding="utf-8"))


def compact(value: dict[str, Any]) -> dict[str, Any]:
    metrics = value["final_development_metrics"]
    return {
        "stage": value["stage"], "best_epoch": value["best_epoch"],
        "selected_checkpoint": value["selected_checkpoint"],
        "metrics": {key: metrics.get(key) for key in (
            "primary_score_db", "rebound_score_db", "selection_score",
            "phase3_selection_score", "worst_path_primary_db",
            "first_window_primary_db", "controller_peak_abs",
            "route_accuracy_after_initialization",
        )},
        "acceptance": value.get("acceptance"), "complexity": value["complexity"],
    }


def main() -> None:
    args = parse_args()
    root = Path(args.output_root or f"runs/phase3_suite_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    root.mkdir(parents=True, exist_ok=False)
    common = ["--dataset-dir", args.dataset_dir, "--device", args.device, "--seed", str(args.seed)]
    if args.max_train_batches:
        common += ["--max-train-batches", str(args.max_train_batches)]
    results: dict[str, dict[str, Any]] = {}
    results["P3-E0"] = run_stage(common, root / "P3-E0", "common", args.p1_checkpoint)
    common_checkpoint = results["P3-E0"]["selected_checkpoint"]
    results["P3-E1"] = run_stage(common, root / "P3-E1", "expert", common_checkpoint)
    oracle_name = "P3-E1"
    if not results[oracle_name]["acceptance"]["passed"]:
        results["P3-E0b"] = run_stage(
            common, root / "P3-E0b", "common", args.p1_checkpoint, ["--fir-length", "4096"],
        )
        results["P3-E1b"] = run_stage(
            common, root / "P3-E1b", "expert", results["P3-E0b"]["selected_checkpoint"],
        )
        oracle_name = "P3-E1b"

    stopped = not results[oracle_name]["acceptance"]["passed"]
    selected = args.p1_checkpoint
    feedback_candidates: dict[str, dict[str, Any]] = {}
    if not stopped:
        oracle_checkpoint = results[oracle_name]["selected_checkpoint"]
        results["P3-E2"] = run_stage(common, root / "P3-E2", "gate", oracle_checkpoint)
        feedback_candidates = {"P3-E2": results["P3-E2"]}
        gate_name = "P3-E2"
        if not results[gate_name]["acceptance"]["passed"]:
            results["P3-E2b"] = run_stage(
                common, root / "P3-E2b", "gate", oracle_checkpoint, ["--gate-hidden-size", "32"],
            )
            feedback_candidates["P3-E2b"] = results["P3-E2b"]
            gate_name = "P3-E2b"

        # Joint fine-tuning cannot rescue a gate that does not satisfy the
        # development acoustic/safety thresholds.  Only enter P3-E3 after the
        # bounded gate correction (if needed) has passed on its own.
        if results[gate_name]["acceptance"]["passed"]:
            joint_name = "P3-E3" if gate_name == "P3-E2" else "P3-E3b"
            results[joint_name] = run_stage(
                common, root / joint_name, "joint", results[gate_name]["selected_checkpoint"],
            )
            feedback_candidates[joint_name] = results[joint_name]
        passing = {name: value for name, value in feedback_candidates.items() if value["acceptance"]["passed"]}
        if passing:
            winner = max(passing, key=lambda name: passing[name]["final_development_metrics"]["phase3_selection_score"])
            selected = passing[winner]["selected_checkpoint"]
        else:
            winner = None
            stopped = True
    else:
        winner = None

    suite = {
        "initial_p1_checkpoint": args.p1_checkpoint,
        "results": {name: compact(value) for name, value in results.items()},
        "oracle_experiment": oracle_name,
        "oracle_passed": results[oracle_name]["acceptance"]["passed"],
        "selected_development_experiment": winner,
        "ready_for_lopo": winner is not None,
        "phase3_stopped": stopped,
        "selected_checkpoint": selected,
        "final_paths_touched": False,
    }
    (root / "suite_summary.json").write_text(json.dumps(suite, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(suite, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
