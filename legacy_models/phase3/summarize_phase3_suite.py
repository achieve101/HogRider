"""Recover a Phase-3 suite summary from completed stage artifacts.

This is intentionally read-only with respect to checkpoints and histories.  It
is useful when the bounded experiment runner was stopped between stages.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


DEFAULT_P1 = "runs/phase1_suite_seed2026/P1-E2/checkpoints/best_official_composite.pt"
METRIC_KEYS = (
    "primary_score_db",
    "rebound_score_db",
    "selection_score",
    "phase3_selection_score",
    "worst_path_primary_db",
    "first_window_primary_db",
    "controller_peak_abs",
    "route_accuracy_after_initialization",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--p1-checkpoint", default=DEFAULT_P1)
    return parser.parse_args()


def compact_summary(summary: dict[str, Any]) -> dict[str, Any]:
    metrics = summary["final_development_metrics"]
    return {
        "stage": summary["stage"],
        "best_epoch": summary["best_epoch"],
        "selected_checkpoint": summary["selected_checkpoint"],
        "metrics": {key: metrics.get(key) for key in METRIC_KEYS},
        "acceptance": summary.get("acceptance"),
        "complexity": summary.get("complexity"),
        "stop_reason": summary.get("stop_reason"),
    }


def recover_from_history(stage_dir: Path) -> dict[str, Any]:
    json_path = stage_dir / "history.json"
    jsonl_path = stage_dir / "history.jsonl"
    if json_path.exists():
        history = json.loads(json_path.read_text(encoding="utf-8"))
    else:
        history = [
            json.loads(line)
            for line in jsonl_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    best = max(
        history,
        key=lambda item: item["validation"]["development"]["phase3_selection_score"],
    )
    metrics = dict(best["validation"]["development"])
    metrics.update(best["validation"].get("gate", {}))
    baseline = json.loads(
        (stage_dir / "baseline_development_metrics.json").read_text(encoding="utf-8")
    )
    config = json.loads((stage_dir / "config.json").read_text(encoding="utf-8"))
    checks = {
        "primary_drop_at_most_0_5_db": metrics["primary_score_db"] >= baseline["primary_score_db"] - 0.5,
        "rebound_increase_at_most_0_3_db": metrics["rebound_score_db"] <= baseline["rebound_score_db"] + 0.3,
        "worst_path_at_least_1_5_db": metrics["worst_path_primary_db"] >= 1.5,
        "first_window_drop_at_most_0_25_db": metrics["first_window_primary_db"] >= baseline["first_window_primary_db"] - 0.25,
        "controller_peak_below_0_98": metrics["controller_peak_abs"] < 0.98,
    }
    return {
        "stage": best.get("stage", "gate"),
        "best_epoch": best["epoch"],
        "selected_checkpoint": str(stage_dir / "checkpoints" / "best_phase3_selection.pt"),
        "metrics": {key: metrics.get(key) for key in METRIC_KEYS},
        "acceptance": {"passed": all(checks.values()), "checks": checks},
        "complexity": best.get("complexity") or config.get("complexity"),
        "stop_reason": "runner interrupted after the recorded epoch; recovered from history",
    }


def main() -> None:
    args = parse_args()
    root = Path(args.root)
    results: dict[str, dict[str, Any]] = {}
    for stage_dir in sorted(path for path in root.glob("P3-E*") if path.is_dir()):
        summary_path = stage_dir / "summary.json"
        history_path = stage_dir / "history.json"
        history_jsonl_path = stage_dir / "history.jsonl"
        if summary_path.exists():
            results[stage_dir.name] = compact_summary(
                json.loads(summary_path.read_text(encoding="utf-8"))
            )
        elif history_path.exists() or history_jsonl_path.exists():
            results[stage_dir.name] = recover_from_history(stage_dir)

    oracle_name = "P3-E1b" if "P3-E1b" in results else "P3-E1"
    gate_names = [name for name in ("P3-E2", "P3-E2b") if name in results]
    passing = [name for name in gate_names if (results[name].get("acceptance") or {}).get("passed")]
    winner = max(
        passing,
        key=lambda name: results[name]["metrics"]["phase3_selection_score"],
        default=None,
    )
    oracle_passed = bool((results.get(oracle_name, {}).get("acceptance") or {}).get("passed"))
    stopped = not oracle_passed or winner is None
    suite = {
        "initial_p1_checkpoint": args.p1_checkpoint,
        "results": results,
        "oracle_experiment": oracle_name,
        "oracle_passed": oracle_passed,
        "selected_development_experiment": winner,
        "ready_for_lopo": winner is not None,
        "phase3_stopped": stopped,
        "stop_reason": (
            "oracle FIR capacity threshold failed"
            if not oracle_passed
            else "both hidden-24 and hidden-32 feedback gates failed development thresholds"
            if winner is None
            else None
        ),
        "selected_checkpoint": results[winner]["selected_checkpoint"] if winner else args.p1_checkpoint,
        "joint_finetuning_ran": any(name in results for name in ("P3-E3", "P3-E3b")),
        "lopo_ran": False,
        "final_paths_touched": False,
    }
    output = root / "suite_summary.json"
    output.write_text(json.dumps(suite, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(suite, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
