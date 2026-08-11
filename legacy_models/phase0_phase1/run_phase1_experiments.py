"""Run the bounded P1-E0/E1/E2/E3 experiment sequence."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--dataset-dir", default="dataset")
    parser.add_argument(
        "--checkpoint",
        default="runs/phase0_60ep_seed2026/checkpoints/best_mean_nr.pt",
    )
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--samples-per-epoch", type=int, default=256)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def _run(common: list[str], output_dir: Path, extra: list[str]) -> dict:
    command = [
        sys.executable, "-u", "-m", "legacy_models.phase0_phase1.train_phase1",
        *common, "--output-dir", str(output_dir), *extra,
    ]
    subprocess.run(command, check=True)
    return json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))


def _compact_result(result: dict) -> dict:
    if result.get("mode") == "P1-E0":
        metrics = result["metrics"]
        return {
            "mode": result["mode"],
            "checkpoint": result["checkpoint"],
            "metrics": {
                key: metrics[key] for key in (
                    "primary_score_db", "rebound_score_db", "selection_score",
                    "worst_path_primary_db", "controller_peak_abs",
                )
            },
        }
    return {
        "mode": result["mode"],
        "best_epoch": result["best_epoch"],
        "best_checkpoint": result["best_checkpoint"],
        "loss_weights": result["loss_weights"],
        "final_metrics": {
            key: result["final_metrics"][key] for key in (
                "primary_score_db", "rebound_score_db", "selection_score",
                "first_window_primary_db", "worst_window_primary_db",
                "worst_path_primary_db", "controller_peak_abs",
            )
        },
        "acceptance": result["acceptance"],
    }


def main() -> None:
    args = parse_args()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    root = Path(args.output_root or f"runs/phase1_suite_{timestamp}")
    root.mkdir(parents=True, exist_ok=False)
    common = [
        "--dataset-dir", args.dataset_dir,
        "--checkpoint", args.checkpoint,
        "--epochs", str(args.epochs),
        "--samples-per-epoch", str(args.samples_per_epoch),
        "--device", args.device,
    ]

    e0 = _run(common, root / "P1-E0", ["--evaluate-only"])
    e1 = _run(common, root / "P1-E1", ["--experiment", "band"])
    e2 = _run(common, root / "P1-E2", ["--experiment", "composite"])
    baseline = e0["metrics"]
    e2_metrics = e2["final_metrics"]

    e3 = None
    e3_reason = None
    if e2_metrics["primary_score_db"] < baseline["primary_score_db"] - 0.25:
        e3_reason = "E2 primary score dropped by more than 0.25 dB"
        weights = (0.85, 0.15)
    else:
        rebound_reduction = (
            (baseline["rebound_score_db"] - e2_metrics["rebound_score_db"])
            / baseline["rebound_score_db"]
        )
        weights = (0.60, 0.40)
        if rebound_reduction < 0.20:
            e3_reason = "E2 rebound reduction was below 20 percent"
    if e3_reason is not None:
        e3 = _run(common, root / "P1-E3", [
            "--experiment", "custom",
            "--primary-weight", str(weights[0]),
            "--rebound-weight", str(weights[1]),
        ])

    candidates = {"P1-E1": e1, "P1-E2": e2}
    if e3 is not None:
        candidates["P1-E3"] = e3
    passing = {
        name: result for name, result in candidates.items()
        if result["acceptance"]["passed"]
    }
    selected_name = max(
        passing,
        key=lambda name: passing[name]["final_metrics"]["selection_score"],
        default=None,
    )
    suite_summary = {
        "baseline": _compact_result(e0),
        "candidates": {
            name: _compact_result(result) for name, result in candidates.items()
        },
        "e3_trigger_reason": e3_reason,
        "selected_experiment": selected_name,
        "phase1_passed": selected_name is not None,
        "selected_checkpoint": (
            passing[selected_name]["best_checkpoint"] if selected_name else args.checkpoint
        ),
    }
    (root / "suite_summary.json").write_text(
        json.dumps(suite_summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(suite_summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
