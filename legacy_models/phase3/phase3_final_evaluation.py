"""One-shot three-seed Phase-3 evaluation on final-only paths 9/10."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from phase1_validation import _aggregate
from phase3_validation import (
    build_phase3_manifests,
    evaluate_phase3_feedback,
    phase3_final_acceptance,
)
from legacy_models.phase0_phase1.train import resolve_device
from legacy_models.phase3.train_phase3 import load_phase3


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite-summaries", nargs=3, required=True)
    parser.add_argument("--runtime-reports", nargs=3, required=True)
    parser.add_argument("--lopo-summary", required=True)
    parser.add_argument("--dataset-dir", default="dataset")
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    lopo = json.loads(Path(args.lopo_summary).read_text(encoding="utf-8"))
    if not lopo["acceptance"]["passed"]:
        raise RuntimeError("LOPO did not pass; final-only paths must remain sealed.")
    suite_values = [json.loads(Path(path).read_text(encoding="utf-8")) for path in args.suite_summaries]
    if not all(value.get("ready_for_lopo") for value in suite_values):
        raise RuntimeError("Every seed must provide a development candidate before final evaluation.")
    runtime_values = [json.loads(Path(path).read_text(encoding="utf-8")) for path in args.runtime_reports]
    if not all(value.get("requires_error") is True for value in runtime_values):
        raise RuntimeError("Runtime reports must come from feedback submissions.")

    p1 = json.loads(Path("runs/phase1_suite_seed2026/P1-E2/summary.json").read_text(encoding="utf-8"))
    baseline_records = p1["final_metrics"]["records"]
    baseline_dev = _aggregate([record for record in baseline_records if record["path_number"] <= 8])
    baseline_final = _aggregate([record for record in baseline_records if record["path_number"] >= 9])
    manifests = build_phase3_manifests(args.dataset_dir)
    device = resolve_device(args.device)
    results = []
    for suite, runtime in zip(suite_values, runtime_values):
        model, _ = load_phase3(Path(suite["selected_checkpoint"]), device)
        dev = evaluate_phase3_feedback(model, args.dataset_dir, device, manifests["development"], include_records=True)
        final = evaluate_phase3_feedback(model, args.dataset_dir, device, manifests["final"], include_records=True)
        acceptance = phase3_final_acceptance(
            baseline_dev, baseline_final, dev, final, float(runtime["real_time_factor"]),
        )
        results.append({
            "checkpoint": suite["selected_checkpoint"], "development": dev,
            "final_unseen": final, "runtime_report": runtime, "acceptance": acceptance,
        })
    passed = all(value["acceptance"]["passed"] for value in results)
    selected = (
        max(results, key=lambda value: value["acceptance"]["candidate_global"]["composite"])["checkpoint"]
        if passed else "runs/phase1_suite_seed2026/P1-E2/checkpoints/best_official_composite.pt"
    )
    summary = {
        "seed_results": results, "all_three_seeds_passed": passed,
        "phase3_passed": passed, "selected_checkpoint": selected,
        "final_paths_touched": True,
    }
    Path(args.output).write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
