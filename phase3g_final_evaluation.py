"""Sealed-path three-seed final evaluation for Phase 3G."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from phase1_validation import _aggregate
from phase3_validation import build_phase3_manifests, phase3_final_acceptance
from phase3g_validation import evaluate_phase3g_development
from train_phase3g import load_phase3g


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed-summaries", nargs=3, required=True)
    parser.add_argument("--lopo-summary", required=True)
    parser.add_argument("--dataset-dir", default="dataset")
    parser.add_argument("--output", required=True)
    args=parser.parse_args()
    lopo=json.loads(Path(args.lopo_summary).read_text(encoding="utf-8"))
    if not lopo["acceptance"]["passed"]:
        raise RuntimeError("LOPO failed; paths 9/10 must remain sealed.")
    seed_values=[json.loads(Path(path).read_text(encoding="utf-8")) for path in args.seed_summaries]
    if not all(value["acceptance"]["passed"] and not value.get("final_paths_touched", True) for value in seed_values):
        raise RuntimeError("All three seeds must pass development before final evaluation.")
    p1=json.loads(Path("runs/phase1_suite_seed2026/P1-E2/summary.json").read_text(encoding="utf-8"))
    records=p1["final_metrics"]["records"]
    baseline_dev=_aggregate([value for value in records if value["path_number"] <= 8])
    baseline_final=_aggregate([value for value in records if value["path_number"] >= 9])
    manifests=build_phase3_manifests(args.dataset_dir)
    results=[]
    for value in seed_values:
        model=load_phase3g(value["selected_checkpoint"], torch.device("cpu")).eval()
        development=evaluate_phase3g_development(model, args.dataset_dir, manifests["development"])
        final=evaluate_phase3g_development(model, args.dataset_dir, manifests["final"])
        rtf=max(float(development["cpu_real_time_factor"]), float(final["cpu_real_time_factor"]))
        acceptance=phase3_final_acceptance(baseline_dev, baseline_final, development, final, rtf)
        results.append({
            "checkpoint":value["selected_checkpoint"], "development":development,
            "final_unseen":final, "cpu_real_time_factor":rtf, "acceptance":acceptance,
        })
    passed=all(value["acceptance"]["passed"] for value in results)
    selected=(max(results, key=lambda value:value["development"]["phase3_selection_score"])["checkpoint"]
              if passed else "runs/phase1_suite_seed2026/P1-E2/checkpoints/best_official_composite.pt")
    summary={
        "seed_results":results, "all_three_seeds_passed":passed, "phase3g_passed":passed,
        "selected_checkpoint":selected, "selection_policy":"highest development D; final paths not used for seed selection",
        "final_paths_touched":True,
    }
    Path(args.output).write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
