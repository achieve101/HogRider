"""Run the bounded Phase-3R template, routing, switch, and PN decision tree."""

from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any

import torch

from phase3_validation import build_phase3_manifests
from phase3r_model import InnovationRoutedFIRController
from phase3r_templates import build_innovation_templates, validate_template_artifact
from phase3r_validation import development_gate, evaluate_development, evaluate_switches


CANDIDATES = {
    "P3R-E1a": {"ewma_lambda": 0.5, "temperature": 0.15, "alpha_update": 0.35},
    "P3R-E1b": {"ewma_lambda": 0.0, "temperature": 0.10, "alpha_update": 0.50},
    "P3R-E1c": {"ewma_lambda": 0.8, "temperature": 0.20, "alpha_update": 0.20},
}


def _save_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def _save_candidate(path: Path, model: InnovationRoutedFIRController, source: dict[str, str]) -> None:
    torch.save({
        "phase": "3R", "model_config": model.model_config,
        "model_state_dict": model.state_dict(), "source": source,
    }, path)


def _compact(metrics: dict[str, Any]) -> dict[str, Any]:
    return {key: metrics[key] for key in (
        "primary_score_db", "rebound_score_db", "selection_score", "phase3_selection_score",
        "worst_path_primary_db", "first_window_primary_db", "controller_peak_abs",
        "route_accuracy_after_initialization", "cpu_real_time_factor", "finite",
    )}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", default="dataset")
    parser.add_argument("--oracle-checkpoint", default="runs/phase3_suite_seed2026_v2/P3-E1/checkpoints/best_phase3_selection.pt")
    parser.add_argument("--oracle-summary", default="runs/phase3_suite_seed2026_v2/P3-E1/summary.json")
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--template", default=None)
    args = parser.parse_args()

    root = Path(args.output_root or f"runs/phase3r_suite_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    root.mkdir(parents=True, exist_ok=False)
    template = Path(args.template) if args.template else root / "phase3r_innovation_templates.npz"
    if not template.is_file():
        config_path = Path(args.oracle_summary).parent / "config.json"
        noises = tuple(json.loads(config_path.read_text(encoding="utf-8"))["train_noises"])
        build_innovation_templates(args.dataset_dir, template, train_noises=noises)
    else:
        shutil.copy2(template, root / template.name)
        manifest_source = template.with_suffix(".manifest.json")
        if manifest_source.is_file():
            shutil.copy2(manifest_source, root / manifest_source.name)
        template = root / template.name

    validate_template_artifact(template)

    manifests = build_phase3_manifests(args.dataset_dir)["development"]
    _save_json(root / "innovation_manifest.json", {
        "development": manifests,
        "switch_pairs_zero_based": [[0, 6], [6, 0], [3, 5], [5, 7]],
        "switch_sample": 96_000,
    })
    oracle_summary = json.loads(Path(args.oracle_summary).read_text(encoding="utf-8"))
    baseline = oracle_summary["baseline_development_metrics"]
    source = {"oracle_checkpoint": str(Path(args.oracle_checkpoint)), "template": str(template)}
    results: dict[str, Any] = {}

    for name, config in CANDIDATES.items():
        candidate_dir = root / name
        candidate_dir.mkdir()
        model = InnovationRoutedFIRController.from_artifacts(args.oracle_checkpoint, template, **config).eval()
        metrics = evaluate_development(model, args.dataset_dir, manifests)
        switches = evaluate_switches(model, args.dataset_dir, manifests)
        gate = development_gate(baseline, metrics, switches)
        checkpoint = candidate_dir / "candidate.pt"
        _save_candidate(checkpoint, model, source)
        _save_json(candidate_dir / "development_metrics.json", metrics)
        _save_json(candidate_dir / "switch_metrics.json", switches)
        _save_json(candidate_dir / "summary.json", {
            "experiment": name, "route_config": config, "metrics": _compact(metrics),
            "switch_maximum_three_correct_recovery_ms": switches["maximum_three_correct_recovery_ms"],
            "acceptance": gate, "checkpoint": str(checkpoint), "complexity": model.get_complexity(),
            "final_paths_touched": False,
        })
        results[name] = {
            "route_config": config, "metrics": _compact(metrics),
            "switch_maximum_three_correct_recovery_ms": switches["maximum_three_correct_recovery_ms"],
            "acceptance": gate, "checkpoint": str(checkpoint), "complexity": model.get_complexity(),
        }

    passing = {name: value for name, value in results.items() if value["acceptance"]["passed"]}
    selected = max(passing, key=lambda name: passing[name]["metrics"]["phase3_selection_score"]) if passing else None
    route_keys = ("route_accuracy_at_least_95_percent", "switch_recovery_at_most_100_ms")
    all_failed_for_routing = all(
        not value["acceptance"]["passed"]
        and not all(value["acceptance"]["checks"][key] for key in route_keys)
        for value in results.values()
    )
    pn_ran = False
    if selected is None and all_failed_for_routing:
        pn_ran = True
        name = "P3R-E2"
        candidate_dir = root / name
        candidate_dir.mkdir()
        config = {**CANDIDATES["P3R-E1a"], "probe_rms": 0.01, "probe_samples": 20_000, "probe_seed": 2026}
        model = InnovationRoutedFIRController.from_artifacts(args.oracle_checkpoint, template, **config).eval()
        metrics = evaluate_development(model, args.dataset_dir, manifests)
        switches = evaluate_switches(model, args.dataset_dir, manifests)
        gate = development_gate(baseline, metrics, switches)
        checkpoint = candidate_dir / "candidate.pt"
        _save_candidate(checkpoint, model, source)
        _save_json(candidate_dir / "development_metrics.json", metrics)
        _save_json(candidate_dir / "switch_metrics.json", switches)
        results[name] = {
            "route_config": config, "metrics": _compact(metrics),
            "switch_maximum_three_correct_recovery_ms": switches["maximum_three_correct_recovery_ms"],
            "acceptance": gate, "checkpoint": str(checkpoint), "complexity": model.get_complexity(),
        }
        _save_json(candidate_dir / "summary.json", results[name])
        if gate["passed"]:
            selected = name

    suite = {
        "phase": "3R", "baseline": {key: baseline[key] for key in (
            "primary_score_db", "rebound_score_db", "worst_path_primary_db", "first_window_primary_db"
        )},
        "results": results, "selected_development_experiment": selected,
        "selected_checkpoint": None if selected is None else results[selected]["checkpoint"],
        "pn_fallback_ran": pn_ran, "ready_for_lopo": selected is not None,
        "formal_model": "P1-E2" if selected is None else "Phase3R development candidate; LOPO pending",
        "final_paths_touched": False,
    }
    _save_json(root / "suite_summary.json", suite)
    print(json.dumps(suite, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
