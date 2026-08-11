"""Close and freeze the E10-A protocol before any candidate training."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from phase3g_data import build_multispace_neighbor_table, multispace_path_distances
from phase3r_templates import sha256_file


ROOT = Path(__file__).resolve().parent
DEFAULT_SPEC = ROOT / "artifacts/phase4r_preregistered_correction.json"
DEFAULT_TEMPLATE = ROOT / "artifacts/phase3r_innovation_templates.npz"
DEFAULT_DATASET = ROOT / "dataset"
DEFAULT_BASELINE_LOPO = ROOT / "runs/phase3g_suite_seed2026_rerun1/P3G-LOPO-rerun1"


def _canonical_sha256(value: dict[str, Any]) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _rank(values: dict[int, float]) -> dict[int, int]:
    ordered = sorted(values, key=lambda path: (values[path], path))
    return {path: index + 1 for index, path in enumerate(ordered)}


def build_protocol_closure(dataset_dir: Path, template_path: Path) -> dict[str, Any]:
    secondary = np.load(dataset_dir / "sh.npy", allow_pickle=False).T[:8].astype(np.float64)
    with np.load(template_path, allow_pickle=False) as artifact:
        primary_real = artifact["primary_real"].copy()
        primary_imag = artifact["primary_imag"].copy()
        band_mask = artifact["band_mask"].copy()

    fold_tables = {}
    for held in range(8):
        keep = [path for path in range(8) if path != held]
        table, _ = build_multispace_neighbor_table(
            secondary, primary_real, primary_imag, band_mask, keep, neighbor_count=3,
        )
        fold_tables[str(held + 1)] = {
            str(first + 1): [second + 1 for second in table[first]] for first in keep
        }
    global_table, all_distances = build_multispace_neighbor_table(
        secondary, primary_real, primary_imag, band_mask, list(range(8)), neighbor_count=3,
    )

    nearest = {
        "aligned_ir_nrmse": {}, "secondary_response_nrmse": {}, "primary_template_nrmse": {},
    }
    for held in range(8):
        candidates = all_distances[held]
        for metric in nearest:
            nearest[metric][held] = min(value[metric] for value in candidates.values())
    ranks = {metric: _rank(values) for metric, values in nearest.items()}
    coverage = {}
    for held in range(8):
        coverage[held] = {
            "nearest_distances": {metric: nearest[metric][held] for metric in nearest},
            "distance_ranks": {metric: ranks[metric][held] for metric in ranks},
            "mean_distance_rank": float(np.mean([ranks[metric][held] for metric in ranks])),
        }
    worst_three = sorted(coverage, key=lambda path: (-coverage[path]["mean_distance_rank"], path))[:3]
    payload = {
        "algorithm_version": 1,
        "path_numbering": "one_based",
        "distance_definition": {
            "aligned_ir": "real-scale NRMSE after direct-arrival alignment",
            "secondary_response": "real-scale complex NRMSE from 50-8000 Hz at n_fft=4096",
            "primary_template": "real-scale complex NRMSE on the frozen template band",
            "neighbor_score": "mean within-source rank across the three distances; ties by path number",
            "neighbor_selection": "three smallest scores; uniform draw only for interpolate/extrapolate",
        },
        "fold_neighbor_tables": fold_tables,
        "global_neighbor_table": {
            str(first + 1): [second + 1 for second in global_table[first]] for first in range(8)
        },
        "held_out_coverage": {str(path + 1): coverage[path] for path in range(8)},
        "worst_three_coverage_folds": [path + 1 for path in worst_three],
        "inputs": {
            "dataset/sh.npy": sha256_file(dataset_dir / "sh.npy"),
            "artifacts/phase3r_innovation_templates.npz": sha256_file(template_path),
        },
    }
    payload["closure_sha256"] = _canonical_sha256(payload)
    return payload


def audit_baseline_training(lopo_root: Path) -> dict[str, Any]:
    records = []
    for held in range(1, 9):
        config_path = lopo_root / f"path_{held:02d}/generalize/config.json"
        config = json.loads(config_path.read_text(encoding="utf-8"))
        records.append({
            "held_out_path": held,
            "batch_size": int(config["batch_size"]),
            "gradient_accumulation": int(config["gradient_accumulation"]),
            "samples_per_epoch": int(config["samples_per_epoch_resolved"]),
            "hidden_size": int(config["hidden_size"]),
            "latent_size": int(config["latent_size"]),
            "generator_lr": float(config["generator_lr"]),
            "dictionary_lr": float(config["dictionary_lr"]),
        })
    first = {key: value for key, value in records[0].items() if key != "held_out_path"}
    if any({key: value for key, value in record.items() if key != "held_out_path"} != first for record in records[1:]):
        raise ValueError("Frozen baseline LOPO folds do not share one training configuration.")
    return {"all_folds_equal": True, "resolved_training": first, "per_fold": records}


def close_spec(spec_path: Path, closure: dict[str, Any], baseline: dict[str, Any]) -> dict[str, Any]:
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    correction = spec.get("selected_correction", {})
    if correction.get("experiment_id") != "E10-A":
        raise ValueError("Only the diagnosed E10-A correction can be closed by this script.")
    if spec.get("status") not in {"frozen_not_run", "implementation_ready_not_run"}:
        raise ValueError("E10-A can only be closed before training starts.")
    resolved = baseline["resolved_training"]
    spec["frozen_training"].update({
        "batch_size": resolved["batch_size"],
        "gradient_accumulation": resolved["gradient_accumulation"],
        "samples_per_epoch": resolved["samples_per_epoch"],
        "generator_lr": resolved["generator_lr"],
        "dictionary_lr": resolved["dictionary_lr"],
    })
    spec["status"] = "implementation_ready_not_run"
    spec["protocol_closure"] = closure
    spec["baseline_training_audit"] = baseline
    spec["protocol_corrections_before_training"] = [{
        "field": "frozen_training.batch_size/gradient_accumulation",
        "from": "1/8", "to": "8/1",
        "reason": "match all eight frozen baseline LOPO configs and preserve a single-factor E10-A comparison",
    }]
    spec["commands_after_the_selected_change_is_implemented"][0] = (
        "python phase3g_lopo.py --output-root runs/phase4r_e10a_seed2026_lopo --device cuda "
        "--seed 2026 --hidden-size 32 --latent-size 16 --batch-size 8 --gradient-accumulation 1 "
        "--correction-spec artifacts/phase4r_preregistered_correction.json"
    )
    temporary = spec_path.with_suffix(spec_path.suffix + ".tmp")
    temporary.write_text(json.dumps(spec, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(spec_path)
    return spec


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", default=str(DEFAULT_SPEC.relative_to(ROOT)))
    parser.add_argument("--dataset-dir", default=str(DEFAULT_DATASET.relative_to(ROOT)))
    parser.add_argument("--template", default=str(DEFAULT_TEMPLATE.relative_to(ROOT)))
    parser.add_argument("--baseline-lopo", default=str(DEFAULT_BASELINE_LOPO.relative_to(ROOT)))
    args = parser.parse_args()
    spec_path = (ROOT / args.spec).resolve()
    closure = build_protocol_closure((ROOT / args.dataset_dir).resolve(), (ROOT / args.template).resolve())
    baseline = audit_baseline_training((ROOT / args.baseline_lopo).resolve())
    spec = close_spec(spec_path, closure, baseline)
    print(json.dumps({
        "status": spec["status"],
        "closure_sha256": closure["closure_sha256"],
        "worst_three_coverage_folds": closure["worst_three_coverage_folds"],
        "batch_size": spec["frozen_training"]["batch_size"],
        "gradient_accumulation": spec["frozen_training"]["gradient_accumulation"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
