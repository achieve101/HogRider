"""Strict eight-fold Phase-3G leave-one-path-out training and evaluation."""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from scipy.signal import fftconvolve

from phase1_data import iter_validation_examples
from phase1_validation import _aggregate, _load_official_scorer
from phase3_validation import _score_record, build_phase3_manifests
from phase3g_validation import evaluate_phase3g_development
from train_phase3g import load_phase3g


def _run_training(output: Path, args: argparse.Namespace, stage: str, paths: list[int], checkpoint: str | None) -> dict:
    command=[
        sys.executable, "-u", "train_phase3g.py", "--stage", stage,
        "--dataset-dir", args.dataset_dir, "--output-dir", str(output),
        "--oracle-checkpoint", args.oracle_checkpoint, "--template", args.template,
        "--p3r-checkpoint", args.p3r_checkpoint, "--oracle-summary", args.oracle_summary,
        "--device", args.device, "--seed", str(args.seed), "--stress-cases", "0",
        "--hidden-size", str(args.hidden_size), "--latent-size", str(args.latent_size),
        "--batch-size", str(args.batch_size), "--gradient-accumulation", str(args.gradient_accumulation),
        "--train-paths", *[str(value) for value in paths],
    ]
    command.extend(["--epochs", str(args.warmup_epochs if stage == "warmup" else args.generalize_epochs)])
    command.extend(["--samples-per-epoch", str(args.samples_per_epoch)])
    if checkpoint: command.extend(["--checkpoint", checkpoint])
    if args.max_train_batches: command.extend(["--max-train-batches", str(args.max_train_batches)])
    subprocess.run(command, check=True)
    return json.loads((output/"summary.json").read_text(encoding="utf-8"))


def _static_filter_metrics(filter_value: np.ndarray, held_out: int, dataset_dir: str) -> dict:
    manifest=build_phase3_manifests(dataset_dir)["development"]
    manifest["path_indices_zero_based"]=[held_out]
    scorer=_load_official_scorer(); records=[]
    for scene,path_index,reference,path,disturbance in iter_validation_examples(dataset_dir, manifest):
        x=reference.numpy().astype(np.float64); s=path.numpy().astype(np.float64); d=disturbance.numpy().astype(np.float64)
        raw=fftconvolve(x, filter_value, mode="full")[:x.size]
        limit=0.98-1e-6; output=limit*np.tanh(raw/limit)
        residual=d-fftconvolve(output, s, mode="full")[:x.size]
        records.append(_score_record(
            scorer, scene, path_index, torch.from_numpy(d)[None],
            torch.from_numpy(residual)[None], torch.from_numpy(output)[None],
        ))
    return _aggregate(records)


def _oracle_latent_filter(model, target: np.ndarray) -> np.ndarray:
    experts=model.expert_filters.detach().cpu().numpy().astype(np.float64)
    dictionary=model.residual_dictionary.detach().cpu().numpy().astype(np.float64)
    bank=np.concatenate((experts, dictionary), axis=0)
    design=bank.T
    scale=np.linalg.norm(design)/np.sqrt(design.size)
    constraint=np.r_[np.ones(experts.shape[0]), np.zeros(dictionary.shape[0])][None]*scale*10
    coefficients=np.linalg.lstsq(
        np.vstack((design, constraint)), np.r_[target, scale*10], rcond=1e-6,
    )[0]
    return coefficients@bank


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--dataset-dir", default="dataset")
    parser.add_argument("--oracle-checkpoint", default="runs/phase3_suite_seed2026_v2/P3-E1/checkpoints/best_phase3_selection.pt")
    parser.add_argument("--template", default="artifacts/phase3r_innovation_templates.npz")
    parser.add_argument("--p3r-checkpoint", default="runs/phase3r_suite_seed2026/P3R-E1c/candidate.pt")
    parser.add_argument("--oracle-summary", default="runs/phase3_suite_seed2026_v2/P3-E1/summary.json")
    parser.add_argument("--warmup-epochs", type=int, default=5)
    parser.add_argument("--generalize-epochs", type=int, default=15)
    parser.add_argument("--samples-per-epoch", type=int, default=128)
    parser.add_argument("--hidden-size", type=int, default=32)
    parser.add_argument("--latent-size", type=int, default=16)
    parser.add_argument("--max-train-batches", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation", type=int, default=8)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", default="auto")
    args=parser.parse_args()
    root=Path(args.output_root or f"runs/phase3g_lopo_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    root.mkdir(parents=True, exist_ok=False)
    p1=json.loads(Path("runs/phase1_suite_seed2026/P1-E2/summary.json").read_text(encoding="utf-8"))
    p1_paths=p1["final_metrics"]["path_metrics"]
    oracle_checkpoint=torch.load(args.oracle_checkpoint, map_location="cpu", weights_only=False)
    oracle_experts=oracle_checkpoint["model_state_dict"]["expert_filters"].numpy().astype(np.float64)
    folds=[]; gains=[]; oracle_gains=[]
    for held_out in range(8):
        keep=[value for value in range(8) if value != held_out]
        fold=root/f"path_{held_out+1:02d}"; fold.mkdir()
        warm=_run_training(fold/"warmup", args, "warmup", keep, None)
        general=_run_training(fold/"generalize", args, "generalize", keep, warm["selected_checkpoint"])
        model=load_phase3g(general["selected_checkpoint"], torch.device("cpu")).eval()
        manifest=build_phase3_manifests(args.dataset_dir)["development"]
        manifest["path_indices_zero_based"]=[held_out]
        metrics=evaluate_phase3g_development(model, args.dataset_dir, manifest)
        baseline=float(p1_paths[str(held_out+1)]["primary_score_db"])
        gain=float(metrics["primary_score_db"]-baseline); gains.append(gain)
        oracle_filter=_oracle_latent_filter(model, oracle_experts[held_out])
        oracle_metrics=_static_filter_metrics(oracle_filter, held_out, args.dataset_dir)
        oracle_gain=float(oracle_metrics["primary_score_db"]-baseline); oracle_gains.append(oracle_gain)
        synthesis=json.loads((fold/"generalize"/"synthesis_manifest.json").read_text(encoding="utf-8"))
        isolation=(
            held_out not in synthesis["path_indices_zero_based"]
            and all(f"scene_{held_out+1:02d}" not in name for name in synthesis["input_sha256"])
        )
        folds.append({
            "held_out_path":held_out+1, "retained_original_indices_zero_based":keep,
            "baseline_primary_db":baseline, "candidate_primary_db":metrics["primary_score_db"],
            "primary_gain_db":gain, "candidate_rebound_db":metrics["rebound_score_db"],
            "oracle_latent_primary_db":oracle_metrics["primary_score_db"],
            "oracle_latent_gain_db":oracle_gain, "checkpoint":general["selected_checkpoint"],
            "isolation_passed":isolation,
        })
    checks={
        "median_gain_at_least_0_5_db":statistics.median(gains) >= 0.5,
        "at_least_6_of_8_non_degrading":sum(value >= 0 for value in gains) >= 6,
        "all_folds_physically_isolated":all(value["isolation_passed"] for value in folds),
    }
    oracle_pass=(statistics.median(oracle_gains) >= 0.5 and sum(value >= 0 for value in oracle_gains) >= 6)
    correction=None if all(checks.values()) else (
        {"latent_size":16, "hidden_size":48, "reason":"oracle latent passed; generator mapping failed"}
        if oracle_pass else {"latent_size":24, "hidden_size":24, "reason":"oracle latent capacity failed"}
    )
    summary={
        "folds":folds, "median_primary_gain_db":statistics.median(gains),
        "non_degrading_fold_count":sum(value >= 0 for value in gains),
        "oracle_latent_median_gain_db":statistics.median(oracle_gains),
        "oracle_latent_non_degrading_fold_count":sum(value >= 0 for value in oracle_gains),
        "oracle_latent_passed":oracle_pass, "recommended_single_correction":correction,
        "acceptance":{"passed":all(checks.values()), "checks":checks},
        "final_paths_touched":False,
    }
    (root/"lopo_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
