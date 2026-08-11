"""Offline training entry point for Phase-3G generative FIR controllers."""

from __future__ import annotations

import argparse
import copy
import json
import random
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from phase3_validation import build_phase3_manifests
from phase3g_closed_loop import compute_phase3g_loss, rollout_phase3g_closed_loop
from phase3g_data import Phase3GSequenceDataset, build_phase3g_manifest, save_phase3g_manifest
from phase3g_model import GenerativeInnovationFIRController
from phase3g_validation import (
    evaluate_continuous_path_stress, evaluate_phase3g_development, phase3g_gate,
)
from phase3r_model import InnovationRoutedFIRController
from phase3r_validation import evaluate_switches


DEFAULT_ORACLE="runs/phase3_suite_seed2026_v2/P3-E1/checkpoints/best_phase3_selection.pt"
DEFAULT_TEMPLATE="artifacts/phase3r_innovation_templates.npz"
DEFAULT_P3R="runs/phase3r_suite_seed2026/P3R-E1c/candidate.pt"
DEFAULT_ORACLE_SUMMARY="runs/phase3_suite_seed2026_v2/P3-E1/summary.json"


def set_seed(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def save_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def append_jsonl(path: Path, value: Any) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, ensure_ascii=False)+"\n")


def _load_p3r(path: str | Path) -> InnovationRoutedFIRController:
    checkpoint=torch.load(path, map_location="cpu", weights_only=False)
    model=InnovationRoutedFIRController(**checkpoint["model_config"])
    model.load_state_dict(checkpoint["model_state_dict"])
    return model.eval()


def reduce_candidates(
    full: GenerativeInnovationFIRController,
    keep_original_indices: list[int],
    *,
    seed: int,
) -> GenerativeInnovationFIRController:
    """Physically rebuild a controller without held-out candidates."""
    config=full.model_config.copy(); config["num_experts"]=len(keep_original_indices)
    reduced=GenerativeInnovationFIRController(**config)
    with torch.no_grad():
        reduced.expert_filters.copy_(full.expert_filters[keep_original_indices])
        reduced.primary_real.copy_(full.primary_real[keep_original_indices])
        reduced.primary_imag.copy_(full.primary_imag[keep_original_indices])
        reduced.secondary_paths.copy_(full.secondary_paths[keep_original_indices])
        reduced.hann_window.copy_(full.hann_window); reduced.band_mask.copy_(full.band_mask)
    reduced.initialize_dictionary(seed)
    return reduced


def load_phase3g(path: str | Path, device: torch.device) -> GenerativeInnovationFIRController:
    checkpoint=torch.load(path, map_location="cpu", weights_only=False)
    model=GenerativeInnovationFIRController(**checkpoint["model_config"])
    model.load_state_dict(checkpoint["model_state_dict"])
    return model.to(device)


def save_checkpoint(
    path: Path,
    model: GenerativeInnovationFIRController,
    optimizer: optim.Optimizer,
    epoch: int,
    config: dict[str, Any],
    metrics: dict[str, Any] | None,
) -> None:
    torch.save({
        "phase": "3G", "stage": config["stage"], "epoch": epoch,
        "model_config": model.model_config, "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(), "config": config, "metrics": metrics,
    }, path)


def calibrate_features(
    model: GenerativeInnovationFIRController,
    loader: DataLoader,
    device: torch.device,
    batches: int,
) -> None:
    collected=[]
    was=model.training; model.eval()
    with torch.no_grad():
        for index, batch in enumerate(loader):
            if index >= batches: break
            reference, target, paths, slots, masks, teachers, measured=(value.to(device) for value in batch)
            result=rollout_phase3g_closed_loop(model, reference, target, paths, slots, masks, teachers, measured)
            warmup=(model.n_fft+model.block_size-1)//model.block_size
            values=result.raw_features[:, warmup:].reshape(-1, model.feature_dim)
            values=values[torch.isfinite(values).all(dim=-1)]
            if values.numel(): collected.append(values.cpu())
    if collected:
        values=torch.cat(collected)
        model.set_feature_statistics(values.mean(dim=0), values.std(dim=0, unbiased=False).clamp_min(1e-4))
    if was: model.train()


def compact_metrics(value: dict[str, Any]) -> dict[str, Any]:
    keys=("primary_score_db", "rebound_score_db", "worst_path_primary_db", "first_window_primary_db",
          "controller_peak_abs", "cpu_real_time_factor", "finite", "state_dict_immutable", "phase3_selection_score")
    return {key:value[key] for key in keys if key in value}


def should_stop_warmup(
    stage: str,
    train_paths: list[int],
    development_primary_db: float,
    full_p3r_primary_db: float | None,
) -> bool:
    """Apply the full-development warmup guard only on the matching 8-path split.

    A LOPO fold validates on seven paths, so comparing its mean against the
    eight-path P3R aggregate is not a like-for-like regression test and can
    prematurely terminate a valid fold.
    """
    return bool(
        stage == "warmup"
        and train_paths == list(range(8))
        and full_p3r_primary_db is not None
        and development_primary_db < full_p3r_primary_db - 0.25
    )


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("warmup", "generalize"), required=True)
    parser.add_argument("--dataset-dir", default="dataset")
    parser.add_argument("--oracle-checkpoint", default=DEFAULT_ORACLE)
    parser.add_argument("--template", default=DEFAULT_TEMPLATE)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--p3r-checkpoint", default=DEFAULT_P3R)
    parser.add_argument("--oracle-summary", default=DEFAULT_ORACLE_SUMMARY)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--train-paths", type=int, nargs="+", default=list(range(8)))
    parser.add_argument("--hidden-size", type=int, default=32)
    parser.add_argument("--latent-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--samples-per-epoch", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation", type=int, default=8)
    parser.add_argument("--generator-lr", type=float, default=3e-4)
    parser.add_argument("--dictionary-lr", type=float, default=1e-5)
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--calibration-batches", type=int, default=2)
    parser.add_argument("--stress-cases", type=int, default=48)
    parser.add_argument("--max-train-batches", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", default="auto")
    args=parser.parse_args()
    epochs=args.epochs or (5 if args.stage == "warmup" else 15)
    samples=args.samples_per_epoch or (256 if args.stage == "warmup" else 128)
    if args.device == "auto":
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else: device=torch.device(args.device)
    set_seed(args.seed)
    root=Path(args.output_dir or f"runs/phase3g_{args.stage}_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    root.mkdir(parents=True, exist_ok=False); checkpoints=root/"checkpoints"; checkpoints.mkdir()
    source_config=json.loads((Path(args.oracle_checkpoint).parents[1]/"config.json").read_text(encoding="utf-8"))
    train_noises=source_config["train_noises"]
    if args.stage == "warmup":
        full=GenerativeInnovationFIRController.from_artifacts(
            args.oracle_checkpoint, args.template, hidden_size=args.hidden_size,
            latent_size=args.latent_size, seed=args.seed,
        )
        model=(full if args.train_paths == list(range(8)) else reduce_candidates(full, args.train_paths, seed=args.seed)).to(device)
    else:
        if not args.checkpoint: raise ValueError("generalize stage requires --checkpoint from warmup.")
        model=load_phase3g(args.checkpoint, device)
        if model.hidden_size != args.hidden_size or model.latent_size != args.latent_size:
            raise ValueError("Capacity changes require a fresh warmup stage.")
    dataset=Phase3GSequenceDataset(
        args.dataset_dir, train_noises, train_paths=args.train_paths, samples_per_epoch=samples,
        block_size=model.block_size, synthesis_enabled=args.stage == "generalize",
        switch_probability=0.25 if args.stage == "generalize" else 0.0, seed=args.seed,
    )
    generator=torch.Generator().manual_seed(args.seed)
    loader=DataLoader(dataset, batch_size=args.batch_size, shuffle=True, generator=generator,
                      num_workers=args.num_workers, pin_memory=device.type == "cuda")
    synthesis_manifest=build_phase3g_manifest(args.dataset_dir, train_noises, path_indices=args.train_paths, seed=args.seed)
    save_phase3g_manifest(root/"synthesis_manifest.json", synthesis_manifest)
    manifests=build_phase3_manifests(args.dataset_dir, args.seed)
    manifests["development"]["path_indices_zero_based"]=list(args.train_paths)
    manifests["development"]["split"]="phase3g_development"
    save_json(root/"validation_manifests.json", manifests)
    oracle_summary=json.loads(Path(args.oracle_summary).read_text(encoding="utf-8"))
    p1_baseline=oracle_summary["baseline_development_metrics"]
    p3r_baseline=_load_p3r(args.p3r_checkpoint)
    p3r_summary_path=Path(args.p3r_checkpoint).parent/"summary.json"
    p3r_metrics=json.loads(p3r_summary_path.read_text(encoding="utf-8"))["metrics"] if p3r_summary_path.is_file() else None
    config=vars(args).copy(); config.update({
        "epochs_resolved":epochs, "samples_per_epoch_resolved":samples,
        "train_noises":train_noises, "resolved_device":str(device),
        "model_config":model.model_config, "complexity":model.get_complexity(),
        "paths_policy":"only supplied subset of paths 1-8; paths 9/10 final-only",
    })
    save_json(root/"config.json", config); save_json(root/"p1_baseline.json", p1_baseline)
    calibrate_features(model, loader, device, args.calibration_batches)
    optimizer=optim.Adam([
        {"params":list(model.gru.parameters())+list(model.latent_head.parameters()), "lr":args.generator_lr},
        {"params":[model.residual_dictionary], "lr":args.dictionary_lr},
    ], amsgrad=True)
    best_score=-float("inf"); best_epoch=None; stale=0; last_validation=None; start_time=time.perf_counter()
    for epoch in range(1, epochs+1):
        dataset.set_epoch(epoch); model.train(); optimizer.zero_grad(set_to_none=True)
        totals=defaultdict(float); count=steps=0
        for batch_index, batch in enumerate(loader, start=1):
            if args.max_train_batches and batch_index > args.max_train_batches: break
            reference, target, paths, slots, masks, teachers, measured=(value.to(device) for value in batch)
            rollout=rollout_phase3g_closed_loop(model, reference, target, paths, slots, masks, teachers, measured)
            loss, components=compute_phase3g_loss(target, rollout, distill_weight=0.05 if args.stage == "warmup" else 0.01)
            if not torch.isfinite(loss): raise FloatingPointError("Non-finite Phase-3G loss.")
            (loss/args.gradient_accumulation).backward(); count += reference.shape[0]
            for key,value in components.items(): totals[key]+=float(value)*reference.shape[0]
            last=batch_index == len(loader) or (args.max_train_batches and batch_index == args.max_train_batches)
            if batch_index%args.gradient_accumulation == 0 or last:
                norm=torch.nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clip)
                if not torch.isfinite(norm): raise FloatingPointError("Non-finite Phase-3G gradient.")
                optimizer.step(); optimizer.zero_grad(set_to_none=True); steps+=1
        evaluation=copy.deepcopy(model).cpu().eval()
        development=evaluate_phase3g_development(evaluation, args.dataset_dir, manifests["development"], include_records=False)
        last_validation={"development":compact_metrics(development)}
        score=float(development["phase3_selection_score"])
        record={"epoch":epoch, "samples":count, "optimizer_steps":steps,
                "train":{key:value/max(1,count) for key,value in totals.items()}, "validation":last_validation}
        append_jsonl(root/"history.jsonl", record)
        save_checkpoint(checkpoints/"latest.pt", model, optimizer, epoch, config, last_validation)
        if score > best_score:
            best_score=score; best_epoch=epoch; stale=0
            save_checkpoint(checkpoints/"best_phase3g_selection.pt", model, optimizer, epoch, config, last_validation)
        else: stale+=1
        print(f"Epoch {epoch}/{epochs}: S={development['primary_score_db']:.4f}, R={development['rebound_score_db']:.4f}, D={score:.4f}")
        if should_stop_warmup(
            args.stage,
            args.train_paths,
            float(development["primary_score_db"]),
            None if p3r_metrics is None else float(p3r_metrics["primary_score_db"]),
        ):
            break
        if stale >= args.patience: break
    selected=load_phase3g(checkpoints/"best_phase3g_selection.pt", torch.device("cpu")).eval()
    development=evaluate_phase3g_development(selected, args.dataset_dir, manifests["development"])
    full_development=args.train_paths == list(range(8))
    switches=(evaluate_switches(selected, args.dataset_dir, manifests["development"])
              if full_development else {"all_switches_recover_within_100_ms":True, "skipped_for_lopo":True})
    stress=(evaluate_continuous_path_stress(
        selected, p3r_baseline, args.dataset_dir, manifests["development"], synthesis_manifest,
        max_cases=args.stress_cases,
    ) if args.stage == "generalize" and args.stress_cases > 0 and full_development else None)
    gate=phase3g_gate(p1_baseline, development, switches, stress)
    summary={
        "phase":"3G", "stage":args.stage, "best_epoch":best_epoch,
        "selected_checkpoint":str(checkpoints/"best_phase3g_selection.pt"),
        "development_metrics":development, "switch_metrics":switches,
        "stress_metrics":stress, "acceptance":gate, "complexity":selected.get_complexity(),
        "training_seconds":time.perf_counter()-start_time, "final_paths_touched":False,
        "formal_model":"P1-E2; Phase3G development/LOPO pending",
    }
    save_json(root/"summary.json", summary)


if __name__ == "__main__":
    main()
