"""Train the score-aligned DEEPANC Phase-1 baseline."""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Dict

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from dataset import PreconvolutedANCDataset, apply_dynamic_path
from legacy_models.phase0_phase1.model import TimeDomainANC
from phase1_data import build_validation_manifest
from phase1_validation import evaluate_v6_model
from legacy_models.phase0_phase1.train import resolve_device, scan_noise_files, seed_worker, set_global_seed
from v6_metrics import compute_v6_loss


DEFAULT_CHECKPOINT = "runs/phase0_60ep_seed2026/checkpoints/best_mean_nr.pt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fine-tune TimeDomainANC against the official v6 acoustic score."
    )
    parser.add_argument("--dataset-dir", default="dataset")
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument(
        "--experiment", choices=("band", "composite", "custom"),
        default="composite",
    )
    parser.add_argument("--evaluate-only", action="store_true")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--early-stop-patience", type=int, default=5)
    parser.add_argument("--early-stop-min-delta", type=float, default=0.05)
    parser.add_argument("--samples-per-epoch", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--gradient-accumulation", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument("--validation-interval", type=int, default=1)
    parser.add_argument("--primary-weight", type=float, default=None)
    parser.add_argument("--rebound-weight", type=float, default=None)
    parser.add_argument("--time-weight", type=float, default=0.1)
    parser.add_argument("--guard-weight", type=float, default=1.0)
    parser.add_argument("--guard-limit", type=float, default=1.0)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--max-train-batches", type=int, default=None,
        help="Optional smoke-test limit; omit for real experiments.",
    )
    return parser.parse_args()


def _save_json(path: Path, value: object) -> None:
    with path.open("w", encoding="utf-8") as file:
        json.dump(value, file, ensure_ascii=False, indent=2)


def _append_jsonl(path: Path, value: object) -> None:
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(value, ensure_ascii=False) + "\n")


def _resolve_weights(args: argparse.Namespace) -> Dict[str, float]:
    defaults = {
        "band": (1.0, 0.0),
        "composite": (0.7, 0.3),
    }
    if args.experiment == "custom":
        if args.primary_weight is None or args.rebound_weight is None:
            raise ValueError(
                "custom experiment requires --primary-weight and --rebound-weight."
            )
        primary, rebound = args.primary_weight, args.rebound_weight
    else:
        primary, rebound = defaults[args.experiment]
        if args.primary_weight is not None:
            primary = args.primary_weight
        if args.rebound_weight is not None:
            rebound = args.rebound_weight
    values = {
        "primary": float(primary),
        "rebound": float(rebound),
        "time": float(args.time_weight),
        "guard": float(args.guard_weight),
        "guard_limit": float(args.guard_limit),
    }
    if any(not math.isfinite(value) or value < 0 for value in values.values()):
        raise ValueError("All loss weights and guard_limit must be finite and nonnegative.")
    if values["guard_limit"] <= 0:
        raise ValueError("guard_limit must be positive.")
    return values


def _load_model(checkpoint_path: Path, device: torch.device) -> TimeDomainANC:
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    checkpoint = torch.load(
        checkpoint_path, map_location=device, weights_only=False,
    )
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model = TimeDomainANC(
        in_channels=1, out_channels=1, hidden_channels=32, num_layers=10,
    ).to(device)
    model.load_state_dict(state_dict)
    return model


def _save_checkpoint(
    path: Path,
    model: TimeDomainANC,
    optimizer: optim.Optimizer,
    epoch: int,
    config: Dict[str, object],
    manifest: Dict[str, object],
    metrics: Dict[str, object] | None,
) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save({
        "phase": 1,
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "config": config,
        "validation_manifest": manifest,
        "metrics": metrics,
    }, temporary)
    os.replace(temporary, path)


def _acceptance(
    baseline: Dict[str, object], candidate: Dict[str, object],
) -> Dict[str, object]:
    baseline_rebound = float(baseline["rebound_score_db"])
    candidate_rebound = float(candidate["rebound_score_db"])
    rebound_reduction = (
        (baseline_rebound - candidate_rebound) / baseline_rebound
        if baseline_rebound > 0 else 0.0
    )
    checks = {
        "selection_score_gain_at_least_0_5": (
            float(candidate["selection_score"])
            - float(baseline["selection_score"]) >= 0.5
        ),
        "primary_drop_at_most_0_25_db": (
            float(candidate["primary_score_db"])
            >= float(baseline["primary_score_db"]) - 0.25
        ),
        "rebound_reduction_at_least_20_percent": rebound_reduction >= 0.20,
        "controller_peak_at_most_1": (
            float(candidate["controller_peak_abs"]) <= 1.0
        ),
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "selection_score_gain": (
            float(candidate["selection_score"])
            - float(baseline["selection_score"])
        ),
        "primary_score_change_db": (
            float(candidate["primary_score_db"])
            - float(baseline["primary_score_db"])
        ),
        "rebound_reduction_fraction": rebound_reduction,
    }


def main() -> None:
    args = parse_args()
    positive_values = {
        "epochs": args.epochs,
        "early-stop-patience": args.early_stop_patience,
        "samples-per-epoch": args.samples_per_epoch,
        "batch-size": args.batch_size,
        "gradient-accumulation": args.gradient_accumulation,
        "validation-interval": args.validation_interval,
    }
    if any(value <= 0 for value in positive_values.values()):
        raise ValueError(f"These arguments must be positive: {positive_values}")
    if args.learning_rate <= 0 or args.gradient_clip <= 0:
        raise ValueError("learning-rate and gradient-clip must be positive.")
    if args.max_train_batches is not None and args.max_train_batches <= 0:
        raise ValueError("max-train-batches must be positive when provided.")

    weights = _resolve_weights(args)
    set_global_seed(args.seed)
    device = resolve_device(args.device)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = "e0" if args.evaluate_only else args.experiment
    output_dir = Path(args.output_dir or f"runs/phase1_{run_name}_{timestamp}")
    checkpoint_dir = output_dir / "checkpoints"
    output_dir.mkdir(parents=True, exist_ok=False)
    if not args.evaluate_only:
        checkpoint_dir.mkdir()

    dataset_dir = Path(args.dataset_dir)
    checkpoint_path = Path(args.checkpoint)
    manifest = build_validation_manifest(dataset_dir)
    config = vars(args).copy()
    config.update({
        "resolved_device": str(device),
        "output_dir": str(output_dir),
        "loss_weights": weights,
        "model": {
            "name": "TimeDomainANC",
            "hidden_channels": 32,
            "num_layers": 10,
            "parameter_count": 42_764,
        },
        "protocol": {
            "version": 6,
            "initialization_seconds": 0.5,
            "scoring_seconds": 3.0,
            "scoring_windows": 6,
        },
    })
    _save_json(output_dir / "config.json", config)
    _save_json(output_dir / "validation_manifest.json", manifest)

    model = _load_model(checkpoint_path, device)
    print(f"Using device: {device}")
    print(f"Loaded Phase-0 initialization: {checkpoint_path}")
    print("Evaluating immutable Phase-1 baseline manifest...")
    baseline = evaluate_v6_model(model, dataset_dir, device, manifest)
    _save_json(output_dir / "baseline_metrics.json", baseline)
    print(
        f"Baseline S={baseline['primary_score_db']:.4f} dB, "
        f"R={baseline['rebound_score_db']:.4f} dB, "
        f"C={baseline['selection_score']:.4f}"
    )
    if args.evaluate_only:
        _save_json(output_dir / "summary.json", {
            "mode": "P1-E0",
            "checkpoint": str(checkpoint_path),
            "metrics": baseline,
        })
        return

    all_noise_files = scan_noise_files(dataset_dir / "NOISE")
    if len(all_noise_files) < 8:
        raise ValueError("Phase 1 expects all eight official noise files.")
    train_noises = all_noise_files[:-2]
    train_paths = list(range(8))
    train_dataset = PreconvolutedANCDataset(
        dataset_dir,
        train_noises,
        train_paths,
        segment_duration=3.5,
        sr=48_000,
        is_train=True,
        samples_per_epoch=args.samples_per_epoch,
    )
    generator = torch.Generator().manual_seed(args.seed)
    loader_args = {
        "num_workers": args.num_workers,
        "pin_memory": device.type == "cuda",
        "worker_init_fn": seed_worker,
    }
    if args.num_workers > 0:
        loader_args["persistent_workers"] = True
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        generator=generator,
        **loader_args,
    )
    optimizer = optim.Adam(
        model.parameters(), lr=args.learning_rate, amsgrad=True,
    )

    history_path = output_dir / "history.jsonl"
    best_composite = -float("inf")
    best_primary = -float("inf")
    best_rebound = float("inf")
    best_epoch = None
    early_stop_reference = -float("inf")
    stale_validations = 0
    last_metrics = None
    training_start = time.perf_counter()

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.perf_counter()
        model.train()
        optimizer.zero_grad(set_to_none=True)
        sums = defaultdict_float()
        sample_count = 0
        optimizer_steps = 0

        for batch_index, (reference, secondary_path, disturbance) in enumerate(
            train_loader, start=1,
        ):
            if args.max_train_batches is not None and batch_index > args.max_train_batches:
                break
            reference = reference.to(device, non_blocking=True)
            secondary_path = secondary_path.to(device, non_blocking=True)
            disturbance = disturbance.to(device, non_blocking=True)
            controller = model(reference)
            residual = disturbance - apply_dynamic_path(controller, secondary_path)
            loss, components = compute_v6_loss(
                disturbance,
                residual,
                controller,
                primary_weight=weights["primary"],
                rebound_weight=weights["rebound"],
                time_weight=weights["time"],
                guard_weight=weights["guard"],
                guard_limit=weights["guard_limit"],
            )
            if not torch.isfinite(loss):
                raise FloatingPointError(
                    f"Non-finite loss at epoch {epoch}, batch {batch_index}."
                )
            (loss / args.gradient_accumulation).backward()

            batch_size = reference.shape[0]
            sample_count += batch_size
            for name, value in components.items():
                sums[name] += float(value.cpu()) * batch_size

            should_step = batch_index % args.gradient_accumulation == 0
            is_last_available = batch_index == len(train_loader)
            hit_smoke_limit = (
                args.max_train_batches is not None
                and batch_index == args.max_train_batches
            )
            if should_step or is_last_available or hit_smoke_limit:
                gradient_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), args.gradient_clip,
                )
                if not torch.isfinite(gradient_norm):
                    raise FloatingPointError(
                        f"Non-finite gradient at epoch {epoch}, batch {batch_index}."
                    )
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                optimizer_steps += 1

        train_metrics = {
            name: value / max(1, sample_count) for name, value in sums.items()
        }
        epoch_record = {
            "epoch": epoch,
            "samples": sample_count,
            "optimizer_steps": optimizer_steps,
            "duration_seconds": time.perf_counter() - epoch_start,
            "train": train_metrics,
        }
        print(
            f"Epoch {epoch}/{args.epochs}: loss={train_metrics['total_loss']:.6f}, "
            f"S~={train_metrics['primary_score_db']:.3f}, "
            f"R~={train_metrics['rebound_score_db']:.3f}"
        )

        should_validate = (
            epoch == 1 or epoch == args.epochs
            or epoch % args.validation_interval == 0
        )
        if should_validate:
            last_metrics = evaluate_v6_model(
                model, dataset_dir, device, manifest, include_records=False,
            )
            epoch_record["validation"] = last_metrics
            score = float(last_metrics["selection_score"])
            primary = float(last_metrics["primary_score_db"])
            rebound = float(last_metrics["rebound_score_db"])
            print(f"  Official v6: S={primary:.4f}, R={rebound:.4f}, C={score:.4f}")

            if score > best_composite:
                best_composite = score
                best_epoch = epoch
                _save_checkpoint(
                    checkpoint_dir / "best_official_composite.pt",
                    model, optimizer, epoch, config, manifest, last_metrics,
                )
            if primary > best_primary:
                best_primary = primary
                _save_checkpoint(
                    checkpoint_dir / "best_primary.pt",
                    model, optimizer, epoch, config, manifest, last_metrics,
                )
            if rebound < best_rebound:
                best_rebound = rebound
                _save_checkpoint(
                    checkpoint_dir / "best_rebound.pt",
                    model, optimizer, epoch, config, manifest, last_metrics,
                )

            if score >= early_stop_reference + args.early_stop_min_delta:
                early_stop_reference = score
                stale_validations = 0
            else:
                stale_validations += 1

        _save_checkpoint(
            checkpoint_dir / "latest.pt",
            model, optimizer, epoch, config, manifest, last_metrics,
        )
        _append_jsonl(history_path, epoch_record)
        if should_validate and stale_validations >= args.early_stop_patience:
            print(f"Early stopping after {stale_validations} stale validations.")
            break

    best_path = checkpoint_dir / "best_official_composite.pt"
    best_checkpoint = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(best_checkpoint["model_state_dict"])
    final_metrics = evaluate_v6_model(model, dataset_dir, device, manifest)
    acceptance = _acceptance(baseline, final_metrics)
    summary = {
        "mode": f"P1-{args.experiment}",
        "initial_checkpoint": str(checkpoint_path),
        "training_seconds": time.perf_counter() - training_start,
        "best_epoch": best_epoch,
        "best_checkpoint": str(best_path),
        "loss_weights": weights,
        "baseline_metrics": baseline,
        "final_metrics": final_metrics,
        "acceptance": acceptance,
    }
    _save_json(output_dir / "summary.json", summary)
    print(
        f"Final S={final_metrics['primary_score_db']:.4f}, "
        f"R={final_metrics['rebound_score_db']:.4f}, "
        f"C={final_metrics['selection_score']:.4f}, "
        f"accepted={acceptance['passed']}"
    )


def defaultdict_float() -> Dict[str, float]:
    return {
        "total_loss": 0.0,
        "primary_loss": 0.0,
        "rebound_loss": 0.0,
        "time_loss": 0.0,
        "guard_loss": 0.0,
        "primary_score_db": 0.0,
        "rebound_score_db": 0.0,
        "controller_peak_abs": 0.0,
    }


if __name__ == "__main__":
    main()
