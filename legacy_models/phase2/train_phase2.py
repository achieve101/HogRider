"""Train the Phase-2 path-robust DEEPANC controller without changing its network."""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from legacy_models.phase0_phase1.model import TimeDomainANC
from phase2_paths import Phase2GroupedDataset, compute_phase2_group_loss
from legacy_models.phase2.phase2_validation import (
    build_phase2_manifests,
    development_gate,
    evaluate_phase2_development,
    evaluate_phase2_stress,
    phase2_acceptance,
)
from legacy_models.phase0_phase1.train import resolve_device, scan_noise_files, seed_worker, set_global_seed


DEFAULT_CHECKPOINT = "runs/phase1_suite_seed2026/P1-E2/checkpoints/best_official_composite.pt"


def parse_index_list(value: str) -> list[int]:
    result = [int(item.strip()) - 1 for item in value.split(",") if item.strip()]
    if not result or len(set(result)) != len(result) or any(index < 0 or index >= 10 for index in result):
        raise argparse.ArgumentTypeError("Use unique one-based path numbers in 1..10.")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", default="dataset")
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--experiment", choices=("control", "real", "augment"), default="augment")
    parser.add_argument("--train-paths", type=parse_index_list, default=parse_index_list("1,2,3,4,5,6,7,8"))
    parser.add_argument("--dev-paths", type=parse_index_list, default=parse_index_list("1,2,3,4,5,6,7,8"))
    parser.add_argument("--evaluate-final", action="store_true", help="Evaluate held-out paths 9/10 once after training.")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--early-stop-patience", type=int, default=5)
    parser.add_argument("--early-stop-min-delta", type=float, default=0.05)
    parser.add_argument("--samples-per-epoch", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--gradient-accumulation", type=int, default=4)
    parser.add_argument("--group-size", type=int, default=None)
    parser.add_argument("--beta", type=float, default=0.25)
    parser.add_argument("--augmentation-probability", type=float, default=0.8)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument("--validation-interval", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-train-batches", type=int, default=None)
    return parser.parse_args()


def save_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def append_jsonl(path: Path, value: Any) -> None:
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(value, ensure_ascii=False) + "\n")


def load_model(path: Path, device: torch.device) -> TimeDomainANC:
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model = TimeDomainANC(in_channels=1, out_channels=1, hidden_channels=32, num_layers=10).to(device)
    model.load_state_dict(checkpoint.get("model_state_dict", checkpoint))
    return model


def save_checkpoint(
    path: Path, model: TimeDomainANC, optimizer: optim.Optimizer, epoch: int,
    config: dict[str, Any], manifests: dict[str, Any], metrics: dict[str, Any] | None,
) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save({
        "phase": 2, "epoch": epoch, "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(), "config": config,
        "validation_manifests": manifests, "metrics": metrics,
    }, temporary)
    os.replace(temporary, path)


def _metric_sums() -> dict[str, float]:
    return {name: 0.0 for name in (
        "total_loss", "mean_path_loss", "worst_quartile_loss", "primary_loss",
        "rebound_loss", "time_loss", "guard_loss", "primary_score_db",
        "rebound_score_db", "controller_peak_abs", "top_quartile_count",
    )}


def main() -> None:
    args = parse_args()
    group_size = args.group_size if args.group_size is not None else (1 if args.experiment == "control" else 4)
    use_augmentation = args.experiment == "augment"
    resolved_beta = 0.0 if args.experiment == "control" else args.beta
    positive = (args.epochs, args.early_stop_patience, args.samples_per_epoch, args.batch_size,
                args.gradient_accumulation, group_size, args.validation_interval)
    if any(value <= 0 for value in positive):
        raise ValueError("Epoch, batch, group, accumulation, and validation values must be positive.")
    if args.beta < 0 or not 0 <= args.augmentation_probability <= 1:
        raise ValueError("beta must be nonnegative and augmentation probability in [0,1].")
    if any(index >= 8 for index in args.train_paths):
        raise ValueError("Paths 9/10 are final-only and cannot be used for Phase-2 training.")
    if args.max_train_batches is not None and args.max_train_batches <= 0:
        raise ValueError("max-train-batches must be positive.")
    if args.learning_rate <= 0 or args.gradient_clip <= 0:
        raise ValueError("learning rate and gradient clip must be positive.")

    set_global_seed(args.seed)
    device = resolve_device(args.device)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir or f"runs/phase2_{args.experiment}_{timestamp}")
    output_dir.mkdir(parents=True, exist_ok=False)
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir()
    dataset_dir = Path(args.dataset_dir)
    checkpoint_path = Path(args.checkpoint)

    manifests = build_phase2_manifests(dataset_dir, args.seed)
    manifests["development"]["path_indices_zero_based"] = list(args.dev_paths)
    manifests["development"]["split"] = "development_custom"
    manifests["stress"]["source_manifest"] = manifests["development"]
    manifests["stress"]["variants"] = [
        item for item in manifests["stress"]["variants"]
        if item["base_path_index_zero_based"] in args.dev_paths
    ]
    config = vars(args).copy()
    config.update({
        "group_size_resolved": group_size, "use_augmentation": use_augmentation,
        "effective_batch_size": args.batch_size * args.gradient_accumulation,
        "beta_resolved": resolved_beta,
        "loss": {"primary": 0.7, "rebound": 0.3, "time": 0.1, "guard": 1.0, "beta": resolved_beta},
        "model": {"name": "TimeDomainANC", "parameter_count": 42_764},
        "split_policy": "paths 9/10 are excluded from gradients and per-epoch model selection",
        "resolved_device": str(device),
    })
    save_json(output_dir / "config.json", config)
    save_json(output_dir / "validation_manifests.json", manifests)

    model = load_model(checkpoint_path, device)
    baseline_dev = evaluate_phase2_development(
        model, dataset_dir, device, manifests["development"], include_records=False,
    )
    baseline_stress = evaluate_phase2_stress(
        model, dataset_dir, device, manifests["stress"], include_records=False,
    )
    save_json(output_dir / "baseline_development_metrics.json", baseline_dev)
    save_json(output_dir / "baseline_stress_metrics.json", baseline_stress)
    print(f"Device={device}; baseline dev S={baseline_dev['primary_score_db']:.4f}, "
          f"R={baseline_dev['rebound_score_db']:.4f}, robust={baseline_dev['robust_development_score']:.4f}")

    noises = scan_noise_files(dataset_dir / "NOISE")
    if len(noises) < 8:
        raise ValueError("Phase 2 expects all eight official noise files.")
    train_noises = noises[:-2]
    config["train_noises"] = train_noises
    config["fixed_validation_noises"] = noises[-2:]
    save_json(output_dir / "config.json", config)
    train_dataset = Phase2GroupedDataset(
        dataset_dir, train_noises, args.train_paths, group_size=group_size,
        use_augmentation=use_augmentation,
        augmentation_probability=args.augmentation_probability,
        samples_per_epoch=args.samples_per_epoch, seed=args.seed,
    )
    generator = torch.Generator().manual_seed(args.seed)
    loader_options: dict[str, Any] = {
        "num_workers": args.num_workers, "pin_memory": device.type == "cuda",
        "worker_init_fn": seed_worker,
    }
    if args.num_workers > 0:
        loader_options["persistent_workers"] = False
    loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                        generator=generator, **loader_options)
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, amsgrad=True)
    history_path = output_dir / "history.jsonl"
    best_robust = -float("inf")
    best_primary = -float("inf")
    best_rebound = float("inf")
    best_epoch = None
    stale = 0
    early_reference = -float("inf")
    last_validation = None
    start_time = time.perf_counter()

    for epoch in range(1, args.epochs + 1):
        train_dataset.set_epoch(epoch)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        sums = _metric_sums()
        sample_count = optimizer_steps = 0
        epoch_start = time.perf_counter()
        for batch_index, (reference, paths, target, _, _) in enumerate(loader, start=1):
            if args.max_train_batches is not None and batch_index > args.max_train_batches:
                break
            reference = reference.to(device, non_blocking=True)
            paths = paths.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            controller = model(reference)
            loss, components = compute_phase2_group_loss(
                target, controller, paths, beta=resolved_beta,
            )
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite loss at epoch {epoch}, batch {batch_index}.")
            (loss / args.gradient_accumulation).backward()
            batch_samples = reference.shape[0]
            sample_count += batch_samples
            for name, value in components.items():
                sums[name] += float(value.detach().cpu()) * batch_samples
            last_batch = batch_index == len(loader)
            smoke_end = args.max_train_batches is not None and batch_index == args.max_train_batches
            if batch_index % args.gradient_accumulation == 0 or last_batch or smoke_end:
                gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clip)
                if not torch.isfinite(gradient_norm):
                    raise FloatingPointError(f"Non-finite gradient at epoch {epoch}, batch {batch_index}.")
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                optimizer_steps += 1

        train_metrics = {name: value / max(1, sample_count) for name, value in sums.items()}
        record: dict[str, Any] = {
            "epoch": epoch, "samples": sample_count, "optimizer_steps": optimizer_steps,
            "duration_seconds": time.perf_counter() - epoch_start, "train": train_metrics,
        }
        should_validate = epoch == 1 or epoch == args.epochs or epoch % args.validation_interval == 0
        if should_validate:
            dev = evaluate_phase2_development(
                model, dataset_dir, device, manifests["development"], include_records=False,
            )
            stress = evaluate_phase2_stress(
                model, dataset_dir, device, manifests["stress"], include_records=False,
            )
            gate = development_gate(baseline_dev, dev)
            last_validation = {"development": dev, "stress": stress, "gate": gate}
            record["validation"] = last_validation
            robust = float(dev["robust_development_score"])
            print(f"Epoch {epoch}: loss={train_metrics['total_loss']:.6f}; dev S={dev['primary_score_db']:.4f}, "
                  f"R={dev['rebound_score_db']:.4f}, robust={robust:.4f}, eligible={gate['passed']}")
            if gate["passed"] and robust > best_robust:
                best_robust, best_epoch = robust, epoch
                save_checkpoint(checkpoint_dir / "best_dev_robust.pt", model, optimizer, epoch, config, manifests, last_validation)
            if gate["passed"] and dev["primary_score_db"] > best_primary:
                best_primary = float(dev["primary_score_db"])
                save_checkpoint(checkpoint_dir / "best_primary.pt", model, optimizer, epoch, config, manifests, last_validation)
            if gate["passed"] and dev["rebound_score_db"] < best_rebound:
                best_rebound = float(dev["rebound_score_db"])
                save_checkpoint(checkpoint_dir / "best_rebound.pt", model, optimizer, epoch, config, manifests, last_validation)
            if gate["passed"] and robust >= early_reference + args.early_stop_min_delta:
                early_reference, stale = robust, 0
            else:
                stale += 1
        save_checkpoint(checkpoint_dir / "latest.pt", model, optimizer, epoch, config, manifests, last_validation)
        append_jsonl(history_path, record)
        if should_validate and stale >= args.early_stop_patience:
            print(f"Early stopping after {stale} stale validations.")
            break

    selected = checkpoint_dir / "best_dev_robust.pt"
    selection_fallback = False
    if not selected.is_file():
        selected = checkpoint_dir / "latest.pt"
        selection_fallback = True
    selected_checkpoint = torch.load(selected, map_location=device, weights_only=False)
    model.load_state_dict(selected_checkpoint["model_state_dict"])
    final_dev = evaluate_phase2_development(model, dataset_dir, device, manifests["development"], include_records=True)
    final_stress = evaluate_phase2_stress(model, dataset_dir, device, manifests["stress"], include_records=True)
    summary: dict[str, Any] = {
        "mode": f"P2-{args.experiment}", "initial_checkpoint": str(checkpoint_path),
        "best_epoch": best_epoch, "selected_checkpoint": str(selected),
        "selection_fallback_to_latest": selection_fallback,
        "training_seconds": time.perf_counter() - start_time,
        "baseline_development_metrics": baseline_dev,
        "baseline_stress_metrics": baseline_stress,
        "final_development_metrics": final_dev,
        "final_stress_metrics": final_stress,
        "development_gate": development_gate(baseline_dev, final_dev),
    }
    if args.evaluate_final:
        baseline_final_model = load_model(checkpoint_path, device)
        baseline_final = evaluate_phase2_development(
            baseline_final_model, dataset_dir, device, manifests["final"], include_records=True,
        )
        candidate_final = evaluate_phase2_development(
            model, dataset_dir, device, manifests["final"], include_records=True,
        )
        summary["baseline_final_unseen_metrics"] = baseline_final
        summary["final_unseen_metrics"] = candidate_final
        summary["phase2_acceptance"] = phase2_acceptance(
            baseline_dev, baseline_final, final_dev, candidate_final,
        )
    save_json(output_dir / "summary.json", summary)
    print(f"Selected {selected}; final-only paths evaluated={args.evaluate_final}")


if __name__ == "__main__":
    main()
