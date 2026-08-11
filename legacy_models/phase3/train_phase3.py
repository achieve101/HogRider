"""Train common FIR, oracle experts, or feedback routing for DEEPANC Phase 3."""

from __future__ import annotations

import argparse
import json
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from dataset import apply_dynamic_path
from legacy_models.phase0_phase1.model import TimeDomainANC
from phase3_closed_loop import compute_phase3_loss, rollout_feedback_closed_loop
from phase3_data import Phase3SequenceDataset
from phase3_model import FeedbackFIRController
from phase3_validation import (
    build_phase3_manifests,
    evaluate_path_switch_stress,
    evaluate_phase3_feedback,
    evaluate_phase3_oracle,
    phase3_development_gate,
    phase3_oracle_gate,
)
from legacy_models.phase0_phase1.train import resolve_device, scan_noise_files, seed_worker, set_global_seed
from v6_metrics import compute_v6_loss


DEFAULT_P1 = "runs/phase1_suite_seed2026/P1-E2/checkpoints/best_official_composite.pt"
DEFAULT_BASELINE = "runs/phase2_suite_seed2026/P2-E2/summary.json"


def parse_path_list(value: str) -> list[int]:
    result = [int(item.strip()) - 1 for item in value.split(",") if item.strip()]
    if not result or len(set(result)) != len(result) or any(item not in range(8) for item in result):
        raise argparse.ArgumentTypeError("Use unique one-based Phase-3 paths in 1..8.")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("common", "expert", "gate", "joint"), required=True)
    parser.add_argument("--dataset-dir", default="dataset")
    parser.add_argument("--checkpoint", default=None, help="P1 checkpoint for common; Phase-3 checkpoint otherwise.")
    parser.add_argument("--baseline-summary", default=DEFAULT_BASELINE)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--fir-length", type=int, default=2048)
    parser.add_argument("--hidden-size", type=int, default=24)
    parser.add_argument("--block-size", type=int, default=240)
    parser.add_argument("--gate-hidden-size", type=int, default=None)
    parser.add_argument("--train-paths", type=parse_path_list, default=parse_path_list("1,2,3,4,5,6,7,8"))
    parser.add_argument("--dev-paths", type=parse_path_list, default=parse_path_list("1,2,3,4,5,6,7,8"))
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--distill-epochs", type=int, default=5)
    parser.add_argument("--samples-per-epoch", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--gradient-accumulation", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--expert-learning-rate", type=float, default=1e-5)
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument("--early-stop-patience", type=int, default=5)
    parser.add_argument("--validation-interval", type=int, default=1)
    parser.add_argument("--switch-probability", type=float, default=None)
    parser.add_argument("--augmentation-probability", type=float, default=None)
    parser.add_argument("--calibration-batches", type=int, default=4)
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


def load_p1(path: Path, device: torch.device) -> TimeDomainANC:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model = TimeDomainANC(hidden_channels=32, num_layers=10).to(device)
    model.load_state_dict(checkpoint.get("model_state_dict", checkpoint))
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def load_phase3(path: Path, device: torch.device) -> tuple[FeedbackFIRController, dict[str, Any]]:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    config = checkpoint["model_config"]
    model = FeedbackFIRController(**config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    return model, checkpoint


def save_checkpoint(
    path: Path, model: FeedbackFIRController, optimizer: optim.Optimizer,
    epoch: int, stage: str, config: dict[str, Any], manifests: dict[str, Any],
    metrics: dict[str, Any] | None,
) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save({
        "phase": 3, "stage": stage, "epoch": epoch,
        "model_config": {
            "num_experts": model.num_experts, "fir_length": model.fir_length,
            "hidden_size": model.hidden_size, "block_size": model.block_size,
            "alpha_update": model.alpha_update, "output_limit": model.output_limit,
        },
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(), "config": config,
        "validation_manifests": manifests, "metrics": metrics,
    }, temporary)
    os.replace(temporary, path)


def baseline_metrics(path: Path) -> dict[str, Any]:
    source = json.loads(path.read_text(encoding="utf-8"))
    return source["baseline_development_metrics"]


def calibrate_features(
    model: FeedbackFIRController, loader: DataLoader, device: torch.device,
    batches: int,
) -> None:
    values = []
    model.eval()
    with torch.inference_mode():
        for index, (reference, target, paths, slots, labels, _, _) in enumerate(loader):
            if index >= batches:
                break
            rollout = rollout_feedback_closed_loop(
                model, reference.to(device), target.to(device), paths.to(device),
                slots.to(device), labels.to(device), truncate_blocks=0,
            )
            values.append(rollout.raw_features.reshape(-1, rollout.raw_features.shape[-1]).cpu())
    stacked = torch.cat(values)
    model.set_feature_statistics(stacked.mean(dim=0), stacked.std(dim=0).clamp_min(1e-3))


def main() -> None:
    args = parse_args()
    defaults = {
        "common": {"epochs": 10, "samples": 256, "lr": 3e-4, "switch": 0.0, "augment": 0.0},
        "expert": {"epochs": 15, "samples": 256, "lr": 3e-4, "switch": 0.0, "augment": 0.0},
        "gate": {"epochs": 15, "samples": 128, "lr": 3e-4, "switch": 0.0, "augment": 0.0},
        "joint": {"epochs": 10, "samples": 128, "lr": 1e-4, "switch": 0.25, "augment": 0.30},
    }[args.stage]
    epochs = args.epochs or defaults["epochs"]
    samples_per_epoch = args.samples_per_epoch or defaults["samples"]
    learning_rate = args.learning_rate or defaults["lr"]
    switch_probability = defaults["switch"] if args.switch_probability is None else args.switch_probability
    augmentation_probability = defaults["augment"] if args.augmentation_probability is None else args.augmentation_probability
    if any(value <= 0 for value in (epochs, samples_per_epoch, args.batch_size, args.gradient_accumulation)):
        raise ValueError("training sizes must be positive.")
    set_global_seed(args.seed)
    device = resolve_device(args.device)
    output_dir = Path(args.output_dir or f"runs/phase3_{args.stage}_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    output_dir.mkdir(parents=True, exist_ok=False)
    checkpoint_dir = output_dir / "checkpoints"; checkpoint_dir.mkdir()
    dataset_dir = Path(args.dataset_dir)
    checkpoint_path = Path(args.checkpoint or (DEFAULT_P1 if args.stage == "common" else ""))
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Required checkpoint not found: {checkpoint_path}")
    manifests = build_phase3_manifests(dataset_dir, args.seed)
    baseline = baseline_metrics(Path(args.baseline_summary))

    if args.stage == "common":
        model = FeedbackFIRController(
            num_experts=8, fir_length=args.fir_length, hidden_size=args.hidden_size,
            block_size=args.block_size,
        ).to(device)
        source_checkpoint = None
    else:
        model, source_checkpoint = load_phase3(checkpoint_path, device)
        if model.fir_length != args.fir_length and args.fir_length != 2048:
            raise ValueError("Changing FIR length requires rerunning the common stage.")
        if args.stage == "gate" and args.gate_hidden_size is not None and args.gate_hidden_size != model.hidden_size:
            replacement = FeedbackFIRController(
                num_experts=model.num_experts, fir_length=model.fir_length,
                hidden_size=args.gate_hidden_size, block_size=model.block_size,
                alpha_update=model.alpha_update, output_limit=model.output_limit,
            ).to(device)
            with torch.no_grad():
                replacement.expert_filters.copy_(model.expert_filters)
                replacement.feature_mean.copy_(model.feature_mean)
                replacement.feature_std.copy_(model.feature_std)
            model = replacement

    all_noises = scan_noise_files(dataset_dir / "NOISE")
    if len(all_noises) < 8:
        raise ValueError("Phase 3 expects all eight official noises.")
    train_noises = all_noises[:-2]
    train_dataset = Phase3SequenceDataset(
        dataset_dir, train_noises, train_paths=args.train_paths,
        samples_per_epoch=samples_per_epoch,
        block_size=model.block_size, switch_probability=switch_probability,
        augmentation_probability=augmentation_probability, seed=args.seed,
    )
    generator = torch.Generator().manual_seed(args.seed)
    loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, generator=generator,
        num_workers=args.num_workers, pin_memory=device.type == "cuda", worker_init_fn=seed_worker,
    )
    config = vars(args).copy()
    config.update({
        "epochs_resolved": epochs, "samples_per_epoch_resolved": samples_per_epoch,
        "learning_rate_resolved": learning_rate,
        "switch_probability_resolved": switch_probability,
        "augmentation_probability_resolved": augmentation_probability,
        "train_noises": train_noises, "fixed_validation_noises": all_noises[-2:],
        "resolved_device": str(device), "complexity": model.get_complexity(),
        "paths_policy": "paths 1-8 train/development; paths 9-10 final-only",
    })
    save_json(output_dir / "config.json", config)
    manifests["development"]["path_indices_zero_based"] = list(args.dev_paths)
    manifests["development"]["split"] = "phase3_custom_development"
    manifests["path_switch"]["source_manifest"] = manifests["development"]
    available = set(args.dev_paths)
    manifests["path_switch"]["pairs_zero_based"] = [
        pair for pair in manifests["path_switch"]["pairs_zero_based"]
        if pair[0] in available and pair[1] in available
    ]
    save_json(output_dir / "validation_manifests.json", manifests)
    save_json(output_dir / "baseline_development_metrics.json", baseline)

    if args.stage == "common":
        teacher = load_p1(checkpoint_path, device)
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        model.expert_filters.requires_grad_(True)
        optimizer = optim.Adam([model.expert_filters], lr=1e-3, amsgrad=True)
        for epoch in range(1, args.distill_epochs + 1):
            train_dataset.set_epoch(epoch)
            total = count = 0.0
            for batch_index, (reference, _, _, _, _, _, _) in enumerate(loader, start=1):
                if args.max_train_batches and batch_index > args.max_train_batches:
                    break
                reference = reference.to(device)
                with torch.inference_mode():
                    teacher_output = teacher(reference)
                _, student = model.oracle_forward(
                    reference, torch.zeros(reference.shape[0], dtype=torch.long, device=device),
                )
                loss = (student - teacher_output).square().mean() / (teacher_output.square().mean() + 1e-12)
                optimizer.zero_grad(set_to_none=True); loss.backward()
                torch.nn.utils.clip_grad_norm_([model.expert_filters], args.gradient_clip)
                optimizer.step()
                total += float(loss.detach()) * reference.shape[0]; count += reference.shape[0]
            with torch.no_grad():
                model.expert_filters[1:].copy_(model.expert_filters[0])
            append_jsonl(output_dir / "distillation_history.jsonl", {"epoch": epoch, "loss": total / max(1, count)})
        optimizer = optim.Adam([model.expert_filters], lr=learning_rate, amsgrad=True)
    elif args.stage == "expert":
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        model.expert_filters.requires_grad_(True)
        optimizer = optim.Adam([model.expert_filters], lr=learning_rate, amsgrad=True)
    elif args.stage == "gate":
        for parameter in model.parameters():
            parameter.requires_grad_(True)
        model.expert_filters.requires_grad_(False)
        calibrate_features(model, loader, device, args.calibration_batches)
        optimizer = optim.Adam(list(model.gru.parameters()) + list(model.route_head.parameters()), lr=learning_rate, amsgrad=True)
    else:
        for parameter in model.parameters():
            parameter.requires_grad_(True)
        optimizer = optim.Adam([
            {"params": list(model.gru.parameters()) + list(model.route_head.parameters()), "lr": learning_rate},
            {"params": [model.expert_filters], "lr": args.expert_learning_rate},
        ], amsgrad=True)

    history_path = output_dir / "history.jsonl"
    best_score = -float("inf"); best_epoch = None; stale = 0; last_validation = None
    stop_reason = None
    training_start = time.perf_counter()
    for epoch in range(1, epochs + 1):
        train_dataset.set_epoch(epoch + (args.distill_epochs if args.stage == "common" else 0))
        model.train(); optimizer.zero_grad(set_to_none=True)
        sums: dict[str, float] = defaultdict_float(args.stage)
        samples = steps = 0
        for batch_index, (reference, target, paths, slots, labels, first_path, _) in enumerate(loader, start=1):
            if args.max_train_batches and batch_index > args.max_train_batches:
                break
            reference, target, paths = reference.to(device), target.to(device), paths.to(device)
            slots, labels, first_path = slots.to(device), labels.to(device), first_path.to(device)
            if args.stage in ("common", "expert"):
                selected = torch.zeros_like(first_path) if args.stage == "common" else first_path
                output, raw = model.oracle_forward(reference, selected)
                residual = target - apply_dynamic_path(output, paths[:, 0])
                loss, components = compute_v6_loss(
                    target, residual, output, 0.7, 0.3, time_weight=0.1,
                    guard_weight=1.0, guard_limit=0.98,
                )
                raw_violation = torch.relu(raw.abs() - 0.9).square()
                raw_guard = raw_violation.mean() + raw_violation.amax()
                loss = loss + raw_guard
                components["total_loss"] = loss.detach()
                components["guard_loss"] = raw_guard.detach()
            else:
                rollout = rollout_feedback_closed_loop(model, reference, target, paths, slots, labels)
                loss, components = compute_phase3_loss(target, rollout, labels)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite Phase-3 loss at epoch {epoch}, batch {batch_index}.")
            (loss / args.gradient_accumulation).backward()
            batch_samples = reference.shape[0]; samples += batch_samples
            for name, value in components.items():
                if name in sums:
                    sums[name] += float(value.detach().cpu()) * batch_samples
            last = batch_index == len(loader) or (args.max_train_batches and batch_index == args.max_train_batches)
            if batch_index % args.gradient_accumulation == 0 or last:
                norm = torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], args.gradient_clip)
                if not torch.isfinite(norm):
                    raise FloatingPointError("Non-finite Phase-3 gradient.")
                optimizer.step(); optimizer.zero_grad(set_to_none=True); steps += 1
        train_metrics = {name: value / max(1, samples) for name, value in sums.items()}
        record: dict[str, Any] = {"epoch": epoch, "samples": samples, "optimizer_steps": steps, "train": train_metrics}
        if epoch == 1 or epoch == epochs or epoch % args.validation_interval == 0:
            if args.stage in ("common", "expert"):
                if args.stage == "common":
                    with torch.no_grad():
                        model.expert_filters[1:].copy_(model.expert_filters[0])
                metrics = evaluate_phase3_oracle(model, dataset_dir, device, manifests["development"])
                gate = phase3_oracle_gate(baseline, metrics) if args.stage == "expert" else {"passed": True, "checks": {}}
                last_validation = {"development": metrics, "gate": gate}
            else:
                metrics = evaluate_phase3_feedback(model, dataset_dir, device, manifests["development"])
                stress = (
                    evaluate_path_switch_stress(model, dataset_dir, device, manifests["path_switch"])
                    if manifests["path_switch"]["pairs_zero_based"] else None
                )
                gate = phase3_development_gate(baseline, metrics)
                last_validation = {"development": metrics, "path_switch": stress, "gate": gate}
            record["validation"] = last_validation
            score = float(metrics["phase3_selection_score"])
            if score > best_score:
                best_score, best_epoch, stale = score, epoch, 0
                save_checkpoint(checkpoint_dir / "best_phase3_selection.pt", model, optimizer, epoch, args.stage, config, manifests, last_validation)
            else:
                stale += 1
            print(f"Epoch {epoch}/{epochs}: S={metrics['primary_score_db']:.4f}, R={metrics['rebound_score_db']:.4f}, "
                  f"worst={metrics['worst_path_primary_db']:.4f}, D={score:.4f}")
        save_checkpoint(checkpoint_dir / "latest.pt", model, optimizer, epoch, args.stage, config, manifests, last_validation)
        append_jsonl(history_path, record)
        if (
            args.stage in ("gate", "joint") and epoch >= 4 and "validation" in record
            and metrics["primary_score_db"] < baseline["primary_score_db"] - 2.0
            and metrics.get("route_accuracy_after_initialization", 0.0) < 0.35
        ):
            stop_reason = "feedback routing remained below 35% while primary score was over 2 dB below baseline"
            print(f"Stopping infeasible feedback branch: {stop_reason}.")
            break
        if stale >= args.early_stop_patience:
            stop_reason = f"selection score was stale for {stale} validations"
            break

    best_path = checkpoint_dir / "best_phase3_selection.pt"
    selected, _ = load_phase3(best_path, device)
    if args.stage in ("common", "expert"):
        final = evaluate_phase3_oracle(selected, dataset_dir, device, manifests["development"], include_records=True)
        final_gate = phase3_oracle_gate(baseline, final) if args.stage == "expert" else None
        switch = None
    else:
        final = evaluate_phase3_feedback(selected, dataset_dir, device, manifests["development"], include_records=True)
        switch = (
            evaluate_path_switch_stress(selected, dataset_dir, device, manifests["path_switch"], include_records=True)
            if manifests["path_switch"]["pairs_zero_based"] else None
        )
        final_gate = phase3_development_gate(baseline, final)
    summary = {
        "stage": args.stage, "initial_checkpoint": str(checkpoint_path), "best_epoch": best_epoch,
        "selected_checkpoint": str(best_path), "training_seconds": time.perf_counter() - training_start,
        "baseline_development_metrics": baseline, "final_development_metrics": final,
        "path_switch_metrics": switch, "acceptance": final_gate,
        "complexity": selected.get_complexity(), "final_paths_touched": False,
        "stop_reason": stop_reason,
    }
    save_json(output_dir / "summary.json", summary)


def defaultdict_float(stage: str) -> dict[str, float]:
    names = ["total_loss", "primary_loss", "rebound_loss", "time_loss", "guard_loss",
             "primary_score_db", "rebound_score_db", "controller_peak_abs"]
    if stage in ("gate", "joint"):
        names += ["route_loss", "gate_delta_loss", "raw_controller_peak_abs", "route_accuracy_after_initialization"]
    return {name: 0.0 for name in names}


if __name__ == "__main__":
    main()
