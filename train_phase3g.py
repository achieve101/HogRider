"""Offline training entry point for Phase-3G generative FIR controllers."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import random
import shutil
import time
import uuid
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
from phase3g_data import (
    NEIGHBOR_POLICY_E10A,
    NEIGHBOR_POLICY_SINGLE,
    Phase3GSequenceDataset,
    build_multispace_neighbor_table,
    build_phase3g_manifest,
    save_phase3g_manifest,
)
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
CHECKPOINT_FORMAT_VERSION=2


def _canonical_sha256(value: dict[str, Any]) -> str:
    encoded=json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def resolve_e10a_correction(
    spec_path: str | Path,
    dataset_dir: str | Path,
    template_path: str | Path,
    train_paths: list[int],
) -> dict[str, Any]:
    """Validate the frozen E10-A protocol and return its retained-only table."""
    path=Path(spec_path)
    spec=json.loads(path.read_text(encoding="utf-8"))
    if spec.get("status") != "implementation_ready_not_run":
        raise ValueError("E10-A correction spec must be closed and not yet run.")
    correction=spec.get("selected_correction", {})
    if correction.get("experiment_id") != "E10-A":
        raise ValueError("Only the diagnosed E10-A correction is supported.")
    closure=spec.get("protocol_closure", {})
    recorded_hash=closure.get("closure_sha256")
    closure_payload={key:value for key,value in closure.items() if key != "closure_sha256"}
    if recorded_hash != _canonical_sha256(closure_payload):
        raise ValueError("E10-A protocol closure hash is invalid.")
    dataset_root=Path(dataset_dir)
    template=Path(template_path)
    from phase3r_templates import sha256_file
    expected_inputs=closure["inputs"]
    if sha256_file(dataset_root/"sh.npy") != expected_inputs["dataset/sh.npy"]:
        raise ValueError("E10-A secondary-path input hash changed after protocol closure.")
    if sha256_file(template) != expected_inputs["artifacts/phase3r_innovation_templates.npz"]:
        raise ValueError("E10-A primary-template input hash changed after protocol closure.")

    paths=[int(value) for value in train_paths]
    if paths == list(range(8)):
        serialized=closure["global_neighbor_table"]
        table_name="global"
    else:
        held=[value for value in range(8) if value not in paths]
        if len(paths) != 7 or len(held) != 1:
            raise ValueError("E10-A training paths must be all eight or one strict LOPO subset.")
        serialized=closure["fold_neighbor_tables"][str(held[0]+1)]
        table_name=f"fold_{held[0]+1}"
    table={int(first)-1:tuple(int(second)-1 for second in values) for first,values in serialized.items()}
    if set(table) != set(paths):
        raise ValueError("E10-A neighbor table does not match the retained training paths.")

    with np.load(template, allow_pickle=False) as artifact:
        secondary=np.load(dataset_root/"sh.npy", allow_pickle=False).T[:8]
        recomputed,_=build_multispace_neighbor_table(
            secondary, artifact["primary_real"], artifact["primary_imag"], artifact["band_mask"],
            paths, neighbor_count=3,
        )
    if table != recomputed:
        raise ValueError("E10-A frozen neighbor table does not match its registered algorithm.")
    return {
        "spec_path":str(path), "spec_sha256":sha256_file(path),
        "experiment_id":"E10-A", "table_name":table_name,
        "closure_sha256":recorded_hash, "neighbor_table":table,
        "frozen_training":spec["frozen_training"],
    }


def validate_e10a_training_config(
    correction: dict[str, Any], args: argparse.Namespace, epochs: int, samples: int,
) -> None:
    frozen=correction["frozen_training"]
    checks={
        "batch_size":(args.batch_size, int(frozen["batch_size"])),
        "gradient_accumulation":(args.gradient_accumulation, int(frozen["gradient_accumulation"])),
        "samples_per_epoch":(samples, int(frozen["samples_per_epoch"])),
        "hidden_size":(args.hidden_size, int(frozen["hidden_size"])),
        "latent_size":(args.latent_size, int(frozen["latent_size"])),
        "generator_lr":(args.generator_lr, float(frozen["generator_lr"])),
        "dictionary_lr":(args.dictionary_lr, float(frozen["dictionary_lr"])),
        "epochs":(epochs, int(frozen["warmup_epochs"] if args.stage == "warmup" else frozen["generalize_epochs"])),
    }
    mismatches={key:{"actual":actual,"expected":expected} for key,(actual,expected) in checks.items() if actual != expected}
    if mismatches:
        raise ValueError(f"E10-A must preserve the frozen training configuration: {mismatches}")


def set_seed(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def save_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def append_jsonl(path: Path, value: Any) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, ensure_ascii=False)+"\n")


def file_sha256(path: str | Path) -> str:
    digest=hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda:handle.read(1024*1024), b""):
            digest.update(block)
    return digest.hexdigest()


def capture_rng_state(loader_generator: torch.Generator) -> dict[str, Any]:
    """Capture every RNG stream which can affect the next training epoch."""
    return {
        "python":random.getstate(),
        "numpy":np.random.get_state(),
        "torch_cpu":torch.get_rng_state(),
        "torch_cuda":torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        "data_loader_generator":loader_generator.get_state(),
    }


def restore_rng_state(state: dict[str, Any], loader_generator: torch.Generator) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    cuda_state=state.get("torch_cuda")
    if cuda_state is not None:
        if not torch.cuda.is_available():
            raise RuntimeError("Checkpoint contains CUDA RNG state but CUDA is unavailable.")
        torch.cuda.set_rng_state_all(cuda_state)
    loader_generator.set_state(state["data_loader_generator"])


def optimizer_to(optimizer: optim.Optimizer, device: torch.device) -> None:
    for state in optimizer.state.values():
        for key,value in state.items():
            if torch.is_tensor(value):
                state[key]=value.to(device)


def _atomic_torch_save(value: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary=path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        torch.save(value, temporary)
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _history_records(path: Path, through_epoch: int | None=None) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    records=[]
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        value=json.loads(line)
        if through_epoch is None or int(value["epoch"]) <= through_epoch:
            records.append(value)
    return records


def _write_history(path: Path, records: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(value, ensure_ascii=False)+"\n" for value in records),
        encoding="utf-8",
    )


def _legacy_training_state(checkpoint: dict[str, Any], checkpoint_path: Path) -> dict[str, Any]:
    epoch=int(checkpoint["epoch"])
    records=_history_records(checkpoint_path.parents[1]/"history.jsonl", epoch)
    if records:
        scored=[
            (float(value["validation"]["development"]["phase3_selection_score"]), int(value["epoch"]))
            for value in records
        ]
        best_score,best_epoch=max(scored)
        stale=epoch-best_epoch
    else:
        metrics=(checkpoint.get("metrics") or {}).get("development", {})
        best_score=float(metrics.get("phase3_selection_score", -float("inf")))
        best_epoch=epoch
        stale=0
    return {
        "completed_epoch":epoch, "best_score":best_score,
        "best_epoch":best_epoch, "stale":stale,
    }


def _simulate_legacy_loader_state(
    generator: torch.Generator,
    *,
    samples_per_epoch: int,
    completed_epochs: int,
) -> None:
    """Best-effort reconstruction for v1 checkpoints; intentionally non-formal.

    Every DataLoader iterator draws one worker base seed and the RandomSampler
    then draws one permutation.  The original run made one calibration pass
    before its completed epochs.
    """
    for _ in range(completed_epochs+1):
        torch.empty((), dtype=torch.int64).random_(generator=generator)
        torch.randperm(samples_per_epoch, generator=generator)


def _resume_source_root(checkpoint_path: Path) -> Path:
    return checkpoint_path.parents[1] if checkpoint_path.parent.name == "checkpoints" else checkpoint_path.parent


def _copy_resume_history_and_best(
    checkpoint_path: Path,
    root: Path,
    checkpoints: Path,
    completed_epoch: int,
    best_epoch: int | None,
) -> None:
    source_root=_resume_source_root(checkpoint_path)
    records=_history_records(source_root/"history.jsonl", completed_epoch)
    if records:
        _write_history(root/"history.jsonl", records)
    source_best=source_root/"checkpoints"/"best_phase3g_selection.pt"
    if source_best.is_file():
        shutil.copy2(source_best, checkpoints/"best_phase3g_selection.pt")
        if best_epoch is not None:
            shutil.copy2(source_best, checkpoints/f"best_epoch_{best_epoch:04d}.pt")


def _validate_exact_resume_config(
    checkpoint: dict[str, Any],
    args: argparse.Namespace,
    *,
    epochs: int,
    samples: int,
    device: torch.device,
    allow_patience_change: bool=False,
) -> None:
    frozen=checkpoint.get("config", {})
    checks={
        "stage":(args.stage, checkpoint.get("stage", frozen.get("stage"))),
        "dataset_dir":(str(Path(args.dataset_dir).resolve()), str(Path(frozen.get("dataset_dir", args.dataset_dir)).resolve())),
        "seed":(args.seed, int(frozen.get("seed", args.seed))),
        "train_paths":(list(args.train_paths), list(frozen.get("train_paths", args.train_paths))),
        "hidden_size":(args.hidden_size, int(frozen.get("hidden_size", args.hidden_size))),
        "latent_size":(args.latent_size, int(frozen.get("latent_size", args.latent_size))),
        "samples_per_epoch":(samples, int(frozen.get("samples_per_epoch_resolved", samples))),
        "batch_size":(args.batch_size, int(frozen.get("batch_size", args.batch_size))),
        "gradient_accumulation":(
            args.gradient_accumulation,
            int(frozen.get("gradient_accumulation", args.gradient_accumulation)),
        ),
        "generator_lr":(args.generator_lr, float(frozen.get("generator_lr", args.generator_lr))),
        "dictionary_lr":(args.dictionary_lr, float(frozen.get("dictionary_lr", args.dictionary_lr))),
        "gradient_clip":(args.gradient_clip, float(frozen.get("gradient_clip", args.gradient_clip))),
        "max_train_batches":(args.max_train_batches, frozen.get("max_train_batches", args.max_train_batches)),
        "num_workers":(args.num_workers, int(frozen.get("num_workers", args.num_workers))),
        "correction_spec":(args.correction_spec, frozen.get("correction_spec", args.correction_spec)),
        "stress_cases":(args.stress_cases, int(frozen.get("stress_cases", args.stress_cases))),
        "p3r_checkpoint":(str(args.p3r_checkpoint), str(frozen.get("p3r_checkpoint", args.p3r_checkpoint))),
        "resolved_device":(str(device), str(frozen.get("resolved_device", device))),
    }
    if not allow_patience_change:
        checks["patience"]=(args.patience, int(frozen.get("patience", args.patience)))
    mismatches={key:{"actual":actual,"expected":expected} for key,(actual,expected) in checks.items() if actual != expected}
    if mismatches:
        raise ValueError(f"Exact resume configuration changed: {mismatches}")
    if epochs <= int(checkpoint["epoch"]):
        raise ValueError("--epochs must be greater than the resumed checkpoint epoch.")


def resume_is_legacy(checkpoint: dict[str, Any]) -> bool:
    return bool(
        int(checkpoint.get("checkpoint_format_version", 1)) < CHECKPOINT_FORMAT_VERSION
        or "rng_state" not in checkpoint
        or "training_state" not in checkpoint
    )


def require_resume_compatibility(checkpoint: dict[str, Any], allow_legacy: bool) -> bool:
    legacy=resume_is_legacy(checkpoint)
    if legacy and not allow_legacy:
        raise ValueError(
            "This checkpoint predates exact resume metadata; pass --allow-legacy-resume "
            "to run a non-formal compatibility control."
        )
    return legacy


def _validate_resume_manifest(
    checkpoint_path: Path,
    filename: str,
    current: dict[str, Any],
) -> None:
    source=_resume_source_root(checkpoint_path)/filename
    if not source.is_file():
        raise ValueError(f"Resume source is missing {filename}.")
    previous=json.loads(source.read_text(encoding="utf-8"))
    if _canonical_sha256(previous) != _canonical_sha256(current):
        raise ValueError(f"Resume {filename} changed, including data hashes or sealed-path policy.")


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
    *,
    loader_generator: torch.Generator | None=None,
    best_score: float | None=None,
    best_epoch: int | None=None,
    stale: int | None=None,
    resume_mode: str="fresh",
    parent_checkpoint: str | Path | None=None,
) -> None:
    payload={
        "checkpoint_format_version":CHECKPOINT_FORMAT_VERSION,
        "phase": "3G", "stage": config["stage"], "epoch": epoch,
        "model_config": model.model_config, "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(), "config": config, "metrics": metrics,
        "training_state":{
            "completed_epoch":epoch, "best_score":best_score,
            "best_epoch":best_epoch, "stale":stale,
        },
        "resume_mode":resume_mode,
        "parent_checkpoint":None if parent_checkpoint is None else str(parent_checkpoint),
        "parent_checkpoint_sha256":None if parent_checkpoint is None else file_sha256(parent_checkpoint),
    }
    if loader_generator is not None:
        payload["rng_state"]=capture_rng_state(loader_generator)
    _atomic_torch_save(payload, path)


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
    parser.add_argument("--resume-checkpoint", default=None)
    parser.add_argument("--allow-legacy-resume", action="store_true")
    parser.add_argument("--save-every-epoch", action="store_true")
    parser.add_argument("--p3r-checkpoint", default=DEFAULT_P3R)
    parser.add_argument("--oracle-summary", default=DEFAULT_ORACLE_SUMMARY)
    parser.add_argument("--correction-spec", default=None)
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
    if args.checkpoint and args.resume_checkpoint:
        parser.error("--checkpoint and --resume-checkpoint are mutually exclusive.")
    if args.allow_legacy_resume and not args.resume_checkpoint:
        parser.error("--allow-legacy-resume requires --resume-checkpoint.")
    epochs=args.epochs or (5 if args.stage == "warmup" else 15)
    samples=args.samples_per_epoch or (256 if args.stage == "warmup" else 128)
    if args.device == "auto":
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else: device=torch.device(args.device)
    set_seed(args.seed)
    root=Path(args.output_dir or f"runs/phase3g_{args.stage}_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    resume_path=Path(args.resume_checkpoint).resolve() if args.resume_checkpoint else None
    resume_checkpoint=(
        torch.load(resume_path, map_location="cpu", weights_only=False)
        if resume_path is not None else None
    )
    resume_is_legacy=(
        require_resume_compatibility(resume_checkpoint, args.allow_legacy_resume)
        if resume_checkpoint is not None else False
    )
    if resume_checkpoint is not None:
        _validate_exact_resume_config(
            resume_checkpoint, args, epochs=epochs, samples=samples, device=device,
            allow_patience_change=resume_is_legacy,
        )
    root.mkdir(parents=True, exist_ok=False); checkpoints=root/"checkpoints"; checkpoints.mkdir()
    source_config=json.loads((Path(args.oracle_checkpoint).parents[1]/"config.json").read_text(encoding="utf-8"))
    train_noises=source_config["train_noises"]
    correction=(resolve_e10a_correction(
        args.correction_spec, args.dataset_dir, args.template, args.train_paths,
    ) if args.correction_spec else None)
    if correction:
        validate_e10a_training_config(correction, args, epochs, samples)
    if resume_checkpoint is not None:
        model=GenerativeInnovationFIRController(**resume_checkpoint["model_config"])
        model.load_state_dict(resume_checkpoint["model_state_dict"])
        model=model.to(device)
        if model.hidden_size != args.hidden_size or model.latent_size != args.latent_size:
            raise ValueError("Resume checkpoint capacity does not match the requested capacity.")
    elif args.stage == "warmup":
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
    neighbor_policy=(NEIGHBOR_POLICY_E10A if correction and args.stage == "generalize" else NEIGHBOR_POLICY_SINGLE)
    neighbor_table=(correction["neighbor_table"] if neighbor_policy == NEIGHBOR_POLICY_E10A else None)
    dataset=Phase3GSequenceDataset(
        args.dataset_dir, train_noises, train_paths=args.train_paths, samples_per_epoch=samples,
        block_size=model.block_size, synthesis_enabled=args.stage == "generalize",
        switch_probability=0.25 if args.stage == "generalize" else 0.0, seed=args.seed,
        neighbor_policy=neighbor_policy, neighbor_table=neighbor_table,
    )
    generator=torch.Generator().manual_seed(args.seed)
    loader=DataLoader(dataset, batch_size=args.batch_size, shuffle=True, generator=generator,
                      num_workers=args.num_workers, pin_memory=device.type == "cuda")
    correction_metadata=(None if correction is None else {
        "experiment_id":correction["experiment_id"], "spec_path":correction["spec_path"],
        "spec_sha256":correction["spec_sha256"], "closure_sha256":correction["closure_sha256"],
        "table_name":correction["table_name"], "active_in_this_stage":neighbor_policy == NEIGHBOR_POLICY_E10A,
    })
    synthesis_manifest=build_phase3g_manifest(
        args.dataset_dir, train_noises, path_indices=args.train_paths, seed=args.seed,
        neighbor_policy=neighbor_policy, neighbor_table=neighbor_table,
        correction_metadata=correction_metadata,
    )
    if resume_path is not None:
        _validate_resume_manifest(resume_path, "synthesis_manifest.json", synthesis_manifest)
    save_phase3g_manifest(root/"synthesis_manifest.json", synthesis_manifest)
    manifests=build_phase3_manifests(args.dataset_dir, args.seed)
    manifests["development"]["path_indices_zero_based"]=list(args.train_paths)
    manifests["development"]["split"]="phase3g_development"
    if resume_path is not None:
        _validate_resume_manifest(resume_path, "validation_manifests.json", manifests)
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
        "correction":correction_metadata,
        "resume_mode":(
            "legacy-compatible" if resume_is_legacy or (
                resume_checkpoint is not None
                and not bool(resume_checkpoint.get("config", {}).get("formal_candidate_eligible", True))
            ) else "exact" if resume_checkpoint is not None else "fresh"
        ),
        "resume_parent":None if resume_path is None else str(resume_path),
        "formal_candidate_eligible":bool(
            not resume_is_legacy
            and (
                resume_checkpoint is None
                or resume_checkpoint.get("config", {}).get("formal_candidate_eligible", True)
            )
        ),
    })
    save_json(root/"config.json", config); save_json(root/"p1_baseline.json", p1_baseline)
    if resume_checkpoint is None:
        calibrate_features(model, loader, device, args.calibration_batches)
    optimizer=optim.Adam([
        {"params":list(model.gru.parameters())+list(model.latent_head.parameters()), "lr":args.generator_lr},
        {"params":[model.residual_dictionary], "lr":args.dictionary_lr},
    ], amsgrad=True)
    best_score=-float("inf"); best_epoch=None; stale=0; last_validation=None; start_epoch=1
    resume_mode=str(config["resume_mode"])
    formal_candidate_eligible=bool(config["formal_candidate_eligible"])
    if resume_checkpoint is not None:
        optimizer.load_state_dict(resume_checkpoint["optimizer_state_dict"])
        optimizer_to(optimizer, device)
        training_state=(
            _legacy_training_state(resume_checkpoint, resume_path)
            if resume_is_legacy else resume_checkpoint["training_state"]
        )
        completed_epoch=int(training_state["completed_epoch"])
        start_epoch=completed_epoch+1
        best_score=float(training_state["best_score"])
        best_epoch=None if training_state["best_epoch"] is None else int(training_state["best_epoch"])
        stale=int(training_state["stale"])
        _copy_resume_history_and_best(resume_path, root, checkpoints, completed_epoch, best_epoch)
        if resume_is_legacy:
            _simulate_legacy_loader_state(
                generator, samples_per_epoch=samples, completed_epochs=completed_epoch,
            )
        else:
            restore_rng_state(resume_checkpoint["rng_state"], generator)
        resume_manifest={
            "mode":resume_mode,
            "formal_candidate_eligible":bool(config["formal_candidate_eligible"]),
            "source_checkpoint":str(resume_path),
            "source_checkpoint_sha256":file_sha256(resume_path),
            "source_epoch":completed_epoch, "target_total_epochs":epochs,
            "start_epoch":start_epoch, "feature_recalibrated":False,
            "optimizer_restored":True, "rng_restored_exactly":not resume_is_legacy,
            "best_score":best_score, "best_epoch":best_epoch, "stale":stale,
        }
        save_json(root/"resume_manifest.json", resume_manifest)
    start_time=time.perf_counter()
    stop_reason="epoch_budget_exhausted"
    for epoch in range(start_epoch, epochs+1):
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
        improved=score > best_score
        if improved:
            best_score=score; best_epoch=epoch; stale=0
        else:
            stale+=1
        checkpoint_kwargs={
            "loader_generator":generator, "best_score":best_score,
            "best_epoch":best_epoch, "stale":stale, "resume_mode":resume_mode,
            "parent_checkpoint":resume_path,
        }
        if args.save_every_epoch:
            save_checkpoint(
                checkpoints/f"epoch_{epoch:04d}.pt", model, optimizer, epoch,
                config, last_validation, **checkpoint_kwargs,
            )
        save_checkpoint(
            checkpoints/"latest.pt", model, optimizer, epoch, config,
            last_validation, **checkpoint_kwargs,
        )
        if improved:
            if args.save_every_epoch:
                save_checkpoint(
                    checkpoints/f"best_epoch_{epoch:04d}.pt", model, optimizer, epoch,
                    config, last_validation, **checkpoint_kwargs,
                )
                shutil.copy2(
                    checkpoints/f"best_epoch_{epoch:04d}.pt",
                    checkpoints/"best_phase3g_selection.pt",
                )
            else:
                save_checkpoint(
                    checkpoints/"best_phase3g_selection.pt", model, optimizer,
                    epoch, config, last_validation, **checkpoint_kwargs,
                )
        print(f"Epoch {epoch}/{epochs}: S={development['primary_score_db']:.4f}, R={development['rebound_score_db']:.4f}, D={score:.4f}")
        if should_stop_warmup(
            args.stage,
            args.train_paths,
            float(development["primary_score_db"]),
            None if p3r_metrics is None else float(p3r_metrics["primary_score_db"]),
        ):
            stop_reason="warmup_guard"
            break
        if stale >= args.patience:
            stop_reason="patience_exhausted"
            break
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
        "resume_mode":resume_mode,
        "formal_candidate_eligible":formal_candidate_eligible,
        "start_epoch":start_epoch, "last_completed_epoch":epoch,
        "target_total_epochs":epochs, "stop_reason":stop_reason,
        "stale_epochs":stale, "best_selection_score":best_score,
    }
    save_json(root/"summary.json", summary)


if __name__ == "__main__":
    main()
