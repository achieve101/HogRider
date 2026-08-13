"""Run the preregistered Phase-3G 30 -> 40 -> 50 long-training protocol.

This orchestrator never evaluates sealed paths 9/10.  Exploration, the
legacy-compatible control, and three-seed replication use separate output
roots so interrupted or completed evidence is never overwritten.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch


ROOT=Path(__file__).resolve().parent
DEFAULT_PROTOCOL=ROOT/"artifacts/phase3g_longtrain_protocol.json"
DEFAULT_BASELINE=ROOT/"artifacts/phase3g_epoch15_baseline.json"
LEGACY_LATEST=ROOT/"runs/phase3g_suite_seed2027/P3G-E2/checkpoints/latest.pt"
FORMAL_BEST=ROOT/"runs/phase3g_suite_seed2027/P3G-E2/checkpoints/best_phase3g_selection.pt"


def sha256_file(path: str | Path) -> str:
    digest=hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda:handle.read(1024*1024), b""):
            digest.update(block)
    return digest.hexdigest()


def checkpoint_state_sha256(path: str | Path) -> str:
    checkpoint=torch.load(path, map_location="cpu", weights_only=False)
    digest=hashlib.sha256()
    for name,value in sorted(checkpoint["model_state_dict"].items()):
        array=value.detach().cpu().contiguous().numpy()
        digest.update(name.encode("utf-8"))
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
        digest.update(array.tobytes())
    return digest.hexdigest()


def read_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path: str | Path, value: dict[str, Any]) -> None:
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def verify_protected_files(
    baseline: dict[str, Any], *, root: Path=ROOT,
) -> list[dict[str, Any]]:
    results=[]
    for expected in baseline["protected_files"]:
        path=root/expected["path"]
        actual={
            "path":expected["path"], "exists":path.is_file(),
            "size":path.stat().st_size if path.is_file() else None,
            "sha256":sha256_file(path) if path.is_file() else None,
        }
        actual["passed"]=(
            actual["exists"] and actual["size"] == int(expected["size"])
            and actual["sha256"].upper() == str(expected["sha256"]).upper()
        )
        results.append(actual)
    if not all(value["passed"] for value in results):
        failed=[value for value in results if not value["passed"]]
        raise RuntimeError(f"A protected formal-model file changed: {failed}")
    return results


def validate_protocol(protocol: dict[str, Any]) -> None:
    if protocol.get("status") != "implementation_ready_not_run":
        raise ValueError("Long-training protocol is not frozen for execution.")
    training=protocol["training"]
    if training["staircase_budgets"] != [30, 40, 50]:
        raise ValueError("Long-training staircase must remain [30, 40, 50].")
    if training["replication_seeds"] != [2026, 2027, 2028]:
        raise ValueError("Replication seeds must remain 2026/2027/2028.")
    sealed=protocol["sealed_path_policy"]
    if sealed["development_paths_one_based"] != list(range(1, 9)):
        raise ValueError("Development is restricted to paths 1-8.")
    if sealed["final_paths_one_based"] != [9, 10]:
        raise ValueError("Final paths must remain 9/10.")
    if sealed["lopo_rerun"]:
        raise ValueError("This protocol explicitly omits a new LOPO run.")


def should_extend_staircase(
    summary: dict[str, Any], budget: int, window: int=3,
) -> bool:
    if summary.get("stop_reason") != "epoch_budget_exhausted":
        return False
    if int(summary.get("last_completed_epoch", -1)) != int(budget):
        return False
    return int(summary["best_epoch"]) >= int(budget)-int(window)+1


def exploration_acceptance(
    summary: dict[str, Any], protocol: dict[str, Any],
) -> dict[str, Any]:
    metrics=summary["development_metrics"]
    threshold=protocol["exploration_acceptance"]
    checks={
        "formal_candidate_eligible":bool(summary.get("formal_candidate_eligible", False)),
        "existing_phase3g_gate_passed":bool(summary["acceptance"]["passed"]),
        "selection_score_at_least_registered":float(metrics["phase3_selection_score"]) >= float(threshold["phase3_selection_score_at_least"]),
        "rebound_at_most_registered":float(metrics["rebound_score_db"]) <= float(threshold["rebound_score_db_at_most"]),
        "worst_path_at_least_registered":float(metrics["worst_path_primary_db"]) >= float(threshold["worst_path_primary_db_at_least"]),
        "sealed_paths_untouched":not bool(summary.get("final_paths_touched", True)),
    }
    return {"passed":all(checks.values()), "checks":checks}


def select_locked_candidate(
    seed_summaries: list[dict[str, Any]], protocol_sha256: str,
) -> dict[str, Any]:
    expected=[2026, 2027, 2028]
    by_seed={int(value["seed"]):value for value in seed_summaries}
    if sorted(by_seed) != expected:
        raise ValueError("Candidate lock requires exactly seeds 2026/2027/2028.")
    checks={
        str(seed):bool(by_seed[seed]["summary"]["acceptance"]["passed"])
        and bool(by_seed[seed]["summary"].get("formal_candidate_eligible", False))
        and not bool(by_seed[seed]["summary"].get("final_paths_touched", True))
        for seed in expected
    }
    if not all(checks.values()):
        raise RuntimeError(f"All three seeds must pass development before locking: {checks}")
    selected=max(
        seed_summaries,
        key=lambda value:float(value["summary"]["development_metrics"]["phase3_selection_score"]),
    )
    checkpoint=Path(selected["summary"]["selected_checkpoint"]).resolve()
    return {
        "lock_version":1, "status":"candidate_locked_final_paths_unopened",
        "protocol_sha256":protocol_sha256, "frozen_budget":int(selected["frozen_budget"]),
        "selection_policy":"highest development D; paths 9/10 unopened",
        "lopo_rerun":False, "final_paths_touched":False,
        "all_seed_development_checks":checks,
        "seed_development_scores":{
            str(value["seed"]):float(value["summary"]["development_metrics"]["phase3_selection_score"])
            for value in seed_summaries
        },
        "selected_seed":int(selected["seed"]),
        "selected_summary":selected["summary_path"],
        "selected_checkpoint":str(checkpoint),
        "selected_checkpoint_sha256":sha256_file(checkpoint),
        "final_evaluation_consumed":False,
    }


def _common_training_args(args: argparse.Namespace, protocol: dict[str, Any]) -> list[str]:
    training=protocol["training"]
    return [
        "--dataset-dir", args.dataset_dir, "--device", args.device,
        "--batch-size", str(training["batch_size"]),
        "--gradient-accumulation", str(training["gradient_accumulation"]),
        "--generator-lr", str(training["generator_lr"]),
        "--dictionary-lr", str(training["dictionary_lr"]),
        "--hidden-size", str(training["hidden_size"]),
        "--latent-size", str(training["latent_size"]),
        "--patience", str(training["patience"]),
        "--stress-cases", str(training["stress_cases"]),
        "--save-every-epoch",
    ]


def _run_training(
    output: Path,
    args: argparse.Namespace,
    protocol: dict[str, Any],
    *,
    stage: str,
    seed: int,
    epochs: int,
    checkpoint: str | Path | None=None,
    resume_checkpoint: str | Path | None=None,
    legacy: bool=False,
) -> dict[str, Any]:
    training=protocol["training"]
    command=[
        sys.executable, "-u", str(ROOT/"train_phase3g.py"),
        "--stage", stage, "--output-dir", str(output), "--seed", str(seed),
        "--epochs", str(epochs), "--samples-per-epoch", str(
            training["warmup_samples_per_epoch"] if stage == "warmup"
            else training["generalize_samples_per_epoch"]
        ),
        *_common_training_args(args, protocol),
    ]
    if checkpoint is not None:
        command.extend(["--checkpoint", str(checkpoint)])
    if resume_checkpoint is not None:
        command.extend(["--resume-checkpoint", str(resume_checkpoint)])
    if legacy:
        command.append("--allow-legacy-resume")
    subprocess.run(command, cwd=ROOT, check=True)
    return read_json(output/"summary.json")


def _prepare_run_root(
    root: Path, protocol_path: Path, baseline_path: Path,
    protected_before: list[dict[str, Any]], mode: str,
) -> None:
    root.mkdir(parents=True, exist_ok=False)
    shutil.copy2(protocol_path, root/"protocol_snapshot.json")
    shutil.copy2(baseline_path, root/"baseline_snapshot.json")
    write_json(root/"run_manifest.json", {
        "mode":mode, "protocol_sha256":sha256_file(protocol_path),
        "baseline_sha256":sha256_file(baseline_path),
        "protected_files_before":protected_before, "final_paths_touched":False,
    })


def run_exploration(
    root: Path, args: argparse.Namespace, protocol: dict[str, Any],
    protocol_sha256: str,
) -> dict[str, Any]:
    training=protocol["training"]
    warm=_run_training(
        root/"P3G-E1", args, protocol, stage="warmup", seed=2027,
        epochs=int(training["warmup_epochs"]),
    )
    if warm.get("stop_reason") == "warmup_guard":
        raise RuntimeError("Seed 2027 warmup failed the frozen P3R regression guard.")
    previous=None; tiers=[]
    for budget in training["staircase_budgets"]:
        output=root/"P3G-E2"/f"ep{budget}"
        if previous is None:
            summary=_run_training(
                output, args, protocol, stage="generalize", seed=2027,
                epochs=int(budget), checkpoint=warm["selected_checkpoint"],
            )
        else:
            summary=_run_training(
                output, args, protocol, stage="generalize", seed=2027,
                epochs=int(budget), resume_checkpoint=previous/"checkpoints/latest.pt",
            )
        tiers.append({
            "budget":int(budget), "summary_path":str(output/"summary.json"),
            "best_epoch":int(summary["best_epoch"]),
            "last_completed_epoch":int(summary["last_completed_epoch"]),
            "stop_reason":summary["stop_reason"],
        })
        previous=output
        if not should_extend_staircase(
            summary, int(budget), int(training["extension_window_epochs"]),
        ):
            break
    acceptance=exploration_acceptance(summary, protocol)
    result={
        "phase":"Phase3G-longtrain-exploration", "seed":2027,
        "protocol_sha256":protocol_sha256,
        "frozen_budget":int(tiers[-1]["budget"]), "tiers":tiers,
        "selected_summary":str(previous/"summary.json"),
        "selected_checkpoint":summary["selected_checkpoint"],
        "selected_checkpoint_sha256":sha256_file(summary["selected_checkpoint"]),
        "development_metrics":summary["development_metrics"],
        "acceptance":acceptance, "final_paths_touched":False,
    }
    write_json(root/"exploration_summary.json", result)
    return result


def run_legacy_control(
    root: Path, args: argparse.Namespace, protocol: dict[str, Any], budget: int,
) -> dict[str, Any]:
    latest_state=checkpoint_state_sha256(LEGACY_LATEST)
    formal_state=checkpoint_state_sha256(FORMAL_BEST)
    if latest_state != formal_state:
        raise RuntimeError(
            "Legacy latest.pt is not model-state-equivalent to the protected epoch-15 best."
        )
    summary=_run_training(
        root/"P3G-E2-legacy-control", args, protocol, stage="generalize",
        seed=2027, epochs=budget, resume_checkpoint=LEGACY_LATEST, legacy=True,
    )
    result={
        "phase":"Phase3G-longtrain-legacy-control", "seed":2027,
        "frozen_budget":budget, "resume_mode":summary["resume_mode"],
        "formal_candidate_eligible":False,
        "legacy_latest_state_sha256":latest_state,
        "formal_best_state_sha256":formal_state,
        "source_model_state_equivalent":True,
        "selected_checkpoint":summary["selected_checkpoint"],
        "development_metrics":summary["development_metrics"],
        "final_paths_touched":False,
    }
    write_json(root/"legacy_control_summary.json", result)
    return result


def run_replications(
    root: Path, args: argparse.Namespace, protocol: dict[str, Any], budget: int,
    protocol_sha256: str,
) -> dict[str, Any]:
    training=protocol["training"]; values=[]
    for seed in training["replication_seeds"]:
        seed_root=root/f"seed_{seed}"
        warm=_run_training(
            seed_root/"P3G-E1", args, protocol, stage="warmup", seed=int(seed),
            epochs=int(training["warmup_epochs"]),
        )
        if warm.get("stop_reason") == "warmup_guard":
            raise RuntimeError(f"Seed {seed} warmup failed the frozen P3R regression guard.")
        summary=_run_training(
            seed_root/"P3G-E2", args, protocol, stage="generalize", seed=int(seed),
            epochs=budget, checkpoint=warm["selected_checkpoint"],
        )
        values.append({
            "seed":int(seed), "frozen_budget":budget,
            "summary_path":str(seed_root/"P3G-E2/summary.json"), "summary":summary,
        })
    lock=select_locked_candidate(values, protocol_sha256)
    write_json(root/"candidate_lock.json", lock)
    result={
        "phase":"Phase3G-longtrain-three-seed-replication",
        "frozen_budget":budget, "all_three_safe":True,
        "candidate_lock":str(root/"candidate_lock.json"),
        "selected_seed":lock["selected_seed"],
        "selected_checkpoint":lock["selected_checkpoint"],
        "lopo_rerun":False, "final_paths_touched":False,
    }
    write_json(root/"replication_summary.json", result)
    return result


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("explore", "legacy-control", "replicate"), required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--protocol", default=str(DEFAULT_PROTOCOL))
    parser.add_argument("--baseline-manifest", default=str(DEFAULT_BASELINE))
    parser.add_argument("--exploration-summary", default=None)
    parser.add_argument("--dataset-dir", default="dataset")
    parser.add_argument("--device", default="auto")
    args=parser.parse_args()
    protocol_path=Path(args.protocol).resolve(); baseline_path=Path(args.baseline_manifest).resolve()
    protocol=read_json(protocol_path); baseline=read_json(baseline_path)
    validate_protocol(protocol)
    protected_before=verify_protected_files(baseline)
    output=Path(args.output_root).resolve()
    _prepare_run_root(output, protocol_path, baseline_path, protected_before, args.mode)
    if args.mode == "explore":
        result=run_exploration(output, args, protocol, sha256_file(protocol_path))
    else:
        if not args.exploration_summary:
            raise ValueError("legacy-control and replicate require --exploration-summary.")
        exploration=read_json(args.exploration_summary)
        if exploration.get("protocol_sha256") != sha256_file(protocol_path):
            raise ValueError("Exploration does not match the current frozen protocol file.")
        if not exploration["acceptance"]["passed"]:
            raise RuntimeError("Exploration did not pass the preregistered balanced gate.")
        budget=int(exploration["frozen_budget"])
        if budget not in protocol["training"]["staircase_budgets"]:
            raise ValueError("Exploration selected a budget outside the frozen staircase.")
        result=(
            run_legacy_control(output, args, protocol, budget)
            if args.mode == "legacy-control" else
            run_replications(output, args, protocol, budget, sha256_file(protocol_path))
        )
    protected_after=verify_protected_files(baseline)
    manifest=read_json(output/"run_manifest.json")
    manifest["protected_files_after"]=protected_after
    manifest["completed"]=True
    manifest["result_summary"]=result
    write_json(output/"run_manifest.json", manifest)


if __name__ == "__main__":
    main()
