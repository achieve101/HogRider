"""One-shot sealed-path evaluation for a locked Phase-3G long-train candidate."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import torch

from phase1_validation import _aggregate
from phase3_validation import build_phase3_manifests, phase3_final_acceptance
from phase3g_validation import evaluate_phase3g_development
from run_phase3g_longtrain import (
    DEFAULT_BASELINE,
    DEFAULT_PROTOCOL,
    read_json,
    sha256_file,
    validate_protocol,
    verify_protected_files,
    write_json,
)
from train_phase3g import load_phase3g


ROOT=Path(__file__).resolve().parent


def strengthened_final_acceptance(
    original: dict[str, Any],
    candidate_development: dict[str, Any],
    candidate_final: dict[str, Any],
    protocol: dict[str, Any],
) -> dict[str, Any]:
    primary=(8.0*float(candidate_development["primary_score_db"])+2.0*float(candidate_final["primary_score_db"]))/10.0
    rebound=(8.0*float(candidate_development["rebound_score_db"])+2.0*float(candidate_final["rebound_score_db"]))/10.0
    composite=0.7*primary-0.3*rebound
    worst=min(
        float(candidate_development["worst_path_primary_db"]),
        float(candidate_final["worst_path_primary_db"]),
    )
    thresholds=protocol["final_promotion"]
    checks={
        "existing_phase3_final_acceptance":bool(original["passed"]),
        "ten_path_primary_not_below_formal":primary >= float(thresholds["ten_path_primary_score_db_at_least"]),
        "ten_path_composite_not_below_formal":composite >= float(thresholds["ten_path_composite_at_least"]),
        "ten_path_rebound_within_tolerance":rebound <= float(thresholds["ten_path_rebound_score_db_at_most"]),
        "ten_path_worst_path_within_tolerance":worst >= float(thresholds["ten_path_worst_path_primary_db_at_least"]),
    }
    return {
        "passed":all(checks.values()), "checks":checks,
        "candidate_ten_path":{
            "primary_score_db":primary, "rebound_score_db":rebound,
            "composite":composite, "worst_path_primary_db":worst,
        },
        "thresholds":thresholds,
    }


def validate_candidate_lock(
    lock: dict[str, Any], protocol_path: Path,
) -> tuple[Path, Path]:
    if lock.get("status") != "candidate_locked_final_paths_unopened":
        raise ValueError("Candidate lock is not in the unopened state.")
    if lock.get("final_evaluation_consumed") is not False or lock.get("final_paths_touched") is not False:
        raise ValueError("Candidate lock already consumed final-path access.")
    if lock.get("lopo_rerun") is not False:
        raise ValueError("Long-training protocol must explicitly record lopo_rerun=false.")
    if lock.get("protocol_sha256") != sha256_file(protocol_path):
        raise ValueError("Candidate lock does not match the frozen protocol.")
    checkpoint=Path(lock["selected_checkpoint"]).resolve()
    summary=Path(lock["selected_summary"]).resolve()
    if not checkpoint.is_file() or sha256_file(checkpoint) != lock["selected_checkpoint_sha256"]:
        raise ValueError("Locked checkpoint is missing or changed.")
    selected=read_json(summary)
    if selected["selected_checkpoint"] != str(checkpoint):
        selected_checkpoint=Path(selected["selected_checkpoint"]).resolve()
        if selected_checkpoint != checkpoint:
            raise ValueError("Locked summary points to a different checkpoint.")
    if not selected["acceptance"]["passed"] or selected.get("final_paths_touched", True):
        raise ValueError("Locked seed is not development-safe and sealed.")
    return checkpoint,summary


def consume_final_evaluation_once(
    receipt_path: Path,
    *,
    lock_path: Path,
    lock: dict[str, Any],
) -> dict[str, Any]:
    receipt={
        "receipt_version":1, "status":"sealed_paths_access_committed",
        "candidate_lock":str(lock_path), "candidate_lock_sha256":sha256_file(lock_path),
        "selected_seed":lock["selected_seed"],
        "selected_checkpoint":lock["selected_checkpoint"],
        "selected_checkpoint_sha256":lock["selected_checkpoint_sha256"],
        "final_paths_one_based":[9,10],
        "replacement_candidate_after_failure_forbidden":True,
    }
    flags=os.O_WRONLY|os.O_CREAT|os.O_EXCL
    descriptor=os.open(receipt_path, flags)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(receipt, handle, ensure_ascii=False, indent=2)
    except BaseException:
        if receipt_path.exists():
            receipt_path.unlink()
        raise
    return receipt


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-lock", required=True)
    parser.add_argument("--protocol", default=str(DEFAULT_PROTOCOL))
    parser.add_argument("--baseline-manifest", default=str(DEFAULT_BASELINE))
    parser.add_argument("--dataset-dir", default="dataset")
    parser.add_argument("--output", required=True)
    args=parser.parse_args()
    lock_path=Path(args.candidate_lock).resolve()
    protocol_path=Path(args.protocol).resolve()
    baseline_path=Path(args.baseline_manifest).resolve()
    output=Path(args.output).resolve()
    if output.exists():
        raise FileExistsError(f"Final evaluation output already exists: {output}")
    protocol=read_json(protocol_path); validate_protocol(protocol)
    baseline=read_json(baseline_path); protected_before=verify_protected_files(baseline)
    lock=read_json(lock_path)
    checkpoint,selected_summary=validate_candidate_lock(lock, protocol_path)
    receipt_path=lock_path.with_name("final_evaluation_receipt.json")
    if receipt_path.exists():
        raise FileExistsError(
            "This candidate lock has already consumed its one sealed-path evaluation."
        )

    # Development is re-evaluated before committing final-path access.  No path
    # 9/10 data is read until the exclusive receipt has been created.
    model=load_phase3g(checkpoint, torch.device("cpu")).eval()
    manifests=build_phase3_manifests(args.dataset_dir)
    development=evaluate_phase3g_development(model, args.dataset_dir, manifests["development"])
    locked_summary=read_json(selected_summary)
    if abs(
        float(development["phase3_selection_score"])
        - float(locked_summary["development_metrics"]["phase3_selection_score"])
    ) > 1e-8:
        raise RuntimeError("Locked development score is not reproducible; final paths remain sealed.")
    receipt=consume_final_evaluation_once(
        receipt_path, lock_path=lock_path, lock=lock,
    )
    final=evaluate_phase3g_development(model, args.dataset_dir, manifests["final"])

    p1=read_json(ROOT/"runs/phase1_suite_seed2026/P1-E2/summary.json")
    records=p1["final_metrics"]["records"]
    baseline_dev=_aggregate([value for value in records if value["path_number"] <= 8])
    baseline_final=_aggregate([value for value in records if value["path_number"] >= 9])
    rtf=max(float(development["cpu_real_time_factor"]), float(final["cpu_real_time_factor"]))
    original=phase3_final_acceptance(
        baseline_dev, baseline_final, development, final, rtf,
    )
    strengthened=strengthened_final_acceptance(original, development, final, protocol)
    summary={
        "phase":"Phase3G-longtrain-final", "candidate_lock":str(lock_path),
        "candidate_lock_sha256":receipt["candidate_lock_sha256"],
        "selected_seed":lock["selected_seed"], "checkpoint":str(checkpoint),
        "checkpoint_sha256":lock["selected_checkpoint_sha256"],
        "selection_policy":"candidate locked by development D before paths 9/10",
        "lopo_rerun":False, "final_paths_touched":True,
        "development":development, "final_unseen":final,
        "existing_final_acceptance":original,
        "strengthened_final_acceptance":strengthened,
        "formal_upgrade_recommended":strengthened["passed"],
        "automatic_export_or_overwrite":False,
        "protected_files_before":protected_before,
        "protected_files_after":verify_protected_files(baseline),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    write_json(output, summary)


if __name__ == "__main__":
    main()
