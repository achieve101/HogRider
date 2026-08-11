"""Development, continuous-path stress, and compliance checks for Phase 3G."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import numpy as np
import torch

from phase1_data import _read_exact, _resolve_expected_path
from phase1_validation import _aggregate, _load_official_scorer
from phase3_validation import _score_record, phase3_selection_score
from phase3g_data import synthesize_path
from phase3g_model import GenerativeInnovationFIRController
from phase3r_validation import evaluate_development, evaluate_switches, stream_closed_loop
from v6_metrics import SAMPLE_RATE, TOTAL_SAMPLES


def _record_rebound_score(record: dict[str, Any]) -> float:
    """Return the official per-record rebound metric.

    ``_score_record`` preserves the Participant Kit field names.  The shorter
    ``rebound_score_db`` alias only exists after ``_aggregate`` has combined a
    collection of records, so stress comparisons must use the official key.
    """
    return float(record["third_octave_rebound_peak_1000_8000_db"])


def state_dict_sha256(model: torch.nn.Module) -> str:
    digest=hashlib.sha256()
    for name, value in model.state_dict().items():
        array=value.detach().cpu().contiguous().numpy()
        digest.update(name.encode("utf-8")); digest.update(str(array.dtype).encode("ascii"))
        digest.update(np.asarray(array.shape, dtype=np.int64).tobytes()); digest.update(array.tobytes())
    return digest.hexdigest()


def evaluate_phase3g_development(
    model: GenerativeInnovationFIRController,
    dataset_dir: str | Path,
    manifest: dict[str, Any],
    *,
    include_records: bool=True,
) -> dict[str, Any]:
    before=state_dict_sha256(model)
    result=evaluate_development(model, dataset_dir, manifest, include_records=include_records)
    after=state_dict_sha256(model)
    result["state_dict_sha256_before"]=before
    result["state_dict_sha256_after"]=after
    result["state_dict_immutable"]=before == after
    result["trainable_parameter_count"]=sum(value.numel() for value in model.parameters() if value.requires_grad)
    return result


def _stress_signals(
    dataset_dir: Path,
    development_manifest: dict[str, Any],
    case: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    source_key="vehicle" if int(case["case"]) % 2 == 0 else "restaurant"
    raw=dataset_dir / "NOISE" / development_manifest["sources"][source_key]
    start=int(round(float(development_manifest["start_seconds"])*SAMPLE_RATE))
    reference=_read_exact(raw, start, TOTAL_SAMPLES).astype(np.float64)
    first=int(case["first_path_zero_based"]); second=int(case["second_path_zero_based"])
    expected_dir=dataset_dir / "EXPECTED_NOISE"
    d_first=_read_exact(_resolve_expected_path(expected_dir, raw, first), start, TOTAL_SAMPLES).astype(np.float64)
    d_second=_read_exact(_resolve_expected_path(expected_dir, raw, second), start, TOTAL_SAMPLES).astype(np.float64)
    all_paths=np.load(dataset_dir / "sh.npy", allow_pickle=False, mmap_mode="r").T
    path, _=synthesize_path(all_paths[first], all_paths[second], mode=case["mode"], amount=float(case["amount"]))
    amount=float(case["amount"])
    disturbance=((1-amount)*d_first+amount*d_second if case["mode"] == "interpolate"
                 else d_first+amount*(d_first-d_second))
    return reference, path.astype(np.float64), disturbance.astype(np.float64)


def evaluate_continuous_path_stress(
    candidate: GenerativeInnovationFIRController,
    baseline: torch.nn.Module,
    dataset_dir: str | Path,
    development_manifest: dict[str, Any],
    synthesis_manifest: dict[str, Any],
    *,
    max_cases: int | None=None,
) -> dict[str, Any]:
    scorer=_load_official_scorer(); root=Path(dataset_dir); records=[]
    cases=synthesis_manifest["stress_cases"]
    if max_cases is not None:
        cases=cases[:max_cases]
    for case in cases:
        reference, path, disturbance=_stress_signals(root, development_manifest, case)
        output, residual, _, rtf=stream_closed_loop(candidate, reference, disturbance, path)
        base_output, base_residual, _, base_rtf=stream_closed_loop(baseline, reference, disturbance, path)
        candidate_record=_score_record(
            scorer, f"phase3g_stress_{case['case']}", int(case["first_path_zero_based"]),
            torch.from_numpy(disturbance).unsqueeze(0), torch.from_numpy(residual).unsqueeze(0),
            torch.from_numpy(output).unsqueeze(0),
        )
        baseline_record=_score_record(
            scorer, f"phase3g_stress_{case['case']}_baseline", int(case["first_path_zero_based"]),
            torch.from_numpy(disturbance).unsqueeze(0), torch.from_numpy(base_residual).unsqueeze(0),
            torch.from_numpy(base_output).unsqueeze(0),
        )
        candidate_rebound=_record_rebound_score(candidate_record)
        baseline_rebound=_record_rebound_score(baseline_record)
        candidate_record.update({
            "case": case, "baseline_primary_score_db": baseline_record["primary_score_db"],
            "baseline_rebound_score_db": baseline_rebound,
            "primary_gain_db": candidate_record["primary_score_db"]-baseline_record["primary_score_db"],
            "rebound_change_db": candidate_rebound-baseline_rebound,
            "cpu_real_time_factor": rtf, "baseline_cpu_real_time_factor": base_rtf,
        })
        records.append(candidate_record)
    gains=np.asarray([value["primary_gain_db"] for value in records], dtype=np.float64)
    aggregate=_aggregate(records)
    aggregate["phase3_selection_score"]=phase3_selection_score(aggregate)
    aggregate["median_primary_gain_db"]=float(np.median(gains)) if gains.size else -float("inf")
    aggregate["non_degrading_fraction"]=float(np.mean(gains >= 0)) if gains.size else 0.0
    aggregate["maximum_rebound_change_db"]=max((value["rebound_change_db"] for value in records), default=float("inf"))
    aggregate["cpu_real_time_factor"]=float(np.mean([value["cpu_real_time_factor"] for value in records])) if records else float("inf")
    aggregate["records"]=records
    return aggregate


def phase3g_gate(
    p1_baseline: dict[str, Any],
    development: dict[str, Any],
    switches: dict[str, Any],
    stress: dict[str, Any] | None,
) -> dict[str, Any]:
    checks={
        "primary_drop_at_most_0_5_db": development["primary_score_db"] >= p1_baseline["primary_score_db"]-0.5,
        "rebound_increase_at_most_0_3_db": development["rebound_score_db"] <= p1_baseline["rebound_score_db"]+0.3,
        "worst_path_at_least_1_5_db": development["worst_path_primary_db"] >= 1.5,
        "first_window_drop_at_most_0_25_db": development["first_window_primary_db"] >= p1_baseline["first_window_primary_db"]-0.25,
        "switch_recovery_at_most_100_ms": switches["all_switches_recover_within_100_ms"],
        "controller_peak_below_0_98": development["controller_peak_abs"] < 0.98,
        "cpu_real_time_factor_at_most_1": development["cpu_real_time_factor"] <= 1.0,
        "finite": development["finite"], "state_dict_immutable": development["state_dict_immutable"],
    }
    if stress is not None:
        checks.update({
            "stress_median_gain_at_least_0_75_db": stress["median_primary_gain_db"] >= 0.75,
            "stress_75_percent_non_degrading": stress["non_degrading_fraction"] >= 0.75,
        })
    return {"passed": all(checks.values()), "checks": checks}
