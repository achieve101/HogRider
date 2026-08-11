"""Fixed development and switch validation for Phase-3R innovation routing."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from phase1_data import iter_validation_examples
from phase1_validation import _aggregate, _load_official_scorer
from phase3_validation import _score_record, phase3_selection_score
from phase3r_model import InnovationRoutedFIRController
from v6_metrics import TOTAL_SAMPLES


PATH_SWITCH_PAIRS = ((0, 6), (6, 0), (3, 5), (5, 7))


def stream_closed_loop(
    controller: InnovationRoutedFIRController,
    reference: np.ndarray,
    disturbance: np.ndarray,
    path_a: np.ndarray,
    *,
    path_b: np.ndarray | None = None,
    switch_sample: int = TOTAL_SAMPLES,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]], float]:
    """Execute the public e[t-1] API with an exact causal physical path."""
    controller.reset()
    output = np.zeros(reference.size, dtype=np.float64)
    residual = np.zeros(reference.size, dtype=np.float64)
    path_length = path_a.size
    ring = np.zeros(2 * path_length, dtype=np.float64)
    pointer = 0
    previous_error = 0.0
    started = time.perf_counter()
    for index, (sample_x, sample_d) in enumerate(zip(reference, disturbance)):
        sample_y = controller.process_sample(float(sample_x), previous_error)
        pointer = (pointer - 1) % path_length
        ring[pointer] = sample_y
        ring[pointer + path_length] = sample_y
        physical_path = path_a if index < switch_sample or path_b is None else path_b
        anti = float(np.dot(physical_path, ring[pointer:pointer + path_length]))
        previous_error = float(sample_d - anti)
        output[index], residual[index] = sample_y, previous_error
    elapsed = time.perf_counter() - started
    return output, residual, controller.route_diagnostics(), elapsed / (reference.size / 48_000.0)


def _route_report(
    trace: list[dict[str, Any]], first: int, second: int | None = None,
    switch_sample: int = TOTAL_SAMPLES,
) -> dict[str, Any]:
    eligible = [item for item in trace if item["completed_samples"] >= 24_000]
    def label(item: dict[str, Any]) -> int:
        return first if second is None or item["completed_samples"] < switch_sample else second
    accuracy = float(np.mean([item["winner_zero_based"] == label(item) for item in eligible])) if eligible else 0.0
    first_identification = next((item["completed_samples"] for item in trace
                                 if item["winner_zero_based"] == first), None)
    report: dict[str, Any] = {
        "route_accuracy_after_initialization": accuracy,
        "initial_identification_ms": None if first_identification is None else 1000.0 * first_identification / 48_000.0,
        "trace": trace,
    }
    if second is not None:
        post = [item for item in trace if item["completed_samples"] >= switch_sample]
        first_correct = next((index for index, item in enumerate(post) if item["winner_zero_based"] == second), None)
        triple = next((index for index in range(max(0, len(post) - 2))
                       if all(post[index + offset]["winner_zero_based"] == second for offset in range(3))), None)
        alpha80 = next((index for index, item in enumerate(post) if item["alpha"][second] >= 0.8), None)
        def recovery(index: int | None) -> float | None:
            if index is None:
                return None
            return 1000.0 * max(0, post[index]["completed_samples"] - switch_sample) / 48_000.0
        report.update({
            "first_correct_recovery_ms": recovery(first_correct),
            "three_correct_recovery_ms": recovery(triple),
            "alpha_0_8_recovery_ms": recovery(alpha80),
        })
    return report


def evaluate_development(
    controller: InnovationRoutedFIRController,
    dataset_dir: str | Path,
    manifest: dict[str, Any],
    *, include_records: bool = True,
) -> dict[str, Any]:
    scorer = _load_official_scorer()
    records, route_reports, rtfs = [], [], []
    for scene, path_index, reference_t, path_t, disturbance_t in iter_validation_examples(dataset_dir, manifest):
        reference = reference_t.numpy().astype(np.float64)
        disturbance = disturbance_t.numpy().astype(np.float64)
        path = path_t.numpy().astype(np.float64)
        output, residual, trace, rtf = stream_closed_loop(controller, reference, disturbance, path)
        record = _score_record(
            scorer, scene, path_index, torch.from_numpy(disturbance).unsqueeze(0),
            torch.from_numpy(residual).unsqueeze(0), torch.from_numpy(output).unsqueeze(0),
        )
        route = _route_report(trace, path_index)
        record.update({key: value for key, value in route.items() if key != "trace"})
        record["innovation_trace"] = trace
        records.append(record); route_reports.append(route); rtfs.append(rtf)
    result = _aggregate(records)
    result["phase3_selection_score"] = phase3_selection_score(result)
    result["route_accuracy_after_initialization"] = float(np.mean(
        [item["route_accuracy_after_initialization"] for item in route_reports]
    ))
    result["cpu_real_time_factor"] = float(np.mean(rtfs))
    result["finite"] = bool(all(np.isfinite(item["full_controller_peak_abs"]) for item in records))
    if include_records:
        result["records"] = records
    return result


def evaluate_switches(
    controller: InnovationRoutedFIRController,
    dataset_dir: str | Path,
    manifest: dict[str, Any],
    *, switch_sample: int = 96_000,
) -> dict[str, Any]:
    scorer = _load_official_scorer()
    examples = {(scene, path): (x.numpy().astype(np.float64), s.numpy().astype(np.float64), d.numpy().astype(np.float64))
                for scene, path, x, s, d in iter_validation_examples(dataset_dir, manifest)
                if scene in {"vehicle_continuous", "restaurant_continuous"}}
    records, recovery_values = [], []
    for first, second in PATH_SWITCH_PAIRS:
        for scene in ("vehicle_continuous", "restaurant_continuous"):
            reference, path_a, disturbance_a = examples[(scene, first)]
            _, path_b, disturbance_b = examples[(scene, second)]
            disturbance = np.concatenate((disturbance_a[:switch_sample], disturbance_b[switch_sample:]))
            output, residual, trace, rtf = stream_closed_loop(
                controller, reference, disturbance, path_a, path_b=path_b, switch_sample=switch_sample,
            )
            route = _route_report(trace, first, second, switch_sample)
            value = route["three_correct_recovery_ms"]
            recovery_values.append(float("inf") if value is None else value)
            record = _score_record(
                scorer, f"{scene}_path_{first + 1}_to_{second + 1}", second,
                torch.from_numpy(disturbance).unsqueeze(0), torch.from_numpy(residual).unsqueeze(0),
                torch.from_numpy(output).unsqueeze(0),
            )
            record.update(route)
            record["cpu_real_time_factor"] = rtf
            record["post_switch_first_window_primary_db"] = record["window_results"][3]["primary_score_db"]
            records.append(record)
    aggregate = _aggregate(records)
    aggregate["maximum_three_correct_recovery_ms"] = float(max(recovery_values))
    aggregate["all_switches_recover_within_100_ms"] = bool(max(recovery_values) <= 100.0)
    aggregate["post_switch_first_window_primary_db"] = float(np.mean(
        [item["post_switch_first_window_primary_db"] for item in records]
    ))
    aggregate["records"] = records
    return aggregate


def development_gate(baseline: dict[str, Any], candidate: dict[str, Any], switches: dict[str, Any]) -> dict[str, Any]:
    checks = {
        "route_accuracy_at_least_95_percent": candidate["route_accuracy_after_initialization"] >= 0.95,
        "switch_recovery_at_most_100_ms": switches["all_switches_recover_within_100_ms"],
        "primary_drop_at_most_0_5_db": candidate["primary_score_db"] >= baseline["primary_score_db"] - 0.5,
        "rebound_increase_at_most_0_3_db": candidate["rebound_score_db"] <= baseline["rebound_score_db"] + 0.3,
        "worst_path_at_least_1_5_db": candidate["worst_path_primary_db"] >= 1.5,
        "first_window_drop_at_most_0_25_db": candidate["first_window_primary_db"] >= baseline["first_window_primary_db"] - 0.25,
        "controller_peak_below_0_98": candidate["controller_peak_abs"] < 0.98,
        "finite": candidate["finite"],
        "cpu_real_time_factor_at_most_1": candidate["cpu_real_time_factor"] <= 1.0,
    }
    return {"passed": all(checks.values()), "checks": checks}
