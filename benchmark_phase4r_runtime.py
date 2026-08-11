"""Reproducible Phase-4R runtime and equivalence benchmark.

Official RTF measurements call the unmodified Participant Kit evaluator.  A
separate instrumented pass measures sample-event tails, so timer overhead is
never mixed into the acceptance RTF.
"""

from __future__ import annotations

import argparse
import cProfile
import hashlib
import importlib
import json
import os
import platform
import pstats
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch


ROOT = Path(__file__).resolve().parent
KIT_DIR = ROOT / "DEEPANC_PARTICIPANT_KIT"
if str(KIT_DIR) not in sys.path:
    sys.path.insert(0, str(KIT_DIR))

from participant_api import load_submission  # noqa: E402
from run_public_demo import (  # noqa: E402
    DEFAULT_DATA_DIR,
    _SecondaryPath,
    _load_scene,
    _run_closed_loop,
    evaluate,
)


ACOUSTIC_KEYS = (
    "primary_score_db",
    "third_octave_rebound_peak_1000_8000_db",
    "controller_peak_abs",
    "full_run_controller_rms",
)


def _processor_name() -> str:
    if sys.platform == "win32":
        try:
            import winreg

            key_path = r"HARDWARE\DESCRIPTION\System\CentralProcessor\0"
            with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, key_path) as key:
                return str(winreg.QueryValueEx(key, "ProcessorNameString")[0]).strip()
        except OSError:
            pass
    return os.environ.get("PROCESSOR_IDENTIFIER", platform.processor())


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file_object:
        for block in iter(lambda: file_object.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _state_dict_sha256(model: Any) -> str:
    module = model.model
    digest = hashlib.sha256()
    for name, value in module.state_dict().items():
        array = value.detach().cpu().contiguous().numpy()
        digest.update(name.encode("utf-8"))
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
        digest.update(array.tobytes())
    return digest.hexdigest()


def _weights_path(entry_point: str) -> Path:
    module_name = entry_point.partition(":")[0]
    module = importlib.import_module(module_name)
    return Path(module.__file__).resolve().with_name("weights.pt")


def _distribution(values: list[float]) -> dict[str, float | int]:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        return {"count": 0, "p50": 0.0, "p95": 0.0, "p99": 0.0, "max": 0.0}
    return {
        "count": int(array.size),
        "p50": float(np.percentile(array, 50)),
        "p95": float(np.percentile(array, 95)),
        "p99": float(np.percentile(array, 99)),
        "max": float(np.max(array)),
    }


def _official_runs(entry_point: str, warmups: int, runs: int) -> list[dict[str, Any]]:
    for _ in range(warmups):
        evaluate(entry_point, "cpu", 0.0, 100.0, 999.0)
    return [evaluate(entry_point, "cpu", 0.0, 100.0, 999.0) for _ in range(runs)]


def _event_profile(
    entry_point: str,
    reference: np.ndarray,
    disturbance: np.ndarray,
    impulse_response: np.ndarray,
) -> dict[str, Any]:
    model = load_submission(entry_point, "cpu")
    block_size = int(model.model.block_size)
    secondary = _SecondaryPath(impulse_response)
    previous_error = 0.0
    ordinary_ms: list[float] = []
    boundary_ms: list[float] = []
    model.reset()
    with torch.inference_mode():
        for index, reference_value in enumerate(reference):
            started = time.perf_counter_ns()
            output = float(model.process_sample(float(reference_value), float(previous_error)))
            elapsed_ms = (time.perf_counter_ns() - started) / 1_000_000.0
            if index > 0 and index % block_size == 0:
                boundary_ms.append(elapsed_ms)
            else:
                ordinary_ms.append(elapsed_ms)
            previous_error = float(disturbance[index] - secondary.process(output))
    return {
        "block_size": block_size,
        "ordinary_event_ms": _distribution(ordinary_ms),
        "block_boundary_event_ms": _distribution(boundary_ms),
    }


def _profile_components(
    entry_point: str,
    reference: np.ndarray,
    disturbance: np.ndarray,
    impulse_response: np.ndarray,
) -> dict[str, Any]:
    model = load_submission(entry_point, "cpu")
    profiler = cProfile.Profile()
    profiler.enable()
    _, _, runtime_seconds = _run_closed_loop(
        model, reference, disturbance, impulse_response, 100.0, 999.0,
    )
    profiler.disable()
    stats = pstats.Stats(profiler)

    def collect(function_name: str, filename_suffix: str | None = None) -> dict[str, float | int]:
        calls = 0
        internal = 0.0
        cumulative = 0.0
        for (filename, _line, name), value in stats.stats.items():
            if name != function_name:
                continue
            normalized = filename.replace("\\", "/")
            if filename_suffix is not None and not normalized.endswith(filename_suffix):
                continue
            _primitive_calls, total_calls, total_time, cumulative_time, _callers = value
            calls += int(total_calls)
            internal += float(total_time)
            cumulative += float(cumulative_time)
        return {
            "calls": calls,
            "internal_seconds": internal,
            "cumulative_seconds": cumulative,
        }

    package_name = entry_point.partition(":")[0].rpartition(".")[0]
    package_path = package_name.replace(".", "/")
    return {
        "profiled_closed_loop_seconds": float(runtime_seconds),
        "note": "Profiling overhead is excluded from official RTF acceptance.",
        "submission_wrapper": collect("process_sample", f"{package_path}/runtime.py"),
        "per_sample_fir_and_state": collect("process_sample", f"{package_path}/model.py"),
        "feedback_append": collect("_append_completed_feedback", f"{package_path}/model.py"),
        "candidate_path_convolution_and_block_finalize": collect(
            "_finalize_feedback_block", f"{package_path}/model.py",
        ),
        "innovation_fft_scoring_gru_and_fir_synthesis": collect(
            "_update_route_and_generator", f"{package_path}/model.py",
        ),
        "official_secondary_path": collect("process", "DEEPANC_PARTICIPANT_KIT/run_public_demo.py"),
        "official_closed_loop": collect("_run_closed_loop", "DEEPANC_PARTICIPANT_KIT/run_public_demo.py"),
    }


def _equivalence(
    baseline_entry: str,
    candidate_entry: str,
    reference: np.ndarray,
    disturbance: np.ndarray,
    impulse_response: np.ndarray,
) -> dict[str, Any]:
    baseline = load_submission(baseline_entry, "cpu")
    candidate = load_submission(candidate_entry, "cpu")
    baseline_before = _state_dict_sha256(baseline)
    candidate_before = _state_dict_sha256(candidate)
    baseline_output, baseline_residual, _ = _run_closed_loop(
        baseline, reference, disturbance, impulse_response, 100.0, 999.0,
    )
    candidate_output, candidate_residual, _ = _run_closed_loop(
        candidate, reference, disturbance, impulse_response, 100.0, 999.0,
    )
    candidate_repeat, candidate_repeat_residual, _ = _run_closed_loop(
        candidate, reference, disturbance, impulse_response, 100.0, 999.0,
    )
    baseline_after = _state_dict_sha256(baseline)
    candidate_after = _state_dict_sha256(candidate)
    return {
        "processed_samples": int(reference.size),
        "output_array_equal": bool(np.array_equal(baseline_output, candidate_output)),
        "residual_array_equal": bool(np.array_equal(baseline_residual, candidate_residual)),
        "candidate_reset_output_array_equal": bool(np.array_equal(candidate_output, candidate_repeat)),
        "candidate_reset_residual_array_equal": bool(
            np.array_equal(candidate_residual, candidate_repeat_residual)
        ),
        "maximum_output_absolute_error": float(np.max(np.abs(baseline_output - candidate_output))),
        "maximum_residual_absolute_error": float(np.max(np.abs(baseline_residual - candidate_residual))),
        "baseline_state_dict_sha256_before": baseline_before,
        "baseline_state_dict_sha256_after": baseline_after,
        "candidate_state_dict_sha256_before": candidate_before,
        "candidate_state_dict_sha256_after": candidate_after,
        "state_dicts_equal": baseline_before == candidate_before,
        "state_dicts_immutable": baseline_before == baseline_after and candidate_before == candidate_after,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--baseline-entry",
        default="phase3g_submission_final_seed2027_v2.submission:create_model",
    )
    parser.add_argument(
        "--candidate-entry",
        default="phase3g_submission_final_seed2027_v3.submission:create_model",
    )
    parser.add_argument("--baseline-runs", type=int, default=3)
    parser.add_argument("--candidate-runs", type=int, default=10)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument(
        "--report", type=Path, default=ROOT / "artifacts" / "phase4r_runtime_report.json",
    )
    args = parser.parse_args()
    if min(args.baseline_runs, args.candidate_runs, args.warmups) < 0:
        raise ValueError("Run counts must be non-negative.")
    if args.baseline_runs == 0 or args.candidate_runs == 0:
        raise ValueError("Measured run counts must be positive.")

    reference, disturbance, impulse_response = _load_scene(DEFAULT_DATA_DIR, 0.0)
    baseline_results = _official_runs(args.baseline_entry, args.warmups, args.baseline_runs)
    candidate_results = _official_runs(args.candidate_entry, args.warmups, args.candidate_runs)
    baseline_rtfs = [float(item["real_time_factor"]) for item in baseline_results]
    candidate_rtfs = [float(item["real_time_factor"]) for item in candidate_results]
    equivalence = _equivalence(
        args.baseline_entry, args.candidate_entry, reference, disturbance, impulse_response,
    )
    baseline_events = _event_profile(
        args.baseline_entry, reference, disturbance, impulse_response,
    )
    candidate_events = _event_profile(
        args.candidate_entry, reference, disturbance, impulse_response,
    )
    baseline_weights = _weights_path(args.baseline_entry)
    candidate_weights = _weights_path(args.candidate_entry)
    acoustic_differences = {
        key: float(candidate_results[0][key] - baseline_results[0][key])
        for key in ACOUSTIC_KEYS
    }
    acceptance = {
        "all_candidate_rtfs_at_most_0_8": max(candidate_rtfs) <= 0.8,
        "block_boundary_p99_at_most_5_ms": (
            candidate_events["block_boundary_event_ms"]["p99"] <= 5.0
        ),
        "block_boundary_p99_not_above_baseline": (
            candidate_events["block_boundary_event_ms"]["p99"]
            <= baseline_events["block_boundary_event_ms"]["p99"]
        ),
        "outputs_and_residuals_bit_exact": (
            equivalence["output_array_equal"] and equivalence["residual_array_equal"]
        ),
        "reset_bit_exact": (
            equivalence["candidate_reset_output_array_equal"]
            and equivalence["candidate_reset_residual_array_equal"]
        ),
        "state_dict_equal_and_immutable": (
            equivalence["state_dicts_equal"] and equivalence["state_dicts_immutable"]
        ),
        "weights_sha256_unchanged": _sha256_file(baseline_weights) == _sha256_file(candidate_weights),
        "official_acoustic_scalars_unchanged": all(value == 0.0 for value in acoustic_differences.values()),
    }
    report = {
        "phase": "4R",
        "experiment": "E09-A",
        "status": "runtime_hardening_passed" if all(acceptance.values()) else "runtime_hardening_failed",
        "environment": {
            "platform": platform.platform(),
            "processor": _processor_name(),
            "python": platform.python_version(),
            "numpy": np.__version__,
            "torch": torch.__version__,
            "torch_threads": torch.get_num_threads(),
            "torch_interop_threads": torch.get_num_interop_threads(),
        },
        "baseline_entry_point": args.baseline_entry,
        "candidate_entry_point": args.candidate_entry,
        "official_rtf": {
            "warmups": args.warmups,
            "baseline": {"values": baseline_rtfs, **_distribution(baseline_rtfs)},
            "candidate": {"values": candidate_rtfs, **_distribution(candidate_rtfs)},
        },
        "event_timing": {
            "unit": "milliseconds",
            "instrumented_separately_from_official_rtf": True,
            "baseline": baseline_events,
            "candidate": candidate_events,
        },
        "component_profile": _profile_components(
            args.candidate_entry, reference, disturbance, impulse_response,
        ),
        "equivalence": equivalence,
        "weights": {
            "baseline_path": baseline_weights.name,
            "candidate_path": candidate_weights.name,
            "baseline_sha256": _sha256_file(baseline_weights),
            "candidate_sha256": _sha256_file(candidate_weights),
        },
        "official_acoustic_metrics": {
            "baseline": {key: float(baseline_results[0][key]) for key in ACOUSTIC_KEYS},
            "candidate": {key: float(candidate_results[0][key]) for key in ACOUSTIC_KEYS},
            "differences": acoustic_differences,
        },
        "complexity": {
            "parameter_count": int(candidate_results[0]["parameter_count"]),
            "peak_macs_in_one_sample_event": int(
                candidate_results[0]["peak_macs_in_one_sample_event"]
            ),
        },
        "acceptance": {**acceptance, "passed": all(acceptance.values())},
        "stopping_decision": (
            "E09-A passed; stop without E09-B."
            if all(acceptance.values())
            else "E09-A did not pass; E09-B may be evaluated."
        ),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
    )
    print(json.dumps(report["acceptance"], ensure_ascii=False, indent=2))
    print(f"Report: {args.report.resolve()}")
    if not report["acceptance"]["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
