"""Wait for Phase 3G exploration and run the preregistered follow-up stages.

This supervisor deliberately does not invoke the path 9/10 final evaluator.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path


def read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def process_is_running(pid: int) -> bool:
    completed = subprocess.run(
        ["tasklist", "/FI", f"PID eq {pid}", "/NH"],
        check=False,
        capture_output=True,
        text=True,
    )
    return str(pid) in completed.stdout


def wait_for_exploration(root: Path, pid: int, poll_seconds: int) -> Path:
    manifest_path = root / "run_manifest.json"
    summary_path = root / "exploration_summary.json"
    while True:
        if manifest_path.exists():
            manifest = read_json(manifest_path)
            if manifest.get("completed") is True:
                if not summary_path.exists():
                    raise RuntimeError("Exploration completed without exploration_summary.json")
                return summary_path
        if not process_is_running(pid):
            raise RuntimeError(
                "Exploration process exited before producing a completed run manifest."
            )
        time.sleep(poll_seconds)


def run_stage(script: Path, mode: str, summary: Path, output: Path, device: str) -> None:
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite follow-up directory: {output}")
    command = [
        sys.executable,
        "-u",
        str(script),
        "--mode",
        mode,
        "--exploration-summary",
        str(summary),
        "--output-root",
        str(output),
        "--device",
        device,
    ]
    print(f"[supervisor] starting {mode}: {' '.join(command)}", flush=True)
    subprocess.run(command, check=True, cwd=script.parent)
    print(f"[supervisor] completed {mode}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exploration-root", required=True)
    parser.add_argument("--exploration-pid", required=True, type=int)
    parser.add_argument("--legacy-output", required=True)
    parser.add_argument("--replication-output", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--poll-seconds", type=int, default=30)
    args = parser.parse_args()

    project = Path(__file__).resolve().parent
    exploration_root = Path(args.exploration_root).resolve()
    print(
        f"[supervisor] waiting for exploration pid={args.exploration_pid} "
        f"root={exploration_root}",
        flush=True,
    )
    summary_path = wait_for_exploration(
        exploration_root, args.exploration_pid, args.poll_seconds
    )
    summary = read_json(summary_path)
    acceptance = summary.get("acceptance", {})
    if acceptance.get("passed") is not True:
        print("[supervisor] exploration failed; follow-up stages will not run.", flush=True)
        return

    print(
        f"[supervisor] exploration passed; frozen_budget={summary['frozen_budget']}",
        flush=True,
    )
    runner = project / "run_phase3g_longtrain.py"
    run_stage(
        runner,
        "legacy-control",
        summary_path,
        Path(args.legacy_output).resolve(),
        args.device,
    )
    run_stage(
        runner,
        "replicate",
        summary_path,
        Path(args.replication_output).resolve(),
        args.device,
    )
    print(
        "[supervisor] preregistered follow-up complete; path 9/10 remains untouched.",
        flush=True,
    )


if __name__ == "__main__":
    main()
