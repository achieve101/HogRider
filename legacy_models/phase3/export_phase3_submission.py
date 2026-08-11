"""Export a self-contained Participant-Kit feedback submission from a Phase-3 checkpoint."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    checkpoint_path = Path(args.checkpoint).resolve()
    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing submission directory: {output}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if checkpoint.get("phase") != 3 or "model_config" not in checkpoint:
        raise ValueError("The supplied file is not a Phase-3 checkpoint.")
    output.mkdir(parents=True)
    project_root = Path(__file__).resolve().parents[2]
    shutil.copy2(project_root / "phase3_model.py", output / "model.py")
    shutil.copy2(Path(__file__).with_name("phase3_submission_runtime.py"), output / "runtime.py")
    torch.save({
        "model_config": checkpoint["model_config"],
        "model_state_dict": checkpoint["model_state_dict"],
    }, output / "weights.pt")
    (output / "__init__.py").write_text("", encoding="utf-8")
    (output / "submission.py").write_text(
        '''from pathlib import Path\n\nfrom .runtime import Phase3FeedbackSubmission\n\n\ndef create_model(device: str = "cpu"):\n    return Phase3FeedbackSubmission(Path(__file__).with_name("weights.pt"), device=device)\n''',
        encoding="utf-8",
    )
    (output / "requirements.txt").write_text(
        "# Uses only the organizer-provided PyTorch and NumPy.\n", encoding="utf-8",
    )
    (output / "config.json").write_text(json.dumps({
        "phase": 3, "source_checkpoint": str(checkpoint_path),
        "model_config": checkpoint["model_config"],
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    (output / "README.md").write_text(
        "# Phase 3 feedback FIR submission\n\n"
        "Entry point: `submission:create_model`; requires `e[t-1]`.\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
