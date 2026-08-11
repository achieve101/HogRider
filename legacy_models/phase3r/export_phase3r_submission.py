"""Export a self-contained, diagnostic-only Phase-3R feedback candidate."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import torch


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    checkpoint_path, output = Path(args.checkpoint).resolve(), Path(args.output).resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}.")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if checkpoint.get("phase") != "3R":
        raise ValueError("Checkpoint is not a Phase-3R candidate.")
    output.mkdir(parents=True)
    project_root = Path(__file__).resolve().parents[2]
    shutil.copy2(project_root / "phase3r_model.py", output / "model.py")
    shutil.copy2(Path(__file__).with_name("phase3r_submission_runtime.py"), output / "runtime.py")
    torch.save({
        "model_config": checkpoint["model_config"],
        "model_state_dict": checkpoint["model_state_dict"],
    }, output / "weights.pt")
    (output / "__init__.py").write_text("", encoding="utf-8")
    (output / "submission.py").write_text(
        "from pathlib import Path\nfrom .runtime import Phase3RSubmission\n\n"
        "def create_model(device: str = 'cpu'):\n"
        "    return Phase3RSubmission(Path(__file__).with_name('weights.pt'), device=device)\n",
        encoding="utf-8",
    )
    (output / "requirements.txt").write_text(
        "# Uses only organizer-provided PyTorch and NumPy.\n", encoding="utf-8",
    )
    (output / "config.json").write_text(json.dumps({
        "phase": "3R", "formal_upgrade": False,
        "reason": "Development passed, but eight-fold LOPO failed.",
        "source_checkpoint": str(checkpoint_path), "model_config": checkpoint["model_config"],
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    (output / "README.md").write_text(
        "# Phase 3R diagnostic candidate\n\n"
        "This package exercises the feedback API but is not the formal model: LOPO failed.\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
