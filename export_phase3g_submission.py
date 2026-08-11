"""Export a self-contained frozen Phase-3G feedback candidate."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import torch


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--formal-upgrade", action="store_true")
    args=parser.parse_args()
    checkpoint_path=Path(args.checkpoint).resolve(); output=Path(args.output).resolve()
    if output.exists(): raise FileExistsError(f"Refusing to overwrite {output}.")
    checkpoint=torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if checkpoint.get("phase") != "3G": raise ValueError("Checkpoint is not a Phase-3G candidate.")
    output.mkdir(parents=True); root=Path(__file__).resolve().parent
    model_source=(root/"phase3g_model.py").read_text(encoding="utf-8")
    begin="    # BEGIN TRAINING_ONLY_EXPORT_EXCLUSION\n"
    end="    # END TRAINING_ONLY_EXPORT_EXCLUSION\n"
    if model_source.count(begin) != 1 or model_source.count(end) != 1:
        raise RuntimeError("Phase-3G model export exclusion markers are missing or ambiguous.")
    prefix, remainder=model_source.split(begin, 1)
    _, suffix=remainder.split(end, 1)
    (output/"model.py").write_text(prefix+suffix, encoding="utf-8")
    shutil.copy2(root/"phase3g_submission_runtime.py", output/"runtime.py")
    torch.save({"model_config":checkpoint["model_config"], "model_state_dict":checkpoint["model_state_dict"]}, output/"weights.pt")
    (output/"__init__.py").write_text("", encoding="utf-8")
    (output/"submission.py").write_text(
        "from pathlib import Path\nfrom .runtime import Phase3GSubmission\n\n"
        "def create_model(device: str = 'cpu'):\n"
        "    return Phase3GSubmission(Path(__file__).with_name('weights.pt'), device=device)\n",
        encoding="utf-8",
    )
    (output/"requirements.txt").write_text("# Uses only organizer-provided PyTorch and NumPy.\n", encoding="utf-8")
    (output/"config.json").write_text(json.dumps({
        "phase":"3G", "formal_upgrade":bool(args.formal_upgrade),
        "source_checkpoint":str(checkpoint_path), "model_config":checkpoint["model_config"],
        "inference_policy":"frozen state_dict; only hidden activations and caches change",
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    (output/"README.md").write_text(
        "# Phase 3G frozen generative FIR controller\n\n"
        "All learned tensors are frozen during inference. The generated FIR is an activation cleared by reset().\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
