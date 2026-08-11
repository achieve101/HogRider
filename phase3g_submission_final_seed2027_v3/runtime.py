"""Participant-Kit-compatible frozen Phase-3G runtime."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

try:
    from .model import GenerativeInnovationFIRController
except ImportError:
    from phase3g_model import GenerativeInnovationFIRController


class Phase3GSubmission:
    sample_rate=48_000
    requires_error=True

    def __init__(self, checkpoint_path: str | Path, device: str="cpu") -> None:
        if device != "cpu":
            raise ValueError("Phase-3G streaming runtime is CPU-only.")
        checkpoint=torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        # Module constructors normally consume the global RNG for their
        # temporary initial values.  Restore it immediately: all real values
        # come from the frozen checkpoint and inference itself is RNG-free.
        with torch.random.fork_rng(devices=[]):
            self.model=GenerativeInnovationFIRController(**checkpoint["model_config"])
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        self.reset()

    def reset(self) -> None:
        self.model.reset()

    def process_sample(self, reference_sample: float, previous_error_sample: float) -> float:
        # The Participant Kit already wraps the strict sample loop in
        # ``torch.inference_mode()``.  More importantly, the controller is
        # NumPy-only on ordinary samples and applies its own local inference
        # guard around the block-rate GRU update.  Entering a new PyTorch
        # context for every sample is therefore redundant and measurably
        # increases the CPU RTF.
        output=float(self.model.process_sample(reference_sample, previous_error_sample))
        if not np.isfinite(output) or not -0.98 < output < 0.98:
            raise FloatingPointError(f"Unsafe Phase-3G controller output: {output}.")
        return output

    def get_complexity(self) -> dict[str, int]:
        return self.model.get_complexity()
