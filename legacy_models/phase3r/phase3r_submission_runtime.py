"""Participant-Kit-compatible runtime for exported Phase-3R candidates."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

try:
    from .model import InnovationRoutedFIRController
except ImportError:
    from phase3r_model import InnovationRoutedFIRController


class Phase3RSubmission:
    sample_rate = 48_000
    requires_error = True

    def __init__(self, checkpoint_path: str | Path, device: str = "cpu") -> None:
        if device != "cpu":
            raise ValueError("Phase-3R streaming runtime is CPU-only.")
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        self.model = InnovationRoutedFIRController(**checkpoint["model_config"])
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()
        self.reset()

    def reset(self) -> None:
        self.model.reset()

    def process_sample(self, reference_sample: float, previous_error_sample: float) -> float:
        output = float(self.model.process_sample(reference_sample, previous_error_sample))
        if not np.isfinite(output) or not -0.98 < output < 0.98:
            raise FloatingPointError(f"Unsafe Phase-3R controller output: {output}.")
        return output

    def get_complexity(self) -> dict[str, int]:
        return self.model.get_complexity()
