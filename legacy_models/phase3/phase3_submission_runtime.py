"""Participant-Kit-compatible runtime wrapper used by exported Phase-3 packages."""

from __future__ import annotations

from pathlib import Path

import torch

try:
    from .model import FeedbackFIRController
except ImportError:  # Local project testing before export.
    from phase3_model import FeedbackFIRController


class Phase3FeedbackSubmission:
    sample_rate = 48_000
    requires_error = True

    def __init__(self, checkpoint_path: str | Path, device: str = "cpu") -> None:
        self.device = torch.device(device)
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        self.model = FeedbackFIRController(**checkpoint["model_config"]).to(self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()
        self.reset()

    def reset(self) -> None:
        self.model.reset_streaming_state()

    def process_sample(self, reference_sample: float, previous_error_sample: float) -> float:
        with torch.inference_mode():
            output = self.model.process_sample(reference_sample, previous_error_sample)
        if not -0.98 < output < 0.98:
            raise FloatingPointError(f"Unsafe Phase-3 controller output: {output}.")
        return float(output)

    def get_complexity(self) -> dict[str, int]:
        return self.model.get_complexity()
