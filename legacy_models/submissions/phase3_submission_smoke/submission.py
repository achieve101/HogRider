from pathlib import Path

from .runtime import Phase3FeedbackSubmission


def create_model(device: str = "cpu"):
    return Phase3FeedbackSubmission(Path(__file__).with_name("weights.pt"), device=device)
