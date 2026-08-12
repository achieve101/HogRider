from pathlib import Path
from .runtime import Phase3GSubmission

def create_model(device: str = 'cpu'):
    return Phase3GSubmission(Path(__file__).with_name('weights.pt'), device=device)
