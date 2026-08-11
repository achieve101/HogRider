import torch
import torch.nn.functional as F
import numpy as np
import random
import soundfile as sf
from pathlib import Path
from torch.utils.data import Dataset

# =====================================================================
# 1. Dynamic Secondary Path Convolution
# =====================================================================

def apply_dynamic_path(signal_batch, path_batch):
    """
    Dynamic Convolution Function (Physical Phase Aligned).
    
    Applies the secondary acoustic path (impulse response) to the predicted 
    anti-noise signal. 
    
    Note: PyTorch's F.conv1d performs cross-correlation natively. To 
    accurately simulate a true causal forward acoustic convolution (as it 
    occurs in physical space), the impulse response filter must be flipped 
    along the time dimension prior to the operation.
    """
    B, T = signal_batch.shape
    L = path_batch.shape[1]
    
    signal_reshaped = signal_batch.view(1, B, T) 
    
    # Flip the secondary path to align with the physical convolution direction
    path_flipped = torch.flip(path_batch, dims=[1])
    path_reshaped = path_flipped.view(B, 1, L)     
    
    pad_len = L - 1
    signal_padded = F.pad(signal_reshaped, (pad_len, 0))
    
    output = F.conv1d(signal_padded, path_reshaped, groups=B) 
    return output.squeeze(0) 

# =====================================================================
# 2. Offline Expected Noise Dataset
# =====================================================================

class PreconvolutedANCDataset(Dataset):
    """
    Dataset class for loading pre-convoluted Active Noise Control data.
    Provides strictly aligned, time-domain triplets for the network:
    [Raw Reference Noise, Secondary Path, Expected Target Noise].
    """
    def __init__(self, dataset_dir, noise_names, path_indices, segment_duration=1.0,
                 sr=48000, is_train=True, samples_per_epoch=None,
                 skip_seconds=20.0):
        self.dataset_dir = Path(dataset_dir)
        self.noise_names = list(noise_names)
        self.path_indices = list(path_indices)
        self.sr = sr
        self.segment_length = int(segment_duration * sr)
        self.is_train = is_train
        self.samples_per_epoch = samples_per_epoch
        self.skip_samples = int(skip_seconds * sr)

        if not self.noise_names:
            raise ValueError("noise_names must contain at least one noise source.")
        if not self.path_indices:
            raise ValueError("path_indices must contain at least one path index.")
        if self.segment_length <= 0:
            raise ValueError("segment_duration must produce a positive segment length.")
        if samples_per_epoch is not None and samples_per_epoch <= 0:
            raise ValueError("samples_per_epoch must be positive when provided.")

        # Load the spatially averaged secondary acoustic paths
        sh_path = self.dataset_dir / 'sh.npy'
        self.sh_paths = np.load(sh_path).T 
        
        self.expected_dir = self.dataset_dir / 'EXPECTED_NOISE'
        self.raw_noise_dir = self.dataset_dir / 'NOISE'
        self.noise_records = [self._build_noise_record(name) for name in self.noise_names]
        self._expected_path_cache = {}

        invalid_paths = [idx for idx in self.path_indices if idx < 0 or idx >= len(self.sh_paths)]
        if invalid_paths:
            raise IndexError(
                f"Path indices {invalid_paths} are outside the available range "
                f"0..{len(self.sh_paths) - 1}."
            )

        # Fail early when the dataset naming or pairing is incomplete.
        for record in self.noise_records:
            for path_idx in self.path_indices:
                self._resolve_expected_path(record, path_idx)

    def __len__(self):
        if self.is_train and self.samples_per_epoch is not None:
            return self.samples_per_epoch
        return len(self.path_indices)

    def _build_noise_record(self, noise_name):
        """Resolve both legacy stem identifiers and exact WAV filenames."""
        requested = str(noise_name)
        candidates = [self.raw_noise_dir / requested]
        if Path(requested).suffix.lower() != '.wav':
            candidates.extend([
                self.raw_noise_dir / f"{requested}.wav",
                self.raw_noise_dir / f"{requested}.WAV",
            ])

        raw_path = next((path for path in candidates if path.is_file()), None)
        if raw_path is None:
            raise FileNotFoundError(
                f"Cannot resolve raw noise {requested!r} in {self.raw_noise_dir}."
            )

        info = sf.info(str(raw_path))
        if info.samplerate != self.sr:
            raise ValueError(
                f"Unexpected sample rate for {raw_path}: {info.samplerate}, expected {self.sr}."
            )

        return {
            'name': requested,
            'raw_path': raw_path,
            'stem': raw_path.stem,
            'filename': raw_path.name,
        }

    def _resolve_expected_path(self, noise_record, path_idx):
        """Resolve standard and filename-preserving expected-noise conventions."""
        cache_key = (noise_record['filename'], path_idx)
        if cache_key in self._expected_path_cache:
            return self._expected_path_cache[cache_key]

        scene_suffix = f"_scene_{path_idx + 1:02d}.wav"
        candidates = [
            self.expected_dir / f"{noise_record['stem']}{scene_suffix}",
            self.expected_dir / f"{noise_record['filename']}{scene_suffix}",
        ]
        expected_path = next((path for path in candidates if path.is_file()), None)
        if expected_path is None:
            candidate_text = ', '.join(str(path) for path in candidates)
            raise FileNotFoundError(
                f"Missing expected noise for {noise_record['filename']} and path "
                f"{path_idx + 1}: tried {candidate_text}."
            )

        info = sf.info(str(expected_path))
        if info.samplerate != self.sr:
            raise ValueError(
                f"Unexpected sample rate for {expected_path}: {info.samplerate}, expected {self.sr}."
            )

        self._expected_path_cache[cache_key] = expected_path
        return expected_path

    def _fast_read_slice(self, filepath, start_idx, num_frames=None):
        """ 
        High-speed audio segment extraction using Soundfile pointers.
        Ensures memory-efficient reading without loading the entire audio file.
        """
        frames = self.segment_length if num_frames is None else num_frames
        y, _ = sf.read(str(filepath), start=start_idx, frames=frames,
                       dtype='float32', always_2d=False)
        # Enforce mono channel output if the source file is multi-channel
        if y.ndim > 1:
            y = np.mean(y, axis=1)
        return y

    def __getitem__(self, idx):
        # Cycling by path keeps path sampling balanced even with a virtual epoch.
        path_idx = self.path_indices[idx % len(self.path_indices)]
        sh = self.sh_paths[path_idx]
        
        if self.is_train:
            # Training Phase: Randomly sample a noise environment
            chosen_noise = random.choice(self.noise_records)
            exp_noise_path = self._resolve_expected_path(chosen_noise, path_idx)
            raw_noise_path = chosen_noise['raw_path']
            
            # Fetch total frames via metadata to avoid full disk I/O
            total_frames = sf.info(str(raw_noise_path)).frames
            max_start = total_frames - self.segment_length
            
            start_idx = (
                np.random.randint(self.skip_samples, max_start + 1)
                if max_start >= self.skip_samples
                else max(0, max_start)
            )
                
            seg_exp = self._fast_read_slice(exp_noise_path, start_idx)
            seg_raw = self._fast_read_slice(raw_noise_path, start_idx)
            
        else:
            # Testing Phase: Deterministic scene transition (splicing logic)
            # Simulates an abrupt acoustic environment change at the midpoint of the sample.
            scene1 = self.noise_records[0]
            scene2 = self.noise_records[1] if len(self.noise_records) >= 2 else scene1
            
            exp_s1_path = self._resolve_expected_path(scene1, path_idx)
            raw_s1_path = scene1['raw_path']
            exp_s2_path = self._resolve_expected_path(scene2, path_idx)
            raw_s2_path = scene2['raw_path']
            
            half_len = self.segment_length // 2
            
            start1 = self.skip_samples + idx * half_len
            start2 = self.skip_samples + idx * half_len
            
            max_s1 = sf.info(str(raw_s1_path)).frames - half_len
            max_s2 = sf.info(str(raw_s2_path)).frames - half_len
            
            # Boundary protection with modulo logic
            if start1 > max_s1:
                start1 = self.skip_samples + (start1 % max(1, max_s1 - self.skip_samples))
            if start2 > max_s2:
                start2 = self.skip_samples + (start2 % max(1, max_s2 - self.skip_samples))
            start1 = max(0, min(start1, max_s1))
            start2 = max(0, min(start2, max_s2))
            
            # Each source contributes exactly half of the requested segment.
            seg_exp_s1 = self._fast_read_slice(exp_s1_path, start1, half_len)
            seg_raw_s1 = self._fast_read_slice(raw_s1_path, start1, half_len)
            seg_exp_s2 = self._fast_read_slice(exp_s2_path, start2, self.segment_length - half_len)
            seg_raw_s2 = self._fast_read_slice(raw_s2_path, start2, self.segment_length - half_len)

            seg_exp = np.concatenate([seg_exp_s1, seg_exp_s2])
            seg_raw = np.concatenate([seg_raw_s1, seg_raw_s2])

        # Zero-padding fallback for dimension safety at the end of audio files
        if len(seg_exp) < self.segment_length:
            seg_exp = np.pad(seg_exp, (0, self.segment_length - len(seg_exp)), 'constant')
        if len(seg_raw) < self.segment_length:
            seg_raw = np.pad(seg_raw, (0, self.segment_length - len(seg_raw)), 'constant')
                

        return torch.tensor(seg_raw, dtype=torch.float32), \
               torch.tensor(sh, dtype=torch.float32), \
               torch.tensor(seg_exp, dtype=torch.float32)
