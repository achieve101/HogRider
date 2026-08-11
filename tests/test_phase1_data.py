import copy
import random
import sys
import unittest
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dataset import PreconvolutedANCDataset
from phase1_data import (
    TRANSITION_SAMPLE,
    build_validation_manifest,
    iter_validation_examples,
)
from legacy_models.phase0_phase1.train import scan_noise_files
from v6_metrics import TOTAL_SAMPLES


class Phase1DataTests(unittest.TestCase):
    def test_manifest_and_transition_example(self):
        manifest = build_validation_manifest(ROOT / "dataset")
        self.assertEqual(manifest["total_samples"], 168_000)
        self.assertEqual(manifest["scoring_window_samples"], 24_000)
        self.assertEqual(manifest["scoring_window_count"], 6)
        self.assertEqual(manifest["transition_sample"], 96_000)
        self.assertEqual(TRANSITION_SAMPLE, 96_000)

        transition_manifest = copy.deepcopy(manifest)
        transition_manifest["scenes"] = [manifest["scenes"][2]]
        transition_manifest["path_indices_zero_based"] = [0]
        scene, path, reference, secondary, disturbance = next(
            iter_validation_examples(ROOT / "dataset", transition_manifest)
        )
        self.assertEqual(scene, "vehicle_to_restaurant")
        self.assertEqual(path, 0)
        self.assertEqual(reference.shape, (TOTAL_SAMPLES,))
        self.assertEqual(disturbance.shape, (TOTAL_SAMPLES,))
        self.assertEqual(secondary.ndim, 1)
        self.assertTrue(torch.isfinite(reference).all())
        self.assertTrue(torch.isfinite(disturbance).all())

    def test_training_sampling_is_reproducible(self):
        noises = scan_noise_files(ROOT / "dataset" / "NOISE")[:-2]
        dataset = PreconvolutedANCDataset(
            ROOT / "dataset", noises, list(range(8)),
            segment_duration=3.5, is_train=True, samples_per_epoch=2,
        )
        random.seed(2026)
        np.random.seed(2026)
        first = dataset[0]
        random.seed(2026)
        np.random.seed(2026)
        second = dataset[0]
        for left, right in zip(first, second):
            self.assertTrue(torch.equal(left, right))


if __name__ == "__main__":
    unittest.main()
