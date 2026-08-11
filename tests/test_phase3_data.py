import sys
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from phase3_data import Phase3SequenceDataset
from phase3_validation import build_phase3_manifests
from legacy_models.phase0_phase1.train import scan_noise_files
from v6_metrics import TOTAL_SAMPLES


class Phase3DataTests(unittest.TestCase):
    def test_static_split_and_final_paths_are_isolated(self):
        manifests = build_phase3_manifests(ROOT / "dataset")
        self.assertEqual(manifests["development"]["path_indices_zero_based"], list(range(8)))
        self.assertEqual(manifests["final"]["path_indices_zero_based"], [8, 9])
        pairs = {value for pair in manifests["path_switch"]["pairs_zero_based"] for value in pair}
        self.assertTrue(pairs.issubset(set(range(8))))

    def test_forced_switch_is_deterministic_and_block_aligned(self):
        noises = scan_noise_files(ROOT / "dataset" / "NOISE")[:-2]
        dataset = Phase3SequenceDataset(
            ROOT / "dataset", noises, samples_per_epoch=1,
            switch_probability=1.0, augmentation_probability=0.0, seed=2026,
        )
        first = dataset[0]
        second = dataset[0]
        for left, right in zip(first, second):
            self.assertTrue(torch.equal(left, right))
        reference, target, paths, slots, labels, first_path, second_path = first
        self.assertEqual(reference.shape, (TOTAL_SAMPLES,))
        self.assertEqual(target.shape, (TOTAL_SAMPLES,))
        self.assertEqual(paths.shape[0], 2)
        self.assertNotEqual(int(first_path), int(second_path))
        self.assertTrue(torch.all(slots[:400] == 0))
        self.assertTrue(torch.all(slots[400:] == 1))
        self.assertTrue(torch.all(labels[:400] == first_path))
        self.assertTrue(torch.all(labels[400:] == second_path))
        self.assertTrue(torch.all(labels < 8))

    def test_training_rejects_final_paths(self):
        noises = scan_noise_files(ROOT / "dataset" / "NOISE")[:-2]
        with self.assertRaises(ValueError):
            Phase3SequenceDataset(ROOT / "dataset", noises, train_paths=[0, 8])


if __name__ == "__main__":
    unittest.main()
