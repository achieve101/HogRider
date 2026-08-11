import sys
import unittest
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from phase2_paths import (
    Phase2GroupedDataset,
    augment_secondary_path,
    compute_phase2_group_loss,
    robust_path_reduce,
)
from legacy_models.phase0_phase1.train import scan_noise_files
from v6_metrics import TOTAL_SAMPLES


class Phase2PathTests(unittest.TestCase):
    def test_augmentation_is_bounded_causal_and_reproducible(self):
        path = np.zeros(1967, dtype=np.float32)
        path[5] = 1.0
        first, spec = augment_secondary_path(path, np.random.default_rng(77))
        second, second_spec = augment_secondary_path(path, np.random.default_rng(77))
        np.testing.assert_array_equal(first, second)
        self.assertEqual(spec, second_spec)
        self.assertEqual(first.shape, path.shape)
        self.assertTrue(np.isfinite(first).all())
        self.assertGreaterEqual(spec["gain_db"], -1.5)
        self.assertLessEqual(spec["gain_db"], 1.5)
        self.assertIn(spec["delay_samples"], range(-2, 3))
        self.assertTrue(np.all(first[:max(0, 5 + spec["delay_samples"])] == 0))

    def test_tail_perturbation_energy_and_early_identity(self):
        rng = np.random.default_rng(4)
        path = rng.standard_normal(1967).astype(np.float32) * 0.01
        augmented, _ = augment_secondary_path(
            path, np.random.default_rng(8), gain_db=0.0, delay_samples=0,
            tail_energy_db=-32.0,
        )
        difference = augmented.astype(np.float64) - path.astype(np.float64)
        self.assertTrue(np.all(difference[:64] == 0))
        ratio_db = 10 * np.log10(np.square(difference[64:]).sum() / np.square(path).sum())
        self.assertAlmostEqual(ratio_db, -32.0, delta=1e-4)

    def test_group_dataset_keeps_real_member_and_excludes_final_paths(self):
        noises = scan_noise_files(ROOT / "dataset" / "NOISE")
        dataset = Phase2GroupedDataset(
            ROOT / "dataset", noises, range(8), group_size=4,
            samples_per_epoch=2, seed=2026,
        )
        first = dataset[0]
        second = dataset[0]
        for left, right in zip(first, second):
            self.assertTrue(torch.equal(left, right))
        reference, paths, targets, base_indices, augmented = first
        self.assertEqual(reference.shape, (TOTAL_SAMPLES,))
        self.assertEqual(paths.shape[0], 4)
        self.assertEqual(targets.shape, (4, TOTAL_SAMPLES))
        self.assertFalse(bool(augmented[0]))
        self.assertEqual(int(base_indices[0]), 0)
        self.assertTrue(torch.all(base_indices < 8))

    def test_top_quartile_is_the_hard_worst_for_four_paths(self):
        losses = torch.tensor([[1.0, 4.0, 2.0, 3.0], [8.0, 5.0, 7.0, 6.0]])
        mean, worst, count = robust_path_reduce(losses)
        self.assertEqual(count, 1)
        self.assertEqual(mean.item(), 4.5)
        self.assertEqual(worst.item(), 6.0)

    def test_group_loss_backward_is_finite_nonzero_and_guard_zero(self):
        generator = torch.Generator().manual_seed(2026)
        target = torch.randn(1, 4, TOTAL_SAMPLES, generator=generator) * 0.02
        controller = (torch.randn(1, TOTAL_SAMPLES, generator=generator) * 0.001).requires_grad_(True)
        paths = torch.zeros(1, 4, 65)
        paths[:, :, 0] = torch.tensor([0.8, 0.9, 1.0, 1.1])
        loss, components = compute_phase2_group_loss(target, controller, paths)
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(torch.isfinite(controller.grad).all())
        self.assertGreater(controller.grad.abs().sum().item(), 0.0)
        self.assertEqual(components["guard_loss"].item(), 0.0)
        self.assertEqual(components["top_quartile_count"].item(), 1)


if __name__ == "__main__":
    unittest.main()
