import math
import sys
import unittest
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
KIT_DIR = ROOT / "DEEPANC_PARTICIPANT_KIT"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(KIT_DIR))

from public_demo_scoring import score_windowed_signals
from v6_metrics import (
    INITIALIZATION_SAMPLES,
    SCORING_WINDOW_COUNT,
    SCORING_WINDOW_SAMPLES,
    TOTAL_SAMPLES,
    compute_v6_loss,
    compute_v6_metrics,
)


class V6MetricsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        generator = torch.Generator().manual_seed(2026)
        cls.disturbance = torch.randn(
            1, TOTAL_SAMPLES, generator=generator, dtype=torch.float64,
        ) * 0.03

    def official(self, disturbance, residual):
        scored = slice(INITIALIZATION_SAMPLES, None)
        return score_windowed_signals(
            disturbance[0, scored].numpy(),
            residual[0, scored].numpy(),
            np.zeros(SCORING_WINDOW_COUNT * SCORING_WINDOW_SAMPLES),
            sample_rate=48_000,
            window_samples=SCORING_WINDOW_SAMPLES,
        )

    def test_matches_official_scorer(self):
        residual = self.disturbance * 0.73
        ours = compute_v6_metrics(self.disturbance, residual)
        official = self.official(self.disturbance, residual)
        self.assertAlmostEqual(
            ours["primary_score_db"].item(), official["primary_score_db"],
            delta=1e-4,
        )
        self.assertAlmostEqual(
            ours["rebound_score_db"].item(),
            official["third_octave_rebound_peak_1000_8000_db"],
            delta=1e-4,
        )
        for index, window in enumerate(official["window_results"]):
            self.assertAlmostEqual(
                ours["primary_window_db"][0, index].item(),
                window["primary_score_db"], delta=1e-4,
            )
            self.assertAlmostEqual(
                ours["rebound_window_db"][0, index].item(),
                window["third_octave_rebound_peak_1000_8000_db"],
                delta=1e-4,
            )

    def test_zero_attenuation_and_amplification(self):
        zero = compute_v6_metrics(self.disturbance, self.disturbance)
        self.assertAlmostEqual(zero["primary_score_db"].item(), 0.0, delta=1e-8)
        self.assertAlmostEqual(zero["rebound_score_db"].item(), 0.0, delta=1e-8)

        attenuated = compute_v6_metrics(self.disturbance, self.disturbance * 0.5)
        self.assertAlmostEqual(
            attenuated["primary_score_db"].item(), 6.020599913, delta=1e-6,
        )
        self.assertAlmostEqual(
            attenuated["rebound_score_db"].item(), 0.0, delta=1e-8,
        )

        amplified = compute_v6_metrics(self.disturbance, self.disturbance * 2.0)
        self.assertAlmostEqual(
            amplified["primary_score_db"].item(), -6.020599913, delta=1e-6,
        )
        self.assertAlmostEqual(
            amplified["rebound_score_db"].item(), 6.020599913, delta=1e-6,
        )

    def test_six_windows_are_averaged_in_db(self):
        gains = torch.tensor(
            [0.5, 0.75, 1.0, 1.25, 1.5, 2.0], dtype=torch.float64,
        )
        residual = self.disturbance.clone()
        scored = residual[:, INITIALIZATION_SAMPLES:].reshape(
            1, SCORING_WINDOW_COUNT, SCORING_WINDOW_SAMPLES,
        )
        scored.mul_(gains.view(1, -1, 1))
        metrics = compute_v6_metrics(self.disturbance, residual)
        expected_primary = float(torch.mean(-20.0 * torch.log10(gains)))
        expected_rebound = float(torch.mean(torch.relu(20.0 * torch.log10(gains))))
        self.assertAlmostEqual(
            metrics["primary_score_db"].item(), expected_primary, delta=1e-6,
        )
        self.assertAlmostEqual(
            metrics["rebound_score_db"].item(), expected_rebound, delta=1e-6,
        )

    def test_initialization_is_excluded_and_guard_is_thresholded(self):
        disturbance_b = self.disturbance.clone()
        disturbance_b[:, :INITIALIZATION_SAMPLES] *= 50.0
        controller = torch.zeros_like(self.disturbance)
        loss_a, comp_a = compute_v6_loss(
            self.disturbance, self.disturbance * 0.8, controller, 0.7, 0.3,
        )
        loss_b, comp_b = compute_v6_loss(
            disturbance_b, self.disturbance * 0.8, controller, 0.7, 0.3,
        )
        self.assertAlmostEqual(loss_a.item(), loss_b.item(), delta=1e-10)
        self.assertEqual(comp_a["guard_loss"].item(), 0.0)

        unsafe = controller.clone()
        unsafe[0, 123] = 1.1
        _, unsafe_components = compute_v6_loss(
            self.disturbance, self.disturbance, unsafe, 0.7, 0.3,
        )
        self.assertGreater(unsafe_components["guard_loss"].item(), 0.0)

    def test_backward_is_finite_and_nonzero(self):
        residual = (self.disturbance.float() * 0.9).detach().requires_grad_(True)
        controller = torch.zeros_like(residual, requires_grad=True)
        loss, _ = compute_v6_loss(
            self.disturbance.float(), residual, controller, 0.7, 0.3,
        )
        loss.backward()
        self.assertTrue(torch.isfinite(residual.grad).all())
        self.assertGreater(residual.grad.abs().sum().item(), 0.0)


if __name__ == "__main__":
    unittest.main()

