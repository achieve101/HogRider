import sys
import unittest
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from phase3r_model import InnovationRoutedFIRController


class Phase3RTests(unittest.TestCase):
    def make_model(self, **kwargs):
        model = InnovationRoutedFIRController(
            num_experts=2, fir_length=8, path_length=4, n_fft=64, block_size=8,
            **kwargs,
        )
        with torch.no_grad():
            model.expert_filters[0] = torch.tensor([0.2, -0.1, 0.03, 0, 0, 0, 0, 0])
            model.expert_filters[1] = torch.tensor([-0.1, 0.2, 0.01, 0, 0, 0, 0, 0])
            model.secondary_paths[0] = torch.tensor([0.8, -0.2, 0.1, 0.03])
            model.secondary_paths[1] = torch.tensor([0.4, 0.1, -0.05, 0.02])
            model.primary_real.zero_(); model.primary_imag.zero_()
        return model.eval()

    def test_candidate_secondary_convolution_matches_direct(self):
        model = self.make_model(); model.reset()
        rng = np.random.default_rng(4)
        values = rng.normal(0, 0.02, 40)
        outputs = []
        for index, value in enumerate(values):
            outputs.append(model.process_sample(value, 0.0))
            # e[t-1] finalizes the previous sample, so each block becomes
            # available on the first call following its final output.
            if index in (8, 16, 24, 32):
                completed = np.asarray(outputs[index - 8:index])
                expected = []
                full = np.asarray(outputs[:index])
                for path in model.secondary_paths.numpy():
                    direct = np.convolve(full, path)[:full.size]
                    expected.append(direct[-8:])
                np.testing.assert_allclose(
                    model._stream["last_candidate_anti_block"], expected, atol=1e-7, rtol=1e-7,
                )

    def test_current_error_cannot_change_current_output(self):
        first, second = self.make_model(), self.make_model()
        reference = np.linspace(-0.02, 0.02, 100)
        y1, y2 = [], []
        for index, value in enumerate(reference):
            y1.append(first.process_sample(value, 0.0))
            y2.append(second.process_sample(value, 0.5 if index >= 50 else 0.0))
        np.testing.assert_array_equal(y1[:50], y2[:50])

    def test_reset_and_future_causality_are_exact(self):
        model = self.make_model()
        reference = np.linspace(-0.03, 0.03, 100)
        def run(values):
            model.reset()
            return np.asarray([model.process_sample(value, 0.01) for value in values])
        first, second = run(reference), run(reference)
        np.testing.assert_array_equal(first, second)
        changed = reference.copy(); changed[70:] *= -2
        future = run(changed)
        np.testing.assert_array_equal(first[:70], future[:70])

    def test_low_energy_holds_uniform_route(self):
        model = self.make_model(); model.reset()
        for _ in range(200):
            model.process_sample(0.0, 0.0)
        np.testing.assert_array_equal(model.current_alpha(), np.asarray([0.5, 0.5]))
        self.assertEqual(model.route_diagnostics(), [])

    def test_first_route_update_is_exactly_on_fft_block_boundary(self):
        model = self.make_model()
        with torch.no_grad():
            model.secondary_paths.zero_()
            model.primary_real.zero_(); model.primary_imag.zero_()
            model.primary_real[0] = 1.0
        model.reset()
        reference = np.sin(2 * np.pi * np.arange(80) / 16) * 0.02
        for index, value in enumerate(reference):
            previous_error = 0.0 if index == 0 else reference[index - 1]
            model.process_sample(value, previous_error)
        trace = model.route_diagnostics()
        self.assertTrue(trace)
        self.assertEqual(trace[0]["completed_samples"], 64)
        self.assertEqual(trace[0]["winner_zero_based"], 0)

    def test_probe_is_bounded_stops_and_resets(self):
        model = self.make_model(probe_rms=0.01, probe_samples=20, probe_seed=2026)
        with torch.no_grad():
            model.expert_filters.zero_()
        def run():
            model.reset()
            return np.asarray([model.process_sample(0.0, 0.0) for _ in range(30)])
        first, second = run(), run()
        np.testing.assert_array_equal(first, second)
        self.assertTrue(np.all(first[:20] != 0.0))
        np.testing.assert_array_equal(first[20:], np.zeros(10))
        self.assertTrue(np.all(np.abs(first) < 0.98))


if __name__ == "__main__":
    unittest.main()
