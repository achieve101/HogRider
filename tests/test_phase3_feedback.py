import sys
import unittest
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from phase3_closed_loop import compute_phase3_loss, rollout_feedback_closed_loop
from phase3_model import FeedbackFIRController, extract_feedback_features
from v6_metrics import TOTAL_SAMPLES


def streaming_closed_loop(model, reference, target, path):
    model.reset()
    previous_error = 0.0
    y_history = np.zeros(len(path), dtype=np.float64)
    outputs, errors = [], []
    for x, d in zip(reference, target):
        output = model.process_sample(float(x), float(previous_error))
        y_history[1:] = y_history[:-1]
        y_history[0] = output
        anti = float(np.dot(path, y_history))
        previous_error = float(d - anti)
        outputs.append(output); errors.append(previous_error)
    return np.asarray(outputs), np.asarray(errors)


class Phase3FeedbackTests(unittest.TestCase):
    def make_model(self):
        torch.manual_seed(12)
        model = FeedbackFIRController(num_experts=2, fir_length=16, hidden_size=4, block_size=240)
        with torch.no_grad():
            model.expert_filters.normal_(0.0, 0.01)
            model.route_head.weight.normal_(0.0, 0.03)
        return model

    def test_streaming_matches_block_closed_loop(self):
        model = self.make_model().eval()
        rng = np.random.default_rng(2026)
        reference = rng.normal(0, 0.02, 720).astype(np.float32)
        target = rng.normal(0, 0.01, 720).astype(np.float32)
        path = np.asarray([0.8, -0.1, 0.03, 0.01], dtype=np.float32)
        stream_y, stream_e = streaming_closed_loop(model, reference, target, path)
        blocks = len(reference) // model.block_size
        rollout = rollout_feedback_closed_loop(
            model, torch.from_numpy(reference).unsqueeze(0),
            torch.from_numpy(target).unsqueeze(0),
            torch.from_numpy(path).reshape(1, 1, -1),
            torch.zeros(1, blocks, dtype=torch.long),
            torch.zeros(1, blocks, dtype=torch.long), truncate_blocks=0,
        )
        np.testing.assert_allclose(stream_y, rollout.output[0].detach().numpy(), atol=1e-6, rtol=1e-6)
        np.testing.assert_allclose(stream_e, rollout.residual[0].detach().numpy(), atol=1e-6, rtol=1e-6)

    def test_feedback_cannot_change_current_block(self):
        model = self.make_model().eval()
        reference = np.linspace(-0.02, 0.02, 480)
        first = []
        model.reset()
        for value in reference:
            first.append(model.process_sample(float(value), 0.0))
        second = []
        model.reset()
        for index, value in enumerate(reference):
            second.append(model.process_sample(float(value), 0.5 if index < 240 else 0.0))
        np.testing.assert_allclose(first[:240], second[:240], atol=0, rtol=0)
        self.assertGreater(np.max(np.abs(np.asarray(first[240:]) - np.asarray(second[240:]))), 0.0)

    def test_reset_is_reproducible_and_future_is_causal(self):
        model = self.make_model().eval()
        reference = np.linspace(-0.03, 0.03, 300)
        def run(values):
            model.reset()
            return np.asarray([model.process_sample(float(value), 0.01) for value in values])
        first = run(reference)
        second = run(reference)
        np.testing.assert_array_equal(first, second)
        changed = reference.copy(); changed[200:] *= -3
        future = run(changed)
        np.testing.assert_array_equal(first[:200], future[:200])

    def test_feature_shape_boundaries_limiter_and_complexity(self):
        model = FeedbackFIRController()
        block = torch.randn(2, 240) * 0.01
        context = torch.zeros(2, 32)
        features, next_x, next_y = extract_feedback_features(block, block * 0.5, block * 0.2, context, context)
        self.assertEqual(features.shape, (2, 10))
        self.assertEqual(next_x.shape, (2, 32)); self.assertEqual(next_y.shape, (2, 32))
        limited = model.soft_limit(torch.tensor([-100.0, 0.0, 100.0]))
        self.assertTrue(torch.all(limited.abs() < 0.98))
        complexity = model.get_complexity()
        self.assertEqual(complexity["parameter_count"], sum(p.numel() for p in model.parameters()))
        self.assertLess(complexity["steady_state_macs_per_sample"], 42_048)
        self.assertLess(complexity["peak_macs_in_one_sample_event"], 42_048)

    def test_complete_loss_backward_is_finite(self):
        torch.manual_seed(2)
        model = FeedbackFIRController(num_experts=2, fir_length=16, hidden_size=4)
        reference = torch.randn(1, TOTAL_SAMPLES) * 0.01
        target = torch.randn(1, TOTAL_SAMPLES) * 0.01
        paths = torch.zeros(1, 1, 8); paths[..., 0] = 0.8
        blocks = TOTAL_SAMPLES // model.block_size
        slots = torch.zeros(1, blocks, dtype=torch.long)
        labels = torch.zeros(1, blocks, dtype=torch.long)
        rollout = rollout_feedback_closed_loop(model, reference, target, paths, slots, labels)
        loss, components = compute_phase3_loss(target, rollout, labels)
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        gradients = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
        self.assertTrue(gradients)
        self.assertTrue(all(torch.isfinite(value).all() for value in gradients))
        self.assertGreater(sum(float(value.abs().sum()) for value in gradients), 0.0)
        self.assertEqual(components["guard_loss"].item(), 0.0)


if __name__ == "__main__":
    unittest.main()
