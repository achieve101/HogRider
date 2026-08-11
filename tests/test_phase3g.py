import sys
import unittest
from pathlib import Path

import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from phase3g_closed_loop import rollout_phase3g_closed_loop
from phase3g_data import dtw_align_ir, synthesize_path
from phase3g_model import GenerativeInnovationFIRController
from phase3g_validation import state_dict_sha256


class Phase3GTests(unittest.TestCase):
    def make_model(self):
        model=GenerativeInnovationFIRController(
            num_experts=2, fir_length=8, path_length=4, n_fft=64,
            block_size=8, hidden_size=4, latent_size=3,
        )
        with torch.no_grad():
            model.expert_filters.copy_(torch.tensor([
                [0.2, -0.1, 0.03, 0, 0, 0, 0, 0],
                [-0.1, 0.2, 0.01, 0, 0, 0, 0, 0],
            ]))
            model.secondary_paths.copy_(torch.tensor([
                [0.8, -0.2, 0.1, 0.03], [0.4, 0.1, -0.05, 0.02],
            ]))
            model.primary_real.zero_(); model.primary_imag.zero_()
            model.primary_real[0] = 1.0
            model.residual_dictionary.normal_(0, 0.002)
            model.latent_head.weight.normal_(0, 0.01)
        return model.eval()

    def test_default_parameter_count_matches_plan(self):
        model=GenerativeInnovationFIRController()
        self.assertEqual(model.get_complexity()["trainable_parameter_count"], 41_552)

    def test_streaming_does_not_modify_state_dict_and_resets_exactly(self):
        model=self.make_model(); before=state_dict_sha256(model)
        reference=np.sin(2*np.pi*np.arange(120)/16)*0.02
        def run():
            model.reset(); result=[]; previous=0.0
            for value in reference:
                result.append(model.process_sample(float(value), previous))
                previous=float(value)*0.1
            return np.asarray(result), [item["latent"] for item in model.route_diagnostics()]
        first, route_first=run(); second, route_second=run()
        np.testing.assert_array_equal(first, second)
        self.assertEqual(route_first, route_second)
        self.assertEqual(before, state_dict_sha256(model))

    def test_current_error_cannot_change_current_output(self):
        first=self.make_model(); second=self.make_model(); second.load_state_dict(first.state_dict())
        reference=np.linspace(-0.02, 0.02, 100); y1=[]; y2=[]
        for index,value in enumerate(reference):
            y1.append(first.process_sample(float(value), 0.0))
            y2.append(second.process_sample(float(value), 0.5 if index >= 50 else 0.0))
        np.testing.assert_array_equal(y1[:50], y2[:50])

    def test_short_batch_and_streaming_are_consistent(self):
        model=self.make_model(); samples=80
        reference=(np.sin(2*np.pi*np.arange(samples)/16)*0.02).astype(np.float32)
        disturbance=(reference*0.2).astype(np.float32)
        path=model.secondary_paths[0].numpy().copy()
        batch=rollout_phase3g_closed_loop(
            model, torch.from_numpy(reference)[None], torch.from_numpy(disturbance)[None],
            torch.from_numpy(np.stack((path, path)))[None],
            torch.zeros((1, samples//8), dtype=torch.long),
            torch.ones((1, samples//8, 2), dtype=torch.bool),
            torch.zeros((1, samples//8), dtype=torch.long),
            torch.ones((1, samples//8), dtype=torch.bool), truncate_blocks=0,
        )
        model.reset(); output=[]; ring=np.zeros(path.size*2); pointer=0; previous=0.0
        for x,d in zip(reference, disturbance):
            y=model.process_sample(float(x), previous); output.append(y)
            pointer=(pointer-1)%path.size; ring[pointer]=y; ring[pointer+path.size]=y
            previous=float(d-np.dot(path, ring[pointer:pointer+path.size]))
        np.testing.assert_allclose(output, batch.output.detach().numpy()[0], atol=2e-6, rtol=2e-6)

    def test_short_rollout_has_finite_nonzero_gradient(self):
        model=self.make_model().train(); samples=80
        reference=torch.sin(2*torch.pi*torch.arange(samples)/16)[None]*0.02
        disturbance=reference*0.2; path=model.secondary_paths[0].detach()
        result=rollout_phase3g_closed_loop(
            model, reference, disturbance, torch.stack((path,path))[None],
            torch.zeros((1,10), dtype=torch.long), torch.ones((1,10,2), dtype=torch.bool),
            torch.zeros((1,10), dtype=torch.long), torch.ones((1,10), dtype=torch.bool),
            truncate_blocks=0,
        )
        loss=result.output.square().mean(); loss.backward()
        gradients=[value.grad for value in model.parameters() if value.grad is not None]
        self.assertTrue(gradients)
        self.assertTrue(all(torch.isfinite(value).all() for value in gradients))
        self.assertGreater(sum(float(value.abs().sum()) for value in gradients), 0.0)

    def test_dtw_synthesis_is_deterministic_and_bounded(self):
        first=np.zeros(96); first[10]=1; first[20]=0.3
        second=np.zeros(96); second[12]=0.9; second[23]=0.2
        aligned_a=dtw_align_ir(first, second, radius=16)
        aligned_b=dtw_align_ir(first, second, radius=16)
        np.testing.assert_array_equal(aligned_a, aligned_b)
        generated, metadata=synthesize_path(first, second, mode="interpolate", amount=0.5)
        self.assertTrue(np.isfinite(generated).all())
        self.assertEqual(metadata["dtw_radius"], 16)


if __name__ == "__main__":
    unittest.main()
