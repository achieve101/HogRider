import hashlib
import json
import sys
import unittest
from pathlib import Path
from unittest import mock

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from legacy_models.submissions.phase3g_submission_final_seed2027_v2.submission import (
    create_model as create_v2,
)
from phase3g_submission_final_seed2027_v3.submission import create_model as create_v3
import phase3g_submission_final_seed2027_v3.runtime as v3_runtime


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class Phase4RRuntimeTests(unittest.TestCase):
    def setUp(self):
        self.v2 = create_v2("cpu")
        self.v3 = create_v3("cpu")

    @staticmethod
    def stream_outputs(model, samples=5000):
        indices = np.arange(samples, dtype=np.float64)
        reference = 0.02 * np.sin(2 * np.pi * indices / 127.0)
        previous_error = 0.01 * np.cos(2 * np.pi * indices / 83.0)
        model.reset()
        return np.asarray([
            model.process_sample(float(x_t), float(e_t))
            for x_t, e_t in zip(reference, previous_error)
        ])

    def test_v3_is_bit_exact_to_frozen_v2_past_first_route_update(self):
        baseline = self.stream_outputs(self.v2)
        candidate = self.stream_outputs(self.v3)
        np.testing.assert_array_equal(candidate, baseline)

    def test_v3_reset_is_bit_exact(self):
        first = self.stream_outputs(self.v3)
        second = self.stream_outputs(self.v3)
        np.testing.assert_array_equal(second, first)

    def test_v3_wrapper_does_not_enter_inference_mode_per_sample(self):
        self.v3.reset()
        with mock.patch.object(
            v3_runtime.torch,
            "inference_mode",
            side_effect=AssertionError("per-sample inference context reintroduced"),
        ):
            for index in range(32):
                output = self.v3.process_sample(index * 1e-5, 0.0)
                self.assertTrue(np.isfinite(output))

    def test_v3_keeps_weights_and_config_portable(self):
        v2_weights = (
            ROOT
            / "legacy_models"
            / "submissions"
            / "phase3g_submission_final_seed2027_v2"
            / "weights.pt"
        )
        v3_weights = ROOT / "phase3g_submission_final_seed2027_v3" / "weights.pt"
        self.assertEqual(file_sha256(v2_weights), file_sha256(v3_weights))
        config_path = ROOT / "phase3g_submission_final_seed2027_v3" / "config.json"
        config = json.loads(config_path.read_text(encoding="utf-8"))
        self.assertEqual(config["weights_file"], "weights.pt")
        self.assertNotIn("source_checkpoint", config)


if __name__ == "__main__":
    unittest.main()
