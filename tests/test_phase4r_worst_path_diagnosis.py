import unittest

import numpy as np

from phase4r_worst_path_diagnosis import (
    _require_path_set,
    decide_correction,
    oracle_sources_disjoint,
    render_blockwise_output,
    render_static_output,
    replay_sampling_metadata,
    validate_report_schema,
)


class Phase4RWorstPathDiagnosisTests(unittest.TestCase):
    def test_lopo_path_set_requires_physical_removal(self):
        self.assertEqual(_require_path_set(list(range(7)), 7), list(range(7)))
        with self.assertRaises(ValueError):
            _require_path_set(list(range(8)), 7)
        with self.assertRaises(ValueError):
            _require_path_set([0, 1, 2, 3, 4, 5, 8], 7)

    def test_oracle_source_split_is_disjoint(self):
        calibration = [{"source": "KTV.wav"}, {"source": "train.wav"}]
        development = [{"source_files": ["vehicle.wav"]}, {"source_files": ["restaurant.wav"]}]
        self.assertTrue(oracle_sources_disjoint(calibration, development))
        development.append({"source_files": ["KTV.wav"]})
        self.assertFalse(oracle_sources_disjoint(calibration, development))

    def test_sampler_replay_is_deterministic_and_uses_real_modes(self):
        arguments = {
            "train_paths": [0, 1, 2], "nearest": {0: 1, 1: 2, 2: 1},
            "seed": 2026, "epochs": 4, "samples_per_epoch": 100,
            "raw_frames": [2_000_000, 2_200_000], "path_length": 80,
            "augmentation_tail_start": 64,
        }
        first = replay_sampling_metadata(**arguments)
        second = replay_sampling_metadata(**arguments)
        self.assertEqual(first, second)
        self.assertEqual(set(first["mode_counts"]), {"measured", "interpolate", "extrapolate", "augment"})
        self.assertTrue(all("->" in key for key in first["directed_pair_counts_one_based"]))

    def test_static_oracle_enforces_simplex_latent_and_soft_limit(self):
        basis = np.ones((2, 3, 48), dtype=np.float32) * 10.0
        output = render_static_output(basis, np.array([0.25, 0.75]), np.array([0.5]))
        self.assertTrue(np.isfinite(output).all())
        self.assertLess(float(np.max(np.abs(output))), 0.98)
        with self.assertRaises(ValueError):
            render_static_output(basis, np.array([0.2, 0.2]), np.array([0.0]))
        with self.assertRaises(ValueError):
            render_static_output(basis, np.array([0.5, 0.5]), np.array([1.1]))

    def test_blockwise_oracle_enforces_each_simplex_row(self):
        basis = np.ones((1, 3, 480), dtype=np.float32)
        alpha = np.tile(np.array([0.4, 0.6]), (2, 1))
        latent = np.zeros((2, 1))
        output = render_blockwise_output(basis, alpha, latent)
        self.assertEqual(output.shape, (1, 480))
        alpha[1] = [0.4, 0.4]
        with self.assertRaises(ValueError):
            render_blockwise_output(basis, alpha, latent)

    def test_root_cause_priority_is_deterministic(self):
        evidence = {
            "free_fir_gate": True, "static_bank_gate": False,
            "blockwise_gate": True, "deployed_gap_to_static_bank_db": 3.0,
            "route_alpha_cosine_distance": 0.9, "memory_monotonic_gate": True,
            "cross_fold_coverage_support": True, "known_path_positive_control": True,
        }
        decision = decide_correction(evidence)
        self.assertEqual(decision["selected_root_cause"], "training_coverage_or_dictionary")
        self.assertTrue(decision["exactly_one_selected"])

    def test_report_schema_requires_eight_folds_and_frozen_correction(self):
        report = {
            "schema_version": 1, "phase": "4R", "status": "done", "frozen_control": {},
            "analysis_lanes": {}, "isolation_audit": [{} for _ in range(8)],
            "lopo_replay": {}, "coverage": {}, "oracle_hierarchy": {},
            "root_cause_attribution": {"exactly_one_selected": True},
            "selected_correction": {"status": "frozen_not_run"}, "acceptance": {},
        }
        validate_report_schema(report)
        report["isolation_audit"].pop()
        with self.assertRaises(ValueError):
            validate_report_schema(report)


if __name__ == "__main__":
    unittest.main()
