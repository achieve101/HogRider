import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from legacy_models.phase2.phase2_validation import build_phase2_manifests, phase2_acceptance


class Phase2ValidationTests(unittest.TestCase):
    def test_split_manifests_do_not_leak_final_paths(self):
        manifests = build_phase2_manifests(ROOT / "dataset", seed=2026)
        self.assertEqual(manifests["development"]["path_indices_zero_based"], list(range(8)))
        self.assertEqual(manifests["final"]["path_indices_zero_based"], [8, 9])
        stress_bases = {item["base_path_index_zero_based"] for item in manifests["stress"]["variants"]}
        self.assertEqual(stress_bases, set(range(8)))
        self.assertTrue(stress_bases.isdisjoint({8, 9}))

    def test_acceptance_enforces_unseen_and_global_gates(self):
        def metrics(paths, primary, rebound):
            return {
                "primary_score_db": primary, "rebound_score_db": rebound,
                "selection_score": 0.7 * primary - 0.3 * rebound,
                "worst_path_primary_db": min(paths.values()),
                "controller_peak_abs": 0.1,
                "path_metrics": {str(key): {"primary_score_db": value, "rebound_score_db": rebound}
                                 for key, value in paths.items()},
            }
        baseline_dev = metrics({key: 5.0 for key in range(1, 9)}, 5.0, 1.0)
        baseline_final = metrics({9: 1.0, 10: 1.5}, 1.25, 1.0)
        candidate_dev = metrics({key: 5.1 for key in range(1, 9)}, 5.1, 1.0)
        candidate_final = metrics({9: 2.5, 10: 2.6}, 2.55, 1.0)
        accepted = phase2_acceptance(baseline_dev, baseline_final, candidate_dev, candidate_final)
        self.assertTrue(accepted["passed"])
        candidate_final["path_metrics"]["10"]["primary_score_db"] = 2.4
        rejected = phase2_acceptance(baseline_dev, baseline_final, candidate_dev, candidate_final)
        self.assertFalse(rejected["checks"]["path10_primary_at_least_2_5_db"])


if __name__ == "__main__":
    unittest.main()
