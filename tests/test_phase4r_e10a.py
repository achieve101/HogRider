import argparse
import json
import unittest
from collections import Counter
from pathlib import Path

import numpy as np

from phase3g_data import (
    NEIGHBOR_POLICY_E10A,
    NEIGHBOR_POLICY_SINGLE,
    Phase3GSequenceDataset,
    build_phase3g_manifest,
)
from prepare_phase4r_e10a import build_protocol_closure
from train_phase3g import resolve_e10a_correction, validate_e10a_training_config


ROOT = Path(__file__).resolve().parents[1]
SPEC = ROOT / "artifacts/phase4r_preregistered_correction.json"


class Phase4RE10ATests(unittest.TestCase):
    def test_protocol_closure_is_reproducible_and_names_worst_folds(self):
        first = build_protocol_closure(ROOT / "dataset", ROOT / "artifacts/phase3r_innovation_templates.npz")
        second = build_protocol_closure(ROOT / "dataset", ROOT / "artifacts/phase3r_innovation_templates.npz")
        self.assertEqual(first, second)
        self.assertEqual(first["worst_three_coverage_folds"], [7, 1, 6])
        spec = json.loads(SPEC.read_text(encoding="utf-8"))
        self.assertEqual(first, spec["protocol_closure"])

    def test_every_fold_neighbor_table_is_retained_only(self):
        spec = json.loads(SPEC.read_text(encoding="utf-8"))
        tables = spec["protocol_closure"]["fold_neighbor_tables"]
        for held_text, table in tables.items():
            held = int(held_text)
            self.assertNotIn(str(held), table)
            for first, neighbors in table.items():
                self.assertEqual(len(neighbors), 3)
                self.assertEqual(len(set(neighbors)), 3)
                self.assertNotIn(int(first), neighbors)
                self.assertNotIn(held, neighbors)
                self.assertTrue(all(1 <= value <= 8 for value in neighbors))

    def test_default_neighbor_selection_is_bitwise_rng_compatible(self):
        dataset = Phase3GSequenceDataset.__new__(Phase3GSequenceDataset)
        dataset.neighbor_policy = NEIGHBOR_POLICY_SINGLE
        dataset.nearest = {0: 3}
        dataset.neighbor_table = None
        first = np.random.default_rng(2026)
        untouched = np.random.default_rng(2026)
        self.assertEqual(dataset._select_second(0, "interpolate", first), 3)
        self.assertEqual(first.random(), untouched.random())

    def test_e10a_draws_uniformly_only_for_synthetic_modes(self):
        dataset = Phase3GSequenceDataset.__new__(Phase3GSequenceDataset)
        dataset.neighbor_policy = NEIGHBOR_POLICY_E10A
        dataset.nearest = {0: 4}
        dataset.neighbor_table = {0: (1, 2, 3)}
        measured_rng = np.random.default_rng(17)
        untouched = np.random.default_rng(17)
        self.assertEqual(dataset._select_second(0, "measured", measured_rng), 4)
        self.assertEqual(measured_rng.random(), untouched.random())
        rng = np.random.default_rng(2026)
        counts = Counter(dataset._select_second(0, "interpolate", rng) for _ in range(6000))
        self.assertEqual(set(counts), {1, 2, 3})
        self.assertLess(max(counts.values()) - min(counts.values()), 250)

    def test_training_resolves_registered_fold_without_held_path(self):
        correction = resolve_e10a_correction(
            SPEC, ROOT / "dataset", ROOT / "artifacts/phase3r_innovation_templates.npz",
            list(range(7)),
        )
        self.assertEqual(correction["table_name"], "fold_8")
        self.assertEqual(set(correction["neighbor_table"]), set(range(7)))
        self.assertTrue(all(7 not in values for values in correction["neighbor_table"].values()))

    def test_registered_training_config_preserves_real_baseline_batching(self):
        correction = resolve_e10a_correction(
            SPEC, ROOT / "dataset", ROOT / "artifacts/phase3r_innovation_templates.npz",
            list(range(7)),
        )
        args = argparse.Namespace(
            stage="generalize", batch_size=8, gradient_accumulation=1,
            hidden_size=32, latent_size=16, generator_lr=3e-4, dictionary_lr=1e-5,
        )
        validate_e10a_training_config(correction, args, epochs=15, samples=128)
        args.batch_size = 1
        with self.assertRaises(ValueError):
            validate_e10a_training_config(correction, args, epochs=15, samples=128)

    def test_manifest_records_frozen_neighbors(self):
        table = {path: tuple(value for value in range(4) if value != path) for path in range(4)}
        manifest = build_phase3g_manifest(
            ROOT / "dataset", ["KTV.wav"], path_indices=list(range(4)),
            neighbor_policy=NEIGHBOR_POLICY_E10A, neighbor_table=table,
            correction_metadata={"experiment_id": "E10-A"},
        )
        self.assertEqual(manifest["neighbor_policy"], NEIGHBOR_POLICY_E10A)
        self.assertEqual(manifest["neighbor_table_one_based"]["1"], [2, 3, 4])
        self.assertEqual(manifest["correction"]["experiment_id"], "E10-A")
        self.assertFalse(manifest["sealed_paths_touched"])


if __name__ == "__main__":
    unittest.main()
