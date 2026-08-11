import unittest

from phase3g_validation import _record_rebound_score
from train_phase3g import should_stop_warmup


class Phase3GValidationTests(unittest.TestCase):
    def test_record_rebound_uses_participant_kit_field(self):
        record = {"third_octave_rebound_peak_1000_8000_db": 1.25}
        self.assertEqual(_record_rebound_score(record), 1.25)

    def test_record_rebound_does_not_require_aggregate_alias(self):
        record = {
            "third_octave_rebound_peak_1000_8000_db": 0.75,
            "rebound_score_db": 99.0,
        }
        self.assertEqual(_record_rebound_score(record), 0.75)

    def test_lopo_warmup_does_not_compare_seven_path_mean_to_full_baseline(self):
        self.assertFalse(should_stop_warmup("warmup", list(range(1, 8)), 10.0, 20.0))

    def test_full_warmup_guard_is_preserved(self):
        self.assertTrue(should_stop_warmup("warmup", list(range(8)), 19.0, 20.0))


if __name__ == "__main__":
    unittest.main()
