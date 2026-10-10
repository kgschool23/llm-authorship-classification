"""Check required-grid completeness and completed-run reuse safeguards."""
import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.experiments import COMBINATIONS, check_completed_run, run_name, validate_comparison_rows


def rows():
    return [dict(architecture=a, input_mode=m, seed=42, test_samples=2700) for a,m in COMBINATIONS]


class ExperimentTests(unittest.TestCase):
    def test_exact_grid_required(self):
        validate_comparison_rows(rows())
        with self.assertRaises(ValueError):
            validate_comparison_rows(rows()[:-1])
        duplicate = rows()
        duplicate[-1] = duplicate[0]
        with self.assertRaises(ValueError):
            validate_comparison_rows(duplicate)

    def test_comparable_seeds_and_test_counts_required(self):
        for field in ("seed", "test_samples"):
            changed = rows()
            changed[0][field] += 1
            with self.assertRaises(ValueError):
                validate_comparison_rows(changed)

    def test_incomplete_run_is_not_silently_skipped(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"run"
            self.assertFalse(check_completed_run(path, "cnn", "input_only", {}, {}, {}))
            path.mkdir()
            with self.assertRaisesRegex(ValueError, "Incomplete"):
                check_completed_run(path, "cnn", "input_only", {}, {}, {})

    def test_run_names_distinguish_all_six_experiments(self):
        names = {run_name("rq12", a, m, 42) for a,m in COMBINATIONS}
        self.assertEqual(len(names), 6)
        self.assertIn("rq12_cnn_output_only_seed42", names)


if __name__ == "__main__":
    unittest.main()
