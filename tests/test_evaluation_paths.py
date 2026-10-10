"""Regression checks for Windows-generated training manifests."""
import importlib.util
from pathlib import Path
import unittest

PATH = Path(__file__).resolve().parents[1]/"scripts/evaluate.py"
spec = importlib.util.spec_from_file_location("evaluation_path_test", PATH)
evaluation = importlib.util.module_from_spec(spec)
spec.loader.exec_module(evaluation)

class EvaluationPathTests(unittest.TestCase):
    def test_windows_checksum_keys(self):
        manifest = {"source_sha256": {"src\\models.py": "model", "src\\text.py": "text"}}
        self.assertEqual(evaluation.source_checksum(manifest, "src/models.py"), "model")
        self.assertEqual(evaluation.source_checksum(manifest, "src/text.py"), "text")

    def test_posix_checksum_keys_and_missing_source(self):
        manifest = {"source_sha256": {"src/models.py": "model"}}
        self.assertEqual(evaluation.source_checksum(manifest, "src/models.py"), "model")
        with self.assertRaises(KeyError):
            evaluation.source_checksum(manifest, "src/data.py")
