"""Check prediction pairing and summary counts independently of model training."""
import importlib.util
from pathlib import Path
import unittest

spec = importlib.util.spec_from_file_location("analysis_test", Path(__file__).resolve().parents[1]/"scripts/analyze_results.py")
analysis = importlib.util.module_from_spec(spec)
spec.loader.exec_module(analysis)


def row(prompt, author, prediction):
    return {"prompt_id": prompt, "true_family": author, "predicted_family": prediction}


class AnalysisTests(unittest.TestCase):
    def test_confusion_orientation_and_duplicate_rejection(self):
        rows = [row("a", "ChatGLM", "MPT"), row("a", "Flan-T5", "Flan-T5")]
        self.assertEqual(analysis.confusion_from_predictions(rows), [[0,0,1],[0,1,0],[0,0,0]])
        with self.assertRaises(ValueError):
            analysis.confusion_from_predictions(rows+[rows[0]])

    def test_paired_improvements_and_regressions(self):
        before = [row("a", "ChatGLM", "MPT"), row("b", "MPT", "MPT")]
        after = [row("b", "MPT", "ChatGLM"), row("a", "ChatGLM", "ChatGLM")]
        self.assertEqual(analysis.paired_changes(before, after), {"both_correct":0,"both_wrong":0,"helped":1,"harmed":1})
        with self.assertRaises(ValueError):
            analysis.paired_changes(before, after[:1])

    def test_fingerprint_newlines_and_empty_response(self):
        features = analysis.fingerprint("Hello,\r\nhello 12!")
        self.assertEqual(features["word_tokens"], 3)
        self.assertAlmostEqual(features["newlines_per_100_words"], 100/3)
        self.assertAlmostEqual(features["punctuation_per_100_words"], 200/3)
        self.assertEqual(analysis.fingerprint("")["word_tokens"], 0)
