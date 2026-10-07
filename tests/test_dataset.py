"""Small fixtures test the safeguards without downloading real data."""
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/prepare_dataset.py"
spec = importlib.util.spec_from_file_location("prepare_dataset", SCRIPT)
prep = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prep)


def record(prompt, missing=False, empty=False):
    candidates = [{"model": m, "text": "" if empty and m == "chatglm-6b" else "A response."}
                  for m in prep.MODELS if not (missing and m == "chatglm-6b")]
    return {"id": "unified_chip2/fixture", "instruction": "", "input": prompt, "candidates": candidates}


class DatasetTests(unittest.TestCase):
    def read(self, records):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "val_data_prepared.jsonl").write_text(
                "\n".join(json.dumps(r) for r in records), encoding="utf-8")
            (root / "test_data_prepared.jsonl").write_text("", encoding="utf-8")
            return prep.read_groups(root)

    def test_normalized_duplicates_removed_as_complete_groups(self):
        groups, exclusions, _, _ = self.read([record("A prompt"), record("  a   PROMPT ")])
        self.assertEqual(len(groups), 1)
        self.assertEqual(len(groups[0]), 3)
        self.assertEqual(exclusions["duplicate_normalized_prompt"], 1)

    def test_incomplete_triplets_and_identifier_patterns_rejected(self):
        groups, exclusions, _, _ = self.read([
            record("Missing", missing=True), record("Empty", empty=True),
            record("Contact person@example.com")])
        self.assertEqual(groups, [])
        self.assertEqual(sum(exclusions.values()), 3)

    def test_non_allowlisted_source_rejected(self):
        item = record("A prompt")
        item["id"] = "sharegpt/fixture"
        groups, exclusions, _, _ = self.read([item])
        self.assertEqual(groups, [])
        self.assertEqual(exclusions["source_not_allowlisted"], 1)

    def test_overlapping_prompt_groups_fail(self):
        groups, _, _, _ = self.read([record("A prompt")])
        with self.assertRaisesRegex(ValueError, "leakage"):
            prep.validate({"train": groups, "test": groups})

    def test_prompt_components_preserved(self):
        self.assertEqual(prep.prompt_text({"instruction": "Explain.", "input": "Context."}), "Explain.\nContext.")


if __name__ == "__main__":
    unittest.main()
