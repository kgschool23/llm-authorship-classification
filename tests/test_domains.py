"""RQ3 leakage checks on synthetic matched triplets."""
import importlib.util
from pathlib import Path
import unittest
spec = importlib.util.spec_from_file_location("domain_test", Path(__file__).resolve().parents[1]/"scripts/run_domain_experiments.py")
domain = importlib.util.module_from_spec(spec);spec.loader.exec_module(domain)

def triplet(prompt, task):
    return [{"prompt_id":prompt,"domain":task,"LLM_Input":prompt,"LLM_name":label,"LLM_output":"example"} for label in domain.LABELS]

class DomainTests(unittest.TestCase):
    def fixture(self):
        return {s:triplet(s+"-coding","coding")+triplet(s+"-other","other") for s in ("train","validation","test")}

    def test_held_domain_absent_from_train_and_validation(self):
        result = domain.partition(self.fixture(), "coding")
        self.assertEqual(len(result["test"]),9)
        self.assertEqual({r["prompt_id"] for r in result["test"]}, {"train-coding","validation-coding","test-coding"})
        for split in ("train","validation","in_domain_test"):
            self.assertTrue(all(r["domain"] != "coding" for r in result[split]))
        sets = [{r["prompt_id"] for r in rows} for rows in result.values()]
        self.assertTrue(all(not a&b for i,a in enumerate(sets) for b in sets[i+1:]))

    def test_overlap_and_incomplete_groups_rejected(self):
        fixture = self.fixture();fixture["test"] += fixture["train"][:3]
        with self.assertRaises(ValueError):domain.partition(fixture,"coding")
        fixture = self.fixture();fixture["train"].pop()
        with self.assertRaises(ValueError):domain.partition(fixture,"coding")

    def test_inconsistent_domain_rejected(self):
        fixture=self.fixture();fixture["train"][0]["domain"]="other"
        with self.assertRaises(ValueError):domain.partition(fixture,"coding")
