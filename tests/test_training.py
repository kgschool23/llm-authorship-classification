"""Metric and selection checks for the training/evaluation pipeline."""
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
from torch.utils.data import DataLoader

from src.data import LABELS, TextDataset, collate_batch
from src.models import TextCNN
from src.text import Vocabulary, feature_texts
from src.training import EarlyStopping, epoch_pass, metrics_for, select_prompt_groups


def fixtures(n=5):
    return [dict(prompt_id=str(i), LLM_name=label, LLM_Input=f"prompt {i}", LLM_output=f"response {i}")
            for i in range(n) for label in LABELS]


class TrainingTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def test_majority_accuracy_and_macro_f1_are_different(self):
        metrics = metrics_for([0, 1, 2], [0, 0, 0])
        self.assertAlmostEqual(metrics["accuracy"], 1/3)
        self.assertAlmostEqual(metrics["macro_f1"], 1/6)
        self.assertEqual(metrics["confusion_matrix"], [[1,0,0],[1,0,0],[1,0,0]])

    def test_perfect_predictions_and_fixed_class_order(self):
        metrics = metrics_for([0, 1, 2], [0, 1, 2])
        self.assertEqual(metrics["accuracy"], 1.)
        self.assertEqual(metrics["macro_f1"], 1.)
        self.assertEqual(metrics["label_order"], list(LABELS))

    def test_subsampling_keeps_complete_prompt_groups(self):
        selected = select_prompt_groups(fixtures(), maximum=2, seed=42)
        self.assertEqual(len(selected), 6)
        self.assertEqual(len({r["prompt_id"] for r in selected}), 2)
        self.assertEqual(selected, select_prompt_groups(fixtures(), maximum=2, seed=42))
        with self.assertRaisesRegex(ValueError, "Incomplete"):
            select_prompt_groups(fixtures()[:-1], maximum=2)

    def test_early_stopping_ties_do_not_replace_best(self):
        stopping = EarlyStopping(patience=2)
        self.assertEqual(stopping.update(.5), (True, False))
        self.assertEqual(stopping.update(.5), (False, False))
        self.assertEqual(stopping.update(.4), (False, True))
        self.assertEqual(stopping.best, .5)

    def test_evaluation_leaves_parameters_unchanged(self):
        records = fixtures(2)
        vocabulary = Vocabulary.fit(feature_texts(records, "output_only"))
        dataset = TextDataset(records, vocabulary, "output_only", {"prompt_max_tokens": 10, "output_max_tokens": 10})
        loader = DataLoader(dataset, batch_size=4, collate_fn=collate_batch)
        model = TextCNN(len(vocabulary), embedding_dim=4, filters_per_kernel=2, dropout=0.)
        before = {name: value.detach().clone() for name, value in model.state_dict().items()}
        metrics, predictions = epoch_pass(model, loader, "cpu")
        self.assertEqual(metrics["samples"], len(records))
        self.assertEqual(len(predictions["probabilities"]), len(records))
        for name, value in model.state_dict().items():
            self.assertTrue(torch.equal(value, before[name]))
        for row in predictions["probabilities"]:
            self.assertAlmostEqual(sum(row), 1., places=6)


if __name__ == "__main__":
    unittest.main()
