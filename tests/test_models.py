"""Check vocabulary leakage, feature construction, and sequence-model padding."""
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch

from src.data import TextDataset, collate_batch
from src.models import TextCNN, TextLSTM
from src.text import EMPTY_ID, SEP_ID, UNK_ID, Vocabulary, encode_record, feature_texts, tokenize


class TextTests(unittest.TestCase):
    def test_unseen_validation_token_remains_unknown(self):
        vocabulary = Vocabulary.fit(["training words"])
        self.assertEqual(vocabulary.encode("validationonly", 10), [UNK_ID])
        self.assertNotIn("validationonly", vocabulary.token_to_id)

    def test_punctuation_and_linebreak_preserved(self):
        self.assertEqual(tokenize("Hi!\n```"), ["hi", "!", "<NL>", "`", "`", "`"])

    def test_combined_input_has_separate_budgets(self):
        record = {"LLM_Input": "one two three four", "LLM_output": "five six seven eight"}
        vocabulary = Vocabulary.fit(feature_texts([record], "input_output"))
        encoded = encode_record(record, vocabulary, "input_output", {"prompt_max_tokens": 2, "output_max_tokens": 3})
        self.assertEqual(len(encoded), 6)
        self.assertEqual(encoded[2], SEP_ID)
        self.assertEqual(encoded[3:], vocabulary.encode(record["LLM_output"], 3))

    def test_empty_sequence_and_vocabulary_roundtrip(self):
        vocabulary = Vocabulary.fit(["some text"])
        self.assertEqual(vocabulary.encode("  ", 10), [EMPTY_ID])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "vocab.json"
            vocabulary.save(path)
            self.assertEqual(Vocabulary.load(path).tokens, vocabulary.tokens)

    def test_metadata_cannot_change_features(self):
        record = {"LLM_name": "ChatGLM", "LLM_Input": "prompt", "LLM_output": "response", "source_id": "one"}
        vocabulary = Vocabulary.fit(feature_texts([record], "output_only"))
        config = {"prompt_max_tokens": 10, "output_max_tokens": 10}
        first = TextDataset([record], vocabulary, "output_only", config)
        changed = dict(record, source_id="two", model_id="MPT", domain="coding")
        second = TextDataset([changed], vocabulary, "output_only", config)
        self.assertTrue(torch.equal(first[0][0], second[0][0]))


class ModelTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        torch.set_num_threads(1)

    def models(self):
        return [TextCNN(30, embedding_dim=8, filters_per_kernel=4, dropout=0.),
                TextLSTM(30, embedding_dim=8, hidden_dim=4, dropout=0.)]

    def test_short_sequences_finite_logits_and_gradients(self):
        tokens, lengths, labels = collate_batch([(torch.tensor([5]), 0), (torch.tensor([6, 7]), 1)])
        for model in self.models():
            with self.subTest(model=type(model).__name__):
                logits = model(tokens, lengths)
                self.assertEqual(tuple(logits.shape), (2, 3))
                self.assertTrue(torch.isfinite(logits).all())
                torch.nn.functional.cross_entropy(logits, labels).backward()
                for parameter in model.parameters():
                    self.assertIsNotNone(parameter.grad)
                    self.assertTrue(torch.isfinite(parameter.grad).all())
                self.assertTrue(torch.equal(model.embedding.weight.grad[0], torch.zeros(8)))

    def test_extra_right_padding_does_not_change_prediction(self):
        tokens = torch.tensor([[5, 6, 7, 8, 9], [6, 7, 0, 0, 0]])
        lengths = torch.tensor([5, 2])
        padded = torch.nn.functional.pad(tokens, (0, 7), value=0)
        for model in self.models():
            model.eval()
            with self.subTest(model=type(model).__name__), torch.no_grad():
                torch.testing.assert_close(model(tokens, lengths), model(padded, lengths), atol=1e-6, rtol=1e-5)

    def test_batch_order_preserved_for_unsorted_lengths(self):
        tokens = torch.tensor([[5, 0, 0, 0, 0], [5, 6, 7, 8, 9], [8, 9, 0, 0, 0]])
        lengths = torch.tensor([1, 5, 2])
        permutation = torch.tensor([2, 0, 1])
        for model in self.models():
            model.eval()
            with self.subTest(model=type(model).__name__), torch.no_grad():
                original = model(tokens, lengths)
                permuted = model(tokens[permutation], lengths[permutation])
                torch.testing.assert_close(original[permutation], permuted, atol=1e-6, rtol=1e-5)


if __name__ == "__main__":
    unittest.main()
