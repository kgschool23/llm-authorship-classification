"""CSV loading and variable-length PyTorch batches without metadata features."""
import csv
from pathlib import Path

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

from .text import PAD_ID, encode_record

LABELS = ("ChatGLM", "Flan-T5", "MPT")
LABEL_TO_ID = {label: i for i, label in enumerate(LABELS)}


def load_records(path):
    with Path(path).open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        required = {"LLM_name", "LLM_Input", "LLM_output", "prompt_id"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f"Missing CSV columns: {required-set(reader.fieldnames or [])}")
        records = list(reader)
    if not records:
        raise ValueError(f"Empty dataset: {path}")
    if any(record["LLM_name"] not in LABEL_TO_ID for record in records):
        raise ValueError("Unknown family label")
    return records


class TextDataset(Dataset):
    def __init__(self, records, vocabulary, mode, text_config):
        self.records = records
        self.sequences = [torch.tensor(encode_record(record, vocabulary, mode, text_config), dtype=torch.long)
                          for record in records]
        self.labels = [LABEL_TO_ID[record["LLM_name"]] for record in records]

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, index):
        return self.sequences[index], self.labels[index]


def collate_batch(batch):
    sequences, labels = zip(*batch)
    lengths = torch.tensor([len(sequence) for sequence in sequences], dtype=torch.long)
    tokens = pad_sequence(sequences, batch_first=True, padding_value=PAD_ID)
    return tokens, lengths, torch.tensor(labels, dtype=torch.long)
