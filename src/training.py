"""Training utilities, prompt-group sampling, and fixed-label evaluation."""
import csv
import hashlib
import json
import random
from collections import Counter, defaultdict
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

from .data import LABELS, LABEL_TO_ID


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def select_prompt_groups(records, maximum=None, seed=42):
    groups = defaultdict(list)
    for record in records:
        groups[record["prompt_id"]].append(record)
    for prompt_id, group in groups.items():
        if len(group) != len(LABELS) or {r["LLM_name"] for r in group} != set(LABELS):
            raise ValueError(f"Incomplete prompt group: {prompt_id}")
        if len({r["LLM_Input"] for r in group}) != 1:
            raise ValueError(f"Prompt group contains different text: {prompt_id}")
    if maximum is None:
        return records
    if maximum < 1:
        raise ValueError("Prompt limit must be positive")
    ids = sorted(groups)
    random.Random(seed).shuffle(ids)
    selected = set(ids[:maximum])
    return [record for record in records if record["prompt_id"] in selected]


def metrics_for(y_true, y_pred, loss=None):
    if not y_true or len(y_true) != len(y_pred):
        raise ValueError("Need nonempty, equal-length targets and predictions")
    label_ids = list(range(len(LABELS)))
    if any(label not in label_ids for label in [*y_true, *y_pred]):
        raise ValueError("Unknown numeric class ID")
    metrics = {
        "samples": len(y_true), "label_order": list(LABELS),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=label_ids, average="macro", zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=label_ids).tolist(),
        "classification_report": classification_report(y_true, y_pred, labels=label_ids,
                                                      target_names=list(LABELS), output_dict=True, zero_division=0),
    }
    if loss is not None:
        metrics["loss"] = float(loss)
    return metrics


def majority_baseline(training_records, evaluation_records):
    counts = Counter(LABEL_TO_ID[r["LLM_name"]] for r in training_records)
    majority = max(range(len(LABELS)), key=lambda i: (counts[i], -i))
    truth = [LABEL_TO_ID[r["LLM_name"]] for r in evaluation_records]
    return {"predicted_family": LABELS[majority],
            "metrics": metrics_for(truth, [majority] * len(truth)),
            "uniform_random_expected_accuracy": 1 / len(LABELS)}


def epoch_pass(model, loader, device, optimizer=None, gradient_clip=1.):
    training = optimizer is not None
    model.train(training)
    loss_sum, y_true, y_pred, probabilities = 0., [], [], []
    context = nullcontext() if training else torch.inference_mode()
    with context:
        for tokens, lengths, labels in loader:
            tokens = tokens.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            if training:
                optimizer.zero_grad(set_to_none=True)
            logits = model(tokens, lengths)
            loss = torch.nn.functional.cross_entropy(logits, labels)
            if not torch.isfinite(loss):
                raise RuntimeError("Nonfinite loss")
            if training:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip, error_if_nonfinite=True)
                optimizer.step()
            loss_sum += float(loss.detach().cpu()) * len(labels)
            y_true.extend(labels.detach().cpu().tolist())
            y_pred.extend(logits.detach().argmax(dim=1).cpu().tolist())
            if not training:
                probabilities.extend(logits.detach().softmax(dim=1).cpu().tolist())
    return metrics_for(y_true, y_pred, loss_sum / len(y_true)), {
        "y_true": y_true, "y_pred": y_pred, "probabilities": probabilities}


def save_predictions(path, records, predictions):
    if len(records) != len(predictions["y_pred"]) or len(records) != len(predictions["probabilities"]):
        raise ValueError("Prediction count does not match records")
    fields = ["prompt_id", "true_family", "predicted_family", "domain", "source_id"]
    fields += [f"probability_{label}" for label in LABELS]
    with Path(path).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for record, truth, prediction, probabilities in zip(
                records, predictions["y_true"], predictions["y_pred"], predictions["probabilities"]):
            if LABEL_TO_ID[record["LLM_name"]] != truth:
                raise ValueError("Prediction order differs from record order")
            row = dict(prompt_id=record["prompt_id"], true_family=LABELS[truth], predicted_family=LABELS[prediction],
                       domain=record.get("domain", "unknown"), source_id=record.get("source_id", "unknown"))
            row.update({f"probability_{label}": float(probabilities[i]) for i, label in enumerate(LABELS)})
            writer.writerow(row)


class EarlyStopping:
    def __init__(self, patience=3):
        if patience < 1:
            raise ValueError("Patience must be positive")
        self.patience = patience
        self.best = float("-inf")
        self.bad_epochs = 0

    def update(self, score):
        improved = score > self.best
        if improved:
            self.best, self.bad_epochs = score, 0
        else:
            self.bad_epochs += 1
        return improved, self.bad_epochs >= self.patience
