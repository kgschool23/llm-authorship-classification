"""Check all six architecture/input combinations on a real training batch."""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
from torch.utils.data import DataLoader

from src.data import TextDataset, collate_batch, load_records
from src.models import build_model, parameter_count
from src.text import Vocabulary, feature_texts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()
    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    config = json.loads((ROOT / "configs/default.json").read_text(encoding="utf-8"))
    records = load_records(ROOT / "data/processed/train.csv")
    print(f"Device: {device}; training rows: {len(records):,}", flush=True)
    results = []
    for mode in config["experiments"]["input_modes"]:
        vocabulary = Vocabulary.fit(feature_texts(records, mode), config["text"]["max_vocab_size"], config["text"]["lowercase"])
        dataset = TextDataset(records[:12], vocabulary, mode, config["text"])
        tokens, lengths, labels = next(iter(DataLoader(dataset, batch_size=12, collate_fn=collate_batch)))
        tokens, labels = tokens.to(device), labels.to(device)
        for architecture in config["experiments"]["architectures"]:
            torch.manual_seed(config["seed"])
            model = build_model(architecture, len(vocabulary), config["model"]).to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=config["training"]["learning_rate"])
            logits = model(tokens, lengths)
            if logits.shape != (len(labels), 3) or not torch.isfinite(logits).all():
                raise RuntimeError("Invalid logits")
            loss = torch.nn.functional.cross_entropy(logits, labels)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            for name, parameter in model.named_parameters():
                if parameter.grad is None or not torch.isfinite(parameter.grad).all():
                    raise RuntimeError(f"Invalid or missing gradient: {name}")
            if not torch.equal(model.embedding.weight.grad[0], torch.zeros_like(model.embedding.weight.grad[0])):
                raise RuntimeError("Padding embedding must not receive a gradient")
            optimizer.step()
            row = dict(architecture=architecture, input_mode=mode, vocabulary_size=len(vocabulary),
                       parameters=parameter_count(model), batch_size=len(labels), batch_width=tokens.size(1),
                       loss=float(loss.detach().cpu()), device=device, status="passed")
            results.append(row)
            print(f"PASS: {architecture:4s} / {mode:12s}; vocab={len(vocabulary):,}; parameters={row['parameters']:,}; loss={row['loss']:.4f}", flush=True)
    report = {"purpose": "Forward/backward smoke checks only; not classification results",
              "torch_version": torch.__version__, "gpu": torch.cuda.get_device_name(0) if device == "cuda" else None,
              "checks": results}
    path = ROOT / "results/model_smoke_test.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print("All six model/input combinations passed. No evaluation data was used.")


if __name__ == "__main__":
    main()
