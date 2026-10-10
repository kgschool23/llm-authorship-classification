"""Evaluate a saved best checkpoint without fitting vocabulary or model weights."""
import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
from torch.utils.data import DataLoader

from src.data import LABELS, LABEL_TO_ID, TextDataset, collate_batch, load_records
from src.models import build_model
from src.plots import plot_confusion
from src.text import Vocabulary
from src.training import (epoch_pass, metrics_for, save_predictions, seed_everything,
                          select_prompt_groups, sha256_file, write_json)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--split", choices=("validation", "test"), required=True)
    parser.add_argument("--data-dir", type=Path, default=ROOT/"data/processed")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()
    run_dir = args.run_dir
    manifest = json.loads((run_dir/"manifest.json").read_text(encoding="utf-8"))
    if manifest["status"] != "completed":
        parser.error("Training did not complete")
    if args.split == "test" and manifest["subset_run"]:
        parser.error("Subset smoke runs must not be used for final test results")
    if (run_dir/f"{args.split}_evaluation.json").exists():
        parser.error("This split was already evaluated by this script; preserve the existing results")
    for name in ("vocabulary", "config"):
        if sha256_file(run_dir/f"{name}.json") != manifest[f"{name}_sha256"]:
            raise ValueError(f"Saved {name} changed after training")
    for relative in ("src/models.py", "src/text.py", "src/data.py"):
        if sha256_file(ROOT/relative) != manifest["source_sha256"][relative]:
            raise ValueError(f"Model/preprocessing code changed: {relative}")
    config = json.loads((run_dir/"config.json").read_text(encoding="utf-8"))
    seed_everything(config["seed"])
    device = args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
    if device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA requested but unavailable")
    path = args.data_dir/f"{args.split}.csv"
    records = select_prompt_groups(load_records(path))
    training_ids = json.loads((run_dir/"prompt_ids.json").read_text(encoding="utf-8"))
    if args.split == "validation":
        if sha256_file(path) != manifest["dataset_sha256"]["validation"]:
            raise ValueError("Validation CSV changed after training")
        wanted = set(training_ids["validation"])
        records = [r for r in records if r["prompt_id"] in wanted]
    else:
        ids = {r["prompt_id"] for r in records}
        if ids & (set(training_ids["train"]) | set(training_ids["validation"])):
            raise ValueError("Test prompt overlap with training or validation")
    vocabulary = Vocabulary.load(run_dir/"vocabulary.json")
    dataset = TextDataset(records, vocabulary, manifest["input_mode"], config["text"])
    loader = DataLoader(dataset, batch_size=config["training"]["batch_size"], shuffle=False,
                        collate_fn=collate_batch, num_workers=0, pin_memory=device == "cuda")
    checkpoint = torch.load(run_dir/"best.pt", map_location=device, weights_only=True)
    if checkpoint["label_order"] != list(LABELS) or checkpoint["vocabulary_size"] != len(vocabulary):
        raise ValueError("Checkpoint schema does not match vocabulary/labels")
    model = build_model(manifest["architecture"], len(vocabulary), config["model"]).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    metrics, predictions = epoch_pass(model, loader, device)
    write_json(run_dir/f"{args.split}_metrics.json", metrics)
    save_predictions(run_dir/f"{args.split}_predictions.csv", records, predictions)
    plot_confusion(metrics, run_dir/f"{args.split}_confusion_matrix.png", f"{args.split.title()} confusion matrix")
    counts = manifest["data_statistics"]["train"]["family_counts"]
    majority = max(range(len(LABELS)), key=lambda i: (counts.get(LABELS[i], 0), -i))
    baseline = {"predicted_family": LABELS[majority], "uniform_random_expected_accuracy": 1/len(LABELS),
                "metrics": metrics_for(predictions["y_true"], [majority]*len(records))}
    write_json(run_dir/f"{args.split}_baseline.json", baseline)
    write_json(run_dir/f"{args.split}_evaluation.json", {
        "split": args.split, "rows": len(records), "device": device, "checkpoint_epoch": checkpoint["epoch"],
        "checkpoint_sha256": sha256_file(run_dir/"best.pt"), "dataset_sha256": sha256_file(path),
        "vocabulary_refitted": False, "weights_updated": False})
    print(f"{manifest['architecture']} / {manifest['input_mode']} / {args.split}: "
          f"accuracy={metrics['accuracy']:.4f}; macro-F1={metrics['macro_f1']:.4f}; rows={len(records):,}")
    print(f"Saved metrics, predictions, confusion matrix, and evaluation metadata in {run_dir}")


if __name__ == "__main__":
    main()
