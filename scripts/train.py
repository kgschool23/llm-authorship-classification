"""Train one experiment using training and validation data only."""
import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import argparse
import csv
import json
import platform
import re
import subprocess
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
from torch.utils.data import DataLoader

from src.data import LABELS, TextDataset, collate_batch, load_records
from src.models import build_model, parameter_count
from src.plots import plot_confusion, plot_history
from src.text import INPUT_MODES, Vocabulary, feature_texts, tokenize
from src.training import (EarlyStopping, epoch_pass, majority_baseline, save_predictions,
                          seed_everything, select_prompt_groups, sha256_file, write_json)


def text_statistics(records, dataset, vocabulary, config, mode):
    lengths = [len(sequence) for sequence in dataset.sequences]
    count = len(records)
    unknown = sum(int((sequence == 1).sum()) for sequence in dataset.sequences)
    stats = {"rows": count, "prompt_groups": len({r["prompt_id"] for r in records}),
             "family_counts": dict(Counter(r["LLM_name"] for r in records)),
             "mean_encoded_tokens": sum(lengths)/count, "max_encoded_tokens": max(lengths),
             "unknown_token_fraction": unknown/sum(lengths)}
    for field, budget in (("LLM_Input", "prompt_max_tokens"), ("LLM_output", "output_max_tokens")):
        if (field == "LLM_Input" and mode == "output_only") or (field == "LLM_output" and mode == "input_only"):
            continue
        stats[f"{field}_truncation_fraction"] = sum(
            len(tokenize(r[field], vocabulary.lowercase)) > config[budget] for r in records)/count
    return stats


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--architecture", choices=("cnn", "lstm"), required=True)
    parser.add_argument("--input-mode", choices=INPUT_MODES, required=True)
    parser.add_argument("--config", type=Path, default=ROOT/"configs/default.json")
    parser.add_argument("--data-dir", type=Path, default=ROOT/"data/processed")
    parser.add_argument("--results-dir", type=Path, default=ROOT/"results/runs")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--max-train-prompts", type=int, default=None)
    parser.add_argument("--max-validation-prompts", type=int, default=None)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    if args.epochs is not None:
        config["training"]["max_epochs"] = args.epochs
    if args.seed is not None:
        config["seed"] = args.seed
    if config["training"]["max_epochs"] < 1:
        parser.error("Epoch count must be positive")
    device = args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
    if device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA requested but unavailable")
    run_name = args.run_name or f"{args.architecture}_{args.input_mode}_seed{config['seed']}"
    if not re.fullmatch(r"[A-Za-z0-9_-]+", run_name):
        parser.error("Run name may contain only letters, digits, underscores, and hyphens")
    run_dir = args.results_dir/run_name
    if run_dir.exists():
        parser.error(f"Run folder already exists: {run_dir}. Choose a new --run-name.")
    seed_everything(config["seed"])
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    # No test path is opened or evaluated by this training script.
    train_path, validation_path = args.data_dir/"train.csv", args.data_dir/"validation.csv"
    train_records = select_prompt_groups(load_records(train_path), args.max_train_prompts, config["seed"])
    validation_records = select_prompt_groups(load_records(validation_path), args.max_validation_prompts, config["seed"])
    if {r["prompt_id"] for r in train_records} & {r["prompt_id"] for r in validation_records}:
        raise ValueError("Training/validation prompt leakage")
    vocabulary = Vocabulary.fit(feature_texts(train_records, args.input_mode), config["text"]["max_vocab_size"], config["text"]["lowercase"])
    train_dataset = TextDataset(train_records, vocabulary, args.input_mode, config["text"])
    validation_dataset = TextDataset(validation_records, vocabulary, args.input_mode, config["text"])
    generator = torch.Generator().manual_seed(config["seed"])
    loader_options = dict(batch_size=config["training"]["batch_size"], collate_fn=collate_batch,
                          num_workers=0, pin_memory=device == "cuda")
    train_loader = DataLoader(train_dataset, shuffle=True, generator=generator, **loader_options)
    validation_loader = DataLoader(validation_dataset, shuffle=False, **loader_options)
    model = build_model(args.architecture, len(vocabulary), config["model"]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["training"]["learning_rate"])
    stopper = EarlyStopping(config["training"]["early_stopping_patience"])
    run_dir.mkdir(parents=True, exist_ok=False)
    vocabulary.save(run_dir/"vocabulary.json")
    write_json(run_dir/"config.json", config)
    write_json(run_dir/"prompt_ids.json", {"train": sorted({r["prompt_id"] for r in train_records}),
                                         "validation": sorted({r["prompt_id"] for r in validation_records})})
    try:
        revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        revision = None
    source_paths = [ROOT/"scripts/train.py", *sorted((ROOT/"src").glob("*.py"))]
    manifest = {"status": "running", "started_at_utc": datetime.now(timezone.utc).isoformat(),
                "architecture": args.architecture, "input_mode": args.input_mode, "seed": config["seed"],
                "device": device, "gpu": torch.cuda.get_device_name(0) if device == "cuda" else None,
                "python": platform.python_version(), "torch": str(torch.__version__), "platform": platform.platform(),
                "git_revision": revision, "source_sha256": {str(p.relative_to(ROOT)): sha256_file(p) for p in source_paths},
                "dataset_sha256": {"train": sha256_file(train_path), "validation": sha256_file(validation_path)},
                "parameters": parameter_count(model), "vocabulary_size": len(vocabulary), "label_order": list(LABELS),
                "vocabulary_sha256": sha256_file(run_dir/"vocabulary.json"),
                "config_sha256": sha256_file(run_dir/"config.json"),
                "optimizer": "Adam", "gradient_clip_norm": 1.0, "precision": "float32",
                "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
                "subset_run": args.max_train_prompts is not None or args.max_validation_prompts is not None,
                "data_statistics": {"train": text_statistics(train_records, train_dataset, vocabulary, config["text"], args.input_mode),
                                    "validation": text_statistics(validation_records, validation_dataset, vocabulary, config["text"], args.input_mode)}}
    write_json(run_dir/"manifest.json", manifest)
    write_json(run_dir/"validation_baseline.json", majority_baseline(train_records, validation_records))
    print(f"Run: {run_name}; device={device}; train={len(train_records):,}; validation={len(validation_records):,}; parameters={parameter_count(model):,}", flush=True)
    history, best_epoch = [], None
    for epoch in range(1, config["training"]["max_epochs"]+1):
        epoch_started = time.perf_counter()
        train_metrics, _ = epoch_pass(model, train_loader, device, optimizer)
        validation_metrics, _ = epoch_pass(model, validation_loader, device)
        improved, stop = stopper.update(validation_metrics["macro_f1"])
        if improved:
            best_epoch = epoch
            torch.save({"model_state_dict": model.state_dict(), "epoch": epoch,
                        "validation_macro_f1": stopper.best, "architecture": args.architecture,
                        "input_mode": args.input_mode, "vocabulary_size": len(vocabulary), "label_order": list(LABELS)}, run_dir/"best.pt")
        row = {"epoch": epoch, "seconds": time.perf_counter()-epoch_started}
        for split, metrics in (("train", train_metrics), ("validation", validation_metrics)):
            for metric in ("loss", "accuracy", "macro_f1"):
                row[f"{split}_{metric}"] = metrics[metric]
        history.append(row)
        with (run_dir/"history.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(row))
            writer.writeheader()
            writer.writerows(history)
        print(f"Epoch {epoch:02d}: train loss={train_metrics['loss']:.4f}, F1={train_metrics['macro_f1']:.4f}; "
              f"validation loss={validation_metrics['loss']:.4f}, accuracy={validation_metrics['accuracy']:.4f}, "
              f"F1={validation_metrics['macro_f1']:.4f}; {row['seconds']:.1f}s{' [best]' if improved else ''}", flush=True)
        if stop:
            print(f"Early stopping after {stopper.patience} epochs without higher validation macro-F1.", flush=True)
            break
    checkpoint = torch.load(run_dir/"best.pt", map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    best_metrics, predictions = epoch_pass(model, validation_loader, device)
    write_json(run_dir/"validation_metrics.json", best_metrics)
    save_predictions(run_dir/"validation_predictions.csv", validation_records, predictions)
    plot_history(history, run_dir/"learning_curves.png")
    plot_confusion(best_metrics, run_dir/"validation_confusion_matrix.png", "Validation confusion matrix")
    manifest.update(status="completed", best_epoch=best_epoch, epochs_run=len(history),
                    elapsed_seconds=time.perf_counter()-started, best_validation_macro_f1=best_metrics["macro_f1"],
                    peak_gpu_memory_mib=torch.cuda.max_memory_allocated()/1024**2 if device == "cuda" else None)
    write_json(run_dir/"manifest.json", manifest)
    print(f"Completed. Best epoch={best_epoch}; validation accuracy={best_metrics['accuracy']:.4f}; macro-F1={best_metrics['macro_f1']:.4f}", flush=True)
    print(f"Artifacts: {run_dir}", flush=True)
    print("Test data was not evaluated.", flush=True)


if __name__ == "__main__":
    main()
