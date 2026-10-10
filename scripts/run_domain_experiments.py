"""RQ3: exclude one heuristic task domain, train both models, then test transfer."""
import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.experiments import check_completed_run

LABELS = ("ChatGLM", "Flan-T5", "MPT")


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, data):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(data, indent=2)+"\n", encoding="utf-8")


def read_csv(path):
    with Path(path).open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def partition(splits, domain):
    """Preserve non-held train/validation/test groups; pool held groups for test."""
    seen = set()
    output = {"train": [], "validation": [], "test": [], "in_domain_test": []}
    for split in ("train", "validation", "test"):
        groups = defaultdict(list)
        for row in splits[split]:
            groups[row["prompt_id"]].append(row)
        for prompt, rows in groups.items():
            if prompt in seen:
                raise ValueError("Prompt overlap in input splits")
            seen.add(prompt)
            if Counter(r["LLM_name"] for r in rows) != Counter(LABELS):
                raise ValueError("Expected one response from each family per prompt")
            if len({r["domain"] for r in rows}) != 1 or len({r["LLM_Input"] for r in rows}) != 1:
                raise ValueError("Prompt or domain differs within triplet")
            target = "test" if rows[0]["domain"] == domain else "in_domain_test" if split == "test" else split
            output[target].extend(rows)
    if any(not rows for rows in output.values()):
        raise ValueError("Each new split must be nonempty")
    if any(r["domain"] == domain for split in ("train", "validation", "in_domain_test") for r in output[split]):
        raise ValueError("Held domain leaked into a non-held split")
    return output


def write_dataset(path, rows):
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    payload = stream.getvalue().encode("utf-8")
    if path.exists() and path.read_bytes() != payload:
        raise ValueError(f"Existing domain dataset differs: {path}")
    if not path.exists():
        path.write_bytes(payload)


def evaluate(run, records_path, name, plan, device):
    import torch
    from torch.utils.data import DataLoader
    from src.data import TextDataset, collate_batch, load_records
    from src.models import build_model
    from src.plots import plot_confusion
    from src.text import Vocabulary
    from src.training import epoch_pass, save_predictions, seed_everything, metrics_for
    marker_path = run/f"{name}_evaluation.json"
    data_hash, checkpoint_hash = sha256(records_path), sha256(run/"best.pt")
    if marker_path.exists():
        marker = read_json(marker_path)
        if marker["dataset_sha256"] != data_hash or marker["checkpoint_sha256"] != checkpoint_hash:
            raise ValueError("Existing domain evaluation identities differ")
        for suffix in ("metrics.json", "predictions.csv", "confusion_matrix.png", "baseline.json"):
            if not (run/f"{name}_{suffix}").exists():
                raise ValueError("Existing domain evaluation is incomplete")
        return read_json(run/f"{name}_metrics.json")
    manifest = read_json(run/"manifest.json")
    for artifact in ("config", "vocabulary"):
        if sha256(run/f"{artifact}.json") != manifest[f"{artifact}_sha256"]:
            raise ValueError(f"Saved {artifact} changed")
    source_hashes = {k.replace("\\", "/"): v for k,v in manifest["source_sha256"].items()}
    for relative in ("src/models.py", "src/text.py", "src/data.py"):
        if sha256(ROOT/relative) != source_hashes[relative]:
            raise ValueError("Model/preprocessing source changed")
    config = read_json(run/"config.json")
    seed_everything(config["seed"])
    records = load_records(records_path)
    ids = read_json(run/"prompt_ids.json")
    if {r["prompt_id"] for r in records} & (set(ids["train"]) | set(ids["validation"])):
        raise ValueError("Domain test prompt overlaps training/validation")
    vocab = Vocabulary.load(run/"vocabulary.json")
    dataset = TextDataset(records, vocab, "output_only", config["text"])
    loader = DataLoader(dataset, batch_size=config["training"]["batch_size"], shuffle=False,
                        collate_fn=collate_batch, num_workers=0, pin_memory=device == "cuda")
    checkpoint = torch.load(run/"best.pt", map_location=device, weights_only=True)
    if checkpoint["label_order"] != list(LABELS) or checkpoint["vocabulary_size"] != len(vocab):
        raise ValueError("Checkpoint labels/vocabulary differ")
    model = build_model(manifest["architecture"], len(vocab), config["model"]).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    metrics, predictions = epoch_pass(model, loader, device)
    write_json(run/f"{name}_metrics.json", metrics)
    save_predictions(run/f"{name}_predictions.csv", records, predictions)
    plot_confusion(metrics, run/f"{name}_confusion_matrix.png", name.replace("_", " "))
    write_json(run/f"{name}_baseline.json", {"uniform_random_accuracy": 1/3,
               "constant_class_metrics": metrics_for(predictions["y_true"], [0]*len(records))})
    write_json(marker_path, {"dataset_sha256": data_hash, "checkpoint_sha256": checkpoint_hash,
                            "rows": len(records), "prompt_groups": len({r["prompt_id"] for r in records}),
                            "domain_plan_sha256": sha256(plan), "device": device,
                            "vocabulary_refitted": False, "weights_updated": False})
    print(f"{manifest['architecture']} / {name}: accuracy={metrics['accuracy']:.4f}; macro-F1={metrics['macro_f1']:.4f}", flush=True)
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--held-domain", choices=("coding", "math", "writing"), default="coding")
    parser.add_argument("--data-dir", type=Path, default=ROOT/"data/processed")
    parser.add_argument("--config", type=Path, default=ROOT/"configs/default.json")
    parser.add_argument("--results-dir", type=Path, default=ROOT/"results")
    args = parser.parse_args()
    config = read_json(args.config)
    prefix = f"rq3_{args.held_domain}"
    prepared = args.data_dir/f"domain_{args.held_domain}"
    summary_dir = args.results_dir/"comparison"/prefix
    prepared.mkdir(parents=True, exist_ok=True)
    summary_dir.mkdir(parents=True, exist_ok=True)
    splits = {s: read_csv(args.data_dir/f"{s}.csv") for s in ("train", "validation", "test")}
    original_hashes = {s: sha256(args.data_dir/f"{s}.csv") for s in splits}
    original_plan = read_json(args.results_dir/"comparison/rq12/experiment_plan.json")
    if original_hashes != original_plan["dataset_sha256"]:
        raise ValueError("Original CSVs differ from the RQ1/RQ2 dataset")
    grouped = partition(splits, args.held_domain)
    for split, rows in grouped.items():
        write_dataset(prepared/f"{split}.csv", rows)
    hashes = {s: sha256(prepared/f"{s}.csv") for s in grouped}
    source_files = [ROOT/"scripts/train.py", *sorted((ROOT/"src").glob("*.py"))]
    source_hashes = {str(p.relative_to(ROOT)): sha256(p) for p in source_files}
    summary = {s: {"responses": len(rows), "prompts": len({r["prompt_id"] for r in rows}),
                   "families": dict(Counter(r["LLM_name"] for r in rows)),
                   "domains": dict(Counter(r["domain"] for r in rows))} for s,rows in grouped.items()}
    plan = {"held_domain": args.held_domain, "input_mode": "output_only", "architectures": ["cnn", "lstm"],
            "config": config, "original_dataset_sha256": original_hashes, "dataset_sha256": hashes,
            "training_source_sha256": source_hashes, "runner_sha256": sha256(Path(__file__)),
            "splits": summary, "policy": "Exclude held domain from training and validation; train both models before scoring either test; no retuning",
            "domain_labels": "Existing keyword heuristics; no human domain validation",
            "relationship_to_rq12": "Separate models and split assignment; held-domain prompts pooled from original train/validation/test"}
    plan_path = summary_dir/"experiment_plan.json"
    if plan_path.exists() and read_json(plan_path) != plan:
        raise ValueError("Existing domain plan differs; preserve the recorded experiment")
    if not plan_path.exists():
        write_json(plan_path, plan)
    print("Domain split:", json.dumps(summary, indent=2), flush=True)
    for architecture in ("cnn", "lstm"):
        name = f"{prefix}_{architecture}_output_only_seed{config['seed']}"
        run = args.results_dir/"runs"/name
        if check_completed_run(run, architecture, "output_only", config, hashes, source_hashes):
            print(f"Training already complete: {name}", flush=True)
        else:
            subprocess.run([sys.executable, str(ROOT/"scripts/train.py"), "--architecture", architecture,
                            "--input-mode", "output_only", "--config", str(args.config), "--data-dir", str(prepared),
                            "--results-dir", str(args.results_dir/"runs"), "--run-name", name, "--device", args.device], check=True, cwd=ROOT)
    import torch
    device = args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
    if device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA requested but unavailable")
    results, per_class = [], []
    for architecture in ("cnn", "lstm"):
        run = args.results_dir/"runs"/f"{prefix}_{architecture}_output_only_seed{config['seed']}"
        for split, name in (("in_domain_test", "in_domain"), ("test", "held_domain")):
            metrics = evaluate(run, prepared/f"{split}.csv", name, plan_path, device)
            results.append({"architecture": architecture, "evaluation": name,
                            "prompts": summary[split]["prompts"], "responses": metrics["samples"],
                            "accuracy": metrics["accuracy"], "macro_f1": metrics["macro_f1"]})
            for family in LABELS:
                per_class.append({"architecture": architecture, "evaluation": name, "family": family,
                                  **metrics["classification_report"][family]})
    lines = ["# RQ3: transfer to held-out "+args.held_domain, "", "| Model | Evaluation | Prompts | Responses | Accuracy | Macro-F1 |",
             "| --- | --- | ---: | ---: | ---: | ---: |"]
    for r in results:
        lines.append(f"| {r['architecture'].upper()} | {r['evaluation']} | {r['prompts']} | {r['responses']} | {100*r['accuracy']:.2f}% | {r['macro_f1']:.4f} |")
    lines.append("")
    for architecture in ("cnn", "lstm"):
        normal = next(r for r in results if r["architecture"]==architecture and r["evaluation"]=="in_domain")
        held = next(r for r in results if r["architecture"]==architecture and r["evaluation"]=="held_domain")
        lines.append(f"{architecture.upper()}: held-domain minus in-domain accuracy = {100*(held['accuracy']-normal['accuracy']):+.2f} percentage points.")
    lines += ["", "Separate response-only models were trained with the held domain absent from training and validation. Both test sets are disjoint from those new training/validation prompt groups. The held test pools that domain across the original splits; RQ1/RQ2 models and results are unchanged.", "", "Domain labels are heuristic, the held-domain sample is small, and only one seed was used. The control set is a mixture of non-held domains. These results are exploratory, not proof of universal domain invariance or a statistically significant difference.", ""]
    (summary_dir/"results.md").write_text("\n".join(lines), encoding="utf-8")
    write_json(summary_dir/"results.json", {"rows": results, "per_class": per_class, "plan_sha256": sha256(plan_path)})
    for filename, rows in (("results.csv", results), ("per_class.csv", per_class)):
        with (summary_dir/filename).open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    print("\n".join(lines), flush=True)
    print(f"Saved: {summary_dir}")


if __name__ == "__main__":
    main()
