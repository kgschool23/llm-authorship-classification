"""Summarize all six required experiments without selecting by test scores."""
import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.experiments import COMBINATIONS, run_name, validate_comparison_rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=ROOT/"results/runs")
    parser.add_argument("--run-prefix", default="rq12")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, default=ROOT/"results/comparison/rq12")
    args = parser.parse_args()
    rows, details, hashes = [], [], set()
    for architecture, mode in COMBINATIONS:
        path = args.results_dir/run_name(args.run_prefix, architecture, mode, args.seed)
        manifest = json.loads((path/"manifest.json").read_text(encoding="utf-8"))
        if manifest["status"] != "completed" or manifest["subset_run"]:
            raise ValueError("Only completed full experiments belong in the comparison")
        if (manifest["architecture"], manifest["input_mode"], manifest["seed"]) != (architecture, mode, args.seed):
            raise ValueError("Run metadata does not match the requested experiment")
        validation = json.loads((path/"validation_metrics.json").read_text(encoding="utf-8"))
        test = json.loads((path/"test_metrics.json").read_text(encoding="utf-8"))
        evaluation = json.loads((path/"test_evaluation.json").read_text(encoding="utf-8"))
        hashes.add(evaluation["dataset_sha256"])
        rows.append({"architecture": architecture, "input_mode": mode, "seed": args.seed,
                     "train_samples": manifest["data_statistics"]["train"]["rows"],
                     "validation_samples": validation["samples"], "test_samples": test["samples"],
                     "best_epoch": manifest["best_epoch"], "epochs_run": manifest["epochs_run"],
                     "parameters": manifest["parameters"], "training_seconds": manifest["elapsed_seconds"],
                     "validation_accuracy": validation["accuracy"], "validation_macro_f1": validation["macro_f1"],
                     "test_accuracy": test["accuracy"], "test_macro_f1": test["macro_f1"]})
        details.append({"architecture": architecture, "input_mode": mode, "test_metrics": test,
                        "validation_metrics": validation, "run_directory": str(path)})
    validate_comparison_rows(rows)
    if len(hashes) != 1:
        raise ValueError("Experiments used different test files")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir/"results.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    table = ["| Model | Input | Best epoch | Validation accuracy | Test accuracy | Test macro-F1 |",
             "| --- | --- | ---: | ---: | ---: | ---: |"]
    for row in rows:
        table.append(f"| {row['architecture'].upper()} | {row['input_mode']} | {row['best_epoch']} | "
                     f"{row['validation_accuracy']:.2%} | {row['test_accuracy']:.2%} | {row['test_macro_f1']:.4f} |")
    deltas = []
    for architecture in ("cnn", "lstm"):
        by_mode = {row["input_mode"]: row for row in rows if row["architecture"] == architecture}
        deltas.append({"architecture": architecture,
                       "input_output_minus_output_only_accuracy_pp": 100*(by_mode["input_output"]["test_accuracy"]-by_mode["output_only"]["test_accuracy"]),
                       "input_output_minus_output_only_macro_f1": by_mode["input_output"]["test_macro_f1"]-by_mode["output_only"]["test_macro_f1"]})
    with (args.output_dir/"prompt_deltas.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(deltas[0]))
        writer.writeheader()
        writer.writerows(deltas)
    notes = ["", "Uniform-random expected accuracy: 33.33%.",
             "Balanced constant-class baseline: accuracy 33.33%, macro-F1 0.1667.",
             "Prompt-only accuracy is a matched-prompt control; expected value is exactly 33.33%.",
             "One seed was run. Differences are descriptive; no significance claim is made."]
    for row in rows:
        if row["input_mode"] == "input_only" and abs(row["test_accuracy"]-1/3) > 1e-8:
            notes.append(f"CHECK: {row['architecture']} prompt-only accuracy differs from the matched-prompt expectation.")
    (args.output_dir/"results.md").write_text("\n".join(table+notes)+"\n", encoding="utf-8")
    (args.output_dir/"results.json").write_text(json.dumps({"rows": rows, "prompt_deltas": deltas,
         "details": details, "test_dataset_sha256": next(iter(hashes)), "notes": notes}, indent=2)+"\n", encoding="utf-8")
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    modes, labels = ("input_only", "output_only", "input_output"), ("Prompt only", "Response only", "Prompt + response")
    x = np.arange(3)
    for metric, axis, title in zip(("test_accuracy", "test_macro_f1"), axes, ("Test accuracy", "Test macro-F1")):
        for offset, architecture, color in ((-.18, "cnn", "#2563eb"), (.18, "lstm", "#d97706")):
            values = [next(row[metric] for row in rows if row["architecture"] == architecture and row["input_mode"] == mode) for mode in modes]
            bars = axis.bar(x+offset, values, .36, label=architecture.upper(), color=color)
            axis.bar_label(bars, labels=[f"{value:.3f}" for value in values], padding=3, fontsize=9)
        baseline = 1/3 if metric == "test_accuracy" else 1/6
        axis.axhline(baseline, color="#64748b", linestyle="--", label="Constant-class baseline")
        axis.set(xticks=x, xticklabels=labels, ylim=(0, 1.08), title=title, ylabel="Score")
        axis.legend(fontsize=8)
        axis.grid(axis="y", alpha=.2)
    fig.savefig(args.output_dir/"comparison.png", dpi=180)
    plt.close(fig)
    print("\n".join(table))
    for delta in deltas:
        print(f"{delta['architecture'].upper()} prompt+response minus response-only: "
              f"{delta['input_output_minus_output_only_accuracy_pp']:+.2f} accuracy percentage points; "
              f"{delta['input_output_minus_output_only_macro_f1']:+.4f} macro-F1")
    print(f"Saved comparison artifacts: {args.output_dir}")


if __name__ == "__main__":
    main()
