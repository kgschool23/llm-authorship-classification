"""Audit the six completed experiments and describe training-only fingerprints."""
import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path
import re
import statistics

ROOT = Path(__file__).resolve().parents[1]
LABELS = ("ChatGLM", "Flan-T5", "MPT")
MODES = ("input_only", "output_only", "input_output")
FEATURES = ("characters", "word_tokens", "newlines_per_100_words", "punctuation_per_100_words", "digit_percent", "unique_word_percent")


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_csv(path):
    with Path(path).open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_csv(path, rows):
    if not rows:
        raise ValueError("Cannot write an empty analysis table")
    with Path(path).open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def confusion_from_predictions(rows):
    matrix = [[0]*3 for _ in LABELS]
    seen = set()
    for row in rows:
        key = (row["prompt_id"], row["true_family"])
        if key in seen:
            raise ValueError("Duplicate prompt/family prediction")
        seen.add(key)
        matrix[LABELS.index(row["true_family"])][LABELS.index(row["predicted_family"])]+=1
    return matrix


def paired_changes(output, combined):
    def mapped(rows):
        result = {(r["prompt_id"], r["true_family"]): r for r in rows}
        if len(result) != len(rows):
            raise ValueError("Duplicate paired prediction")
        return result
    left, right = mapped(output), mapped(combined)
    if left.keys() != right.keys():
        raise ValueError("Paired predictions must have identical examples")
    counts = Counter()
    for key in left:
        a = left[key]["predicted_family"] == key[1]
        b = right[key]["predicted_family"] == key[1]
        counts["both_correct" if a and b else "helped" if b else "harmed" if a else "both_wrong"] += 1
    return {k: counts[k] for k in ("both_correct", "both_wrong", "helped", "harmed")}


def fingerprint(text):
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    words = re.findall(r"\w+", text.lower(), re.UNICODE)
    n = max(len(words), 1)
    return {
        "characters": len(text), "word_tokens": len(words),
        "newlines_per_100_words": 100*text.count("\n")/n,
        "punctuation_per_100_words": 100*len(re.findall(r"[^\w\s]", text))/n,
        "digit_percent": 100*sum(c.isdigit() for c in text)/max(len(text), 1),
        "unique_word_percent": 100*len(set(words))/n,
    }


def training_fingerprints(records):
    values = defaultdict(lambda: defaultdict(list))
    tokens = {family: Counter() for family in LABELS}
    document_counts = {family: Counter() for family in LABELS}
    for row in records:
        family = row["LLM_name"]
        for feature, value in fingerprint(row["LLM_output"]).items():
            values[family][feature].append(value)
        words = [w for w in re.findall(r"\w+", row["LLM_output"].lower())
                 if len(w) >= 3 and w.isalpha()]
        tokens[family].update(words)
        document_counts[family].update(set(words))
    summaries = []
    for family in LABELS:
        if not values[family]:
            raise ValueError(f"No training examples for {family}")
        for feature in FEATURES:
            items = values[family][feature]
            summaries.append({"family": family, "feature": feature, "responses": len(items),
                              "mean": statistics.mean(items), "median": statistics.median(items)})
    vocabulary = set().union(*tokens.values())
    distinctive = []
    for family in LABELS:
        other = sum((tokens[f] for f in LABELS if f != family), Counter())
        n, m, v = sum(tokens[family].values()), sum(other.values()), len(vocabulary)
        scores = [(math.log((count+1)/(n+v))-math.log((other[word]+1)/(m+v)), word)
                  for word, count in tokens[family].items() if count >= 25 and document_counts[family][word] >= 10]
        for rank, (score, word) in enumerate(sorted(scores, reverse=True)[:10], 1):
            distinctive.append({"family": family, "rank": rank, "token": word,
                                "family_count": tokens[family][word], "family_response_count": document_counts[family][word],
                                "other_count": other[word],
                                "smoothed_log_frequency_ratio": score})
    return summaries, distinctive


def plot_summary(class_rows, fingerprints, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    colors = ("#2563eb", "#7c3aed", "#d97706", "#16a34a")
    for j, (architecture, mode) in enumerate((("cnn", "output_only"), ("cnn", "input_output"), ("lstm", "output_only"), ("lstm", "input_output"))):
        rows = [r for r in class_rows if r["architecture"] == architecture and r["input_mode"] == mode]
        axes[0].bar([i+(j-1.5)*.2 for i in range(3)], [100*r["recall"] for r in rows], .2,
                    label=f"{architecture.upper()} {mode}", color=colors[j])
    axes[0].set(xticks=range(3), xticklabels=LABELS, ylim=(85, 100), ylabel="Test recall (%)", title="Recall by author (axis starts at 85%)")
    axes[0].legend(fontsize=8)
    medians = [next(r["median"] for r in fingerprints if r["family"] == f and r["feature"] == "word_tokens") for f in LABELS]
    axes[1].bar(LABELS, medians, color=("#2563eb", "#d97706", "#16a34a"))
    axes[1].set(ylabel="Median Unicode word tokens", title="Response length: training split only")
    for axis in axes:
        axis.grid(axis="y", alpha=.2)
        axis.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def analyze(results_dir, data_dir, output_dir):
    comparison = read_json(results_dir/"comparison/rq12/results.json")
    plan = read_json(results_dir/"comparison/rq12/experiment_plan.json")
    seed = plan["config"]["seed"]
    predictions, classes, subgroups, learning = {}, [], [], []
    identities = set()
    for row in comparison["rows"]:
        architecture, mode = row["architecture"], row["input_mode"]
        identity = (architecture, mode)
        if identity in identities:
            raise ValueError("Duplicate experiment")
        identities.add(identity)
        run = results_dir/"runs"/f"rq12_{architecture}_{mode}_seed{seed}"
        manifest, metrics = read_json(run/"manifest.json"), read_json(run/"test_metrics.json")
        marker = read_json(run/"test_evaluation.json")
        if manifest["status"] != "completed" or manifest["subset_run"] or manifest["seed"] != seed:
            raise ValueError("Expected a completed full run with the planned seed")
        if (manifest["architecture"], manifest["input_mode"]) != identity:
            raise ValueError("Run identity differs")
        if marker["dataset_sha256"] != plan["dataset_sha256"]["test"]:
            raise ValueError("Test dataset identity differs")
        pred = read_csv(run/"test_predictions.csv")
        matrix = confusion_from_predictions(pred)
        if matrix != metrics["confusion_matrix"] or len(pred) != row["test_samples"]:
            raise ValueError("Predictions disagree with recorded metrics/sample count")
        if not math.isclose(sum(matrix[i][i] for i in range(3))/len(pred), row["test_accuracy"], abs_tol=1e-12):
            raise ValueError("Comparison accuracy disagrees with predictions")
        if not math.isclose(metrics["macro_f1"], row["test_macro_f1"], abs_tol=1e-12):
            raise ValueError("Comparison macro-F1 disagrees with saved metrics")
        predictions[identity] = pred
        for family in LABELS:
            report = metrics["classification_report"][family]
            classes.append({"architecture": architecture, "input_mode": mode, "family": family,
                            **{k: report[k] for k in ("precision", "recall", "f1-score", "support")}})
        for grouping in ("domain", "source"):
            groups = defaultdict(list)
            for p in pred:
                value = p["domain"] if grouping == "domain" else p["source_id"].split("/", 1)[0]
                groups[value].append(p)
            for name, examples in sorted(groups.items()):
                correct = sum(p["true_family"] == p["predicted_family"] for p in examples)
                subgroups.append({"architecture": architecture, "input_mode": mode, "grouping": grouping,
                                  "group": name, "prompts": len({p["prompt_id"] for p in examples}),
                                  "responses": len(examples), "correct": correct, "accuracy": correct/len(examples)})
        history = read_csv(run/"history.csv")
        best = next(h for h in history if int(h["epoch"]) == manifest["best_epoch"])
        learning.append({"architecture": architecture, "input_mode": mode, "best_epoch": manifest["best_epoch"],
                         "epochs_run": manifest["epochs_run"], "train_accuracy_at_best": float(best["train_accuracy"]),
                         "validation_accuracy_at_best": float(best["validation_accuracy"]),
                         "validation_minus_test_accuracy_pp": 100*(row["validation_accuracy"]-row["test_accuracy"])})
    if identities != {(a, m) for a in ("cnn", "lstm") for m in MODES}:
        raise ValueError("Expected exactly six experiments")
    keys = [{(r["prompt_id"], r["true_family"]) for r in pred} for pred in predictions.values()]
    if any(k != keys[0] for k in keys):
        raise ValueError("Experiment test examples differ")
    paired = [{"architecture": a, **paired_changes(predictions[a, "output_only"], predictions[a, "input_output"])} for a in ("cnn", "lstm")]
    train_path = data_dir/"train.csv"
    if sha256(train_path) != plan["dataset_sha256"]["train"]:
        raise ValueError("Training CSV differs from experiment plan")
    train_records = read_csv(train_path)
    fingerprints, tokens = training_fingerprints(train_records)
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, rows in (("per_class.csv", classes), ("subgroups.csv", subgroups), ("learning_summary.csv", learning),
                       ("paired_prompt_changes.csv", paired), ("training_fingerprints.csv", fingerprints), ("distinctive_tokens.csv", tokens)):
        write_csv(output_dir/name, rows)
    plot_summary(classes, fingerprints, output_dir/"analysis.png")
    lines = ["# Experiment analysis", "", "All six test prediction files reproduce their recorded confusion matrices and accuracies.", "", "## RQ1: response-only classification", "", "| Model | Accuracy | Macro-F1 | Errors / 2,700 |", "| --- | ---: | ---: | ---: |"]
    for row in comparison["rows"]:
        if row["input_mode"] == "output_only":
            errors = sum(p["true_family"] != p["predicted_family"] for p in predictions[row["architecture"], "output_only"])
            lines.append(f"| {row['architecture'].upper()} | {100*row['test_accuracy']:.2f}% | {row['test_macro_f1']:.4f} | {errors} |")
    lines += ["", "## RQ2: adding the prompt", "", "| Model | Both correct | Both wrong | Helped | Harmed | Net accuracy change |", "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for row in paired:
        lines.append(f"| {row['architecture'].upper()} | {row['both_correct']} | {row['both_wrong']} | {row['helped']} | {row['harmed']} | {100*(row['helped']-row['harmed'])/len(keys[0]):+.2f} pp |")
    lines += ["", "Prompt-only inputs are identical within each three-author prompt group. A deterministic prediction therefore gets exactly one of three responses correct. Its macro-F1 depends on predicted-class frequencies and is not necessarily 0.3333.", "", "The LSTM combined model is the highest-scoring reported test configuration (97.67% accuracy, 0.9767 macro-F1). This is descriptive comparison across predeclared runs; no test-based retuning or significance claim is made.", "", "## Errors and learning curves", "", "| Configuration | Validation minus test accuracy |", "| --- | ---: |"]
    for row in learning:
        lines.append(f"| {row['architecture'].upper()} / {row['input_mode']} | {row['validation_minus_test_accuracy_pp']:+.2f} pp |")
    lines += ["", "Class-specific precision and recall are in `per_class.csv`. The confusion-matrix rows are true labels and columns predicted labels, in ChatGLM, Flan-T5, MPT order. Training curves describe optimization; checkpoint selection used validation macro-F1 only.", "", "## RQ4: descriptive response fingerprints", "", "These statistics use only the 12,600 training responses. They characterize author-associated style and vocabulary; they do not establish which features the CNN or LSTM uses.", "", "| Family | Median characters | Median word tokens | Median newlines / 100 words | Median punctuation / 100 words |", "| --- | ---: | ---: | ---: | ---: |"]
    def median(f, feature):
        return next(r["median"] for r in fingerprints if r["family"] == f and r["feature"] == feature)
    for family in LABELS:
        lines.append(f"| {family} | {median(family,'characters'):.1f} | {median(family,'word_tokens'):.1f} | {median(family,'newlines_per_100_words'):.2f} | {median(family,'punctuation_per_100_words'):.2f} |")
    lines += ["", "Distinctive tokens are ranked by additive-one-smoothed log frequency in one family versus the other two, requiring at least 25 occurrences across at least 10 of that family's training responses. They are exploratory vocabulary associations, not model attribution. Unicode `\\w+` units are tokenizer counts, not linguistic word counts; languages and response lengths affect them. Raw responses are not exported.", ""]
    for family in LABELS:
        lines.append(f"- {family}: "+", ".join(r["token"] for r in tokens if r["family"] == family))
    lines += ["", "## Scope and limitations", "", "- One seed, one dataset, three model families; no statistical significance or generalization to unseen model families is claimed.", "- Responses share prompts within a split. Response rows are dependent within prompt groups; splits are disjoint by normalized prompt.", "- Models received fixed token budgets, so longer responses were truncated as recorded in their manifests.", "- Domain labels are keyword heuristics and minority groups are small. `subgroups.csv` is descriptive evaluation on the existing test split. It is not a held-out-domain experiment and does not answer optional RQ3.", "- Fingerprint summaries describe the training set; they neither select checkpoints nor change the fixed six-run test results.", ""]
    (output_dir/"analysis.md").write_text("\n".join(lines), encoding="utf-8")
    audit = {"test_predictions_checked": 6, "training_fingerprint_rows": len(train_records),
             "training_dataset_sha256": sha256(train_path), "test_dataset_sha256": plan["dataset_sha256"]["test"],
             "analysis_script_sha256": sha256(Path(__file__)), "fingerprints_split": "train",
             "models_retrained": False, "new_test_inference": False, "paired_prompt_changes": paired}
    (output_dir/"analysis_audit.json").write_text(json.dumps(audit, indent=2)+"\n", encoding="utf-8")
    print((output_dir/"analysis.md").read_text(encoding="utf-8"))
    print(f"Saved analysis tables, figure, and audit: {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=ROOT/"results")
    parser.add_argument("--data-dir", type=Path, default=ROOT/"data/processed")
    parser.add_argument("--output-dir", type=Path, default=ROOT/"results/analysis")
    args = parser.parse_args()
    analyze(args.results_dir, args.data_dir, args.output_dir)
