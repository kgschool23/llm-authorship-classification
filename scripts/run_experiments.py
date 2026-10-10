"""Train all six fixed experiments, then test them and build a comparison."""
import argparse
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.experiments import COMBINATIONS, check_completed_run, run_name


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--config", type=Path, default=ROOT/"configs/default.json")
    parser.add_argument("--data-dir", type=Path, default=ROOT/"data/processed")
    parser.add_argument("--results-dir", type=Path, default=ROOT/"results/runs")
    parser.add_argument("--summary-dir", type=Path, default=ROOT/"results/comparison")
    parser.add_argument("--run-prefix", default="rq12")
    args = parser.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9_-]+", args.run_prefix):
        parser.error("Run prefix may contain only letters, digits, underscores, and hyphens")
    config = json.loads(args.config.read_text(encoding="utf-8"))
    if config["experiments"] != {"architectures": ["cnn", "lstm"], "input_modes": ["input_only", "output_only", "input_output"]}:
        parser.error("The suite requires the fixed CNN/LSTM x three-input experiment grid")
    # Hashes identify the reserved test version; no test metrics are computed yet.
    data_hashes = {split: sha256(args.data_dir/f"{split}.csv") for split in ("train", "validation", "test")}
    training_sources = [ROOT/"scripts/train.py", *sorted((ROOT/"src").glob("*.py"))]
    source_hashes = {str(path.relative_to(ROOT)): sha256(path) for path in training_sources}
    plan = {"config": config, "combinations": [list(pair) for pair in COMBINATIONS],
            "dataset_sha256": data_hashes, "training_source_sha256": source_hashes,
            "evaluation_script_sha256": sha256(ROOT/"scripts/evaluate.py"),
            "policy": "Train all six first; select by validation macro-F1; report all six tests without retuning",
            "run_prefix": args.run_prefix}
    output_dir = args.summary_dir/args.run_prefix
    output_dir.mkdir(parents=True, exist_ok=True)
    plan_path = output_dir/"experiment_plan.json"
    if plan_path.exists():
        if json.loads(plan_path.read_text(encoding="utf-8")) != plan:
            parser.error("Existing experiment plan differs. Preserve it and use a new --run-prefix.")
    else:
        plan_path.write_text(json.dumps(plan, indent=2)+"\n", encoding="utf-8")
    print("Phase 1/3: train all six experiments with fixed settings.", flush=True)
    for index, (architecture, mode) in enumerate(COMBINATIONS, 1):
        name = run_name(args.run_prefix, architecture, mode, config["seed"])
        path = args.results_dir/name
        if check_completed_run(path, architecture, mode, config, data_hashes, source_hashes):
            print(f"[{index}/6] Training already complete: {name}", flush=True)
            continue
        print(f"[{index}/6] Training: {name}", flush=True)
        subprocess.run([sys.executable, str(ROOT/"scripts/train.py"), "--architecture", architecture,
                        "--input-mode", mode, "--device", args.device, "--config", str(args.config),
                        "--data-dir", str(args.data_dir), "--results-dir", str(args.results_dir),
                        "--run-name", name], check=True, cwd=ROOT)
    print("Phase 2/3: evaluate the six selected checkpoints on the reserved test split.", flush=True)
    for index, (architecture, mode) in enumerate(COMBINATIONS, 1):
        name = run_name(args.run_prefix, architecture, mode, config["seed"])
        path = args.results_dir/name
        marker = path/"test_evaluation.json"
        if marker.exists():
            evaluation = json.loads(marker.read_text(encoding="utf-8"))
            if evaluation["dataset_sha256"] != data_hashes["test"] or evaluation["checkpoint_sha256"] != sha256(path/"best.pt"):
                raise ValueError("Existing test evaluation does not match its checkpoint/data")
            for name_to_check in ("test_metrics.json", "test_predictions.csv", "test_confusion_matrix.png", "test_baseline.json"):
                if not (path/name_to_check).exists():
                    raise ValueError(f"Missing test artifact: {path/name_to_check}")
            print(f"[{index}/6] Test already complete: {name}", flush=True)
            continue
        print(f"[{index}/6] Testing: {name}", flush=True)
        subprocess.run([sys.executable, str(ROOT/"scripts/evaluate.py"), "--run-dir", str(path),
                        "--split", "test", "--device", args.device, "--data-dir", str(args.data_dir)], check=True, cwd=ROOT)
    print("Phase 3/3: create the comparison table and figure.", flush=True)
    subprocess.run([sys.executable, str(ROOT/"scripts/summarize_results.py"), "--results-dir", str(args.results_dir),
                    "--run-prefix", args.run_prefix, "--seed", str(config["seed"]), "--output-dir", str(output_dir)],
                   check=True, cwd=ROOT)
    print(f"All six experiments completed. Comparison: {output_dir}", flush=True)


if __name__ == "__main__":
    main()
