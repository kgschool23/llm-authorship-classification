"""Required experiment identities and checks for safely reusing completed runs."""
import json
from pathlib import Path

ARCHITECTURES = ("cnn", "lstm")
MODES = ("input_only", "output_only", "input_output")
COMBINATIONS = tuple((architecture, mode) for architecture in ARCHITECTURES for mode in MODES)


def run_name(prefix, architecture, mode, seed):
    return f"{prefix}_{architecture}_{mode}_seed{seed}"


def check_completed_run(run_dir, architecture, mode, config, dataset_hashes, source_hashes):
    run_dir = Path(run_dir)
    if not run_dir.exists():
        return False
    manifest_path = run_dir/"manifest.json"
    if not manifest_path.exists():
        raise ValueError(f"Incomplete run folder: {run_dir}. Preserve it and use a new --run-prefix.")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    saved_config = json.loads((run_dir/"config.json").read_text(encoding="utf-8"))
    if manifest.get("status") != "completed":
        raise ValueError(f"Interrupted run: {run_dir}. Preserve it and use a new --run-prefix.")
    if manifest.get("subset_run"):
        raise ValueError("A subset check cannot be reused as a full experiment")
    if (manifest["architecture"], manifest["input_mode"]) != (architecture, mode) or saved_config != config:
        raise ValueError(f"Run identity/settings differ: {run_dir}")
    if manifest["dataset_sha256"] != {k: dataset_hashes[k] for k in ("train", "validation")}:
        raise ValueError("Training/validation data changed")
    if manifest["source_sha256"] != source_hashes:
        raise ValueError("Training/model source changed")
    for name in ("best.pt", "vocabulary.json", "validation_metrics.json", "prompt_ids.json"):
        if not (run_dir/name).exists():
            raise ValueError(f"Missing saved artifact: {run_dir/name}")
    return True


def validate_comparison_rows(rows):
    pairs = [(row["architecture"], row["input_mode"]) for row in rows]
    if len(pairs) != len(COMBINATIONS) or set(pairs) != set(COMBINATIONS):
        raise ValueError("The comparison must contain each of the six required experiments exactly once")
    seeds = {row["seed"] for row in rows}
    sample_counts = {row["test_samples"] for row in rows}
    if len(seeds) != 1 or len(sample_counts) != 1:
        raise ValueError("Experiments must use the same seed and test sample count")
