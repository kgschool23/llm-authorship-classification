"""Apply the evaluator path-format repair to an unevaluated experiment plan."""
import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.experiments import COMBINATIONS, check_completed_run, run_name

OLD_HASH = "12f6f2152c6cc0303b3e6ad9e7b4fd67428773d860d0b978dd6a8edad82e275d"
NEW_HASH = "41ee3f8dda8a6967d32b297fa9fa399e779f8aadf82694e36bd442d3dcb05508"

def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def repair(root):
    plan_path = root/"results/comparison/rq12/experiment_plan.json"
    audit_path = plan_path.with_name("evaluation_portability_fix.json")
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    if sha256(root/"scripts/evaluate.py") != NEW_HASH:
        raise ValueError("Expected the supplied corrected evaluator; extract the fix first")
    if plan["evaluation_script_sha256"] == NEW_HASH and audit_path.exists():
        print("Evaluator repair already applied. Continue with run_experiments.py.")
        return
    if plan["evaluation_script_sha256"] != OLD_HASH:
        raise ValueError("Unexpected original evaluator checksum; experiment plan was not changed")
    current_sources = {str(p.relative_to(root)): sha256(p)
                       for p in [root/"scripts/train.py", *sorted((root/"src").glob("*.py"))]}
    if current_sources != plan["training_source_sha256"]:
        raise ValueError("Training sources changed; experiment plan was not changed")
    current_data = {split: sha256(root/f"data/processed/{split}.csv")
                    for split in ("train", "validation", "test")}
    if current_data != plan["dataset_sha256"]:
        raise ValueError("Dataset changed; experiment plan was not changed")
    for architecture, mode in COMBINATIONS:
        run = root/"results/runs"/run_name("rq12", architecture, mode, plan["config"]["seed"])
        if (run/"test_evaluation.json").exists():
            raise ValueError("Test evaluation already exists; experiment plan was not changed")
        if not check_completed_run(run, architecture, mode, plan["config"], current_data, current_sources):
            raise ValueError(f"Training not complete: {run}")
    audit = {"reason": "Normalize Windows checksum path keys before first test evaluation",
             "old_evaluation_script_sha256": OLD_HASH, "new_evaluation_script_sha256": NEW_HASH,
             "utc": datetime.now(timezone.utc).isoformat(),
             "training_sources_changed": False, "datasets_changed": False,
             "test_evaluated_before_repair": False}
    audit_path.write_text(json.dumps(audit, indent=2)+"\n", encoding="utf-8")
    plan["evaluation_script_sha256"] = NEW_HASH
    temporary = plan_path.with_suffix(".tmp")
    temporary.write_text(json.dumps(plan, indent=2)+"\n", encoding="utf-8")
    temporary.replace(plan_path)
    print("Repaired evaluator checksum in the plan. All six trained checkpoints preserved.")

if __name__ == "__main__":
    repair(ROOT)
