"""Build balanced, prompt-disjoint authorship splits from MixInstruct candidates."""
import argparse
import csv
import hashlib
import json
import random
import re
import unicodedata
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODELS = {"chatglm-6b": "ChatGLM", "mpt-7b-instruct": "MPT", "flan-t5-xxl": "Flan-T5"}
ALLOWED_SOURCES = {"unified_chip2", "dolly_15k"}
FIELDS = ["prompt_id", "LLM_name", "LLM_Input", "LLM_output", "model_id", "domain", "source_id", "upstream_split", "decoding_method"]
EMAIL = re.compile(r"\b[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}\b")
SSN = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")


def normalized(text):
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def prompt_text(item):
    instruction, context = item.get("instruction", ""), item.get("input", "")
    if not isinstance(instruction, str) or not isinstance(context, str):
        raise ValueError("Prompt components must be strings")
    return "\n".join(part for part in (instruction, context) if part.strip())


def domain(text):
    # Approximate task categories, not human-verified annotations.
    lower = text.lower()
    if re.search(r"\b(python|javascript|java|c\+\+|sql|programming|code|algorithm|function|debug)\b", lower):
        return "coding"
    if re.search(r"\b(equation|calculate|arithmetic|derivative|integral|probability|triangle|algebra|mathematics)\b", lower):
        return "math"
    if re.search(r"\b(story|poem|poetry|essay|creative|rewrite|paragraph|dialogue|fiction)\b", lower):
        return "writing"
    return "other"


def read_groups(raw):
    groups, seen, exclusions = [], set(), Counter()
    inventory, sources = Counter(), Counter()
    for upstream in ("val", "test"):
        path = raw / f"{upstream}_data_prepared.jsonl"
        with path.open(encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, 1):
                if not line.strip():
                    continue
                item = json.loads(line)
                candidates = item.get("candidates", [])
                if not isinstance(candidates, list):
                    raise ValueError(f"Invalid candidates: {path}:{line_number}")
                inventory.update(c.get("model", "<missing>") for c in candidates)
                source_id = str(item.get("id", f"{upstream}/{line_number}"))
                sources[source_id.split("/")[0]] += 1
                if source_id.split("/")[0] not in ALLOWED_SOURCES:
                    exclusions["source_not_allowlisted"] += 1
                    continue
                prompt = prompt_text(item)
                if not prompt.strip():
                    exclusions["empty_prompt"] += 1
                    continue
                selected = {model: [c for c in candidates if c.get("model") == model] for model in MODELS}
                if any(len(c) != 1 for c in selected.values()):
                    exclusions["missing_or_repeated_selected_model"] += 1
                    continue
                texts = [selected[m][0].get("text") for m in MODELS]
                if any(not isinstance(t, str) or not t.strip() for t in texts):
                    exclusions["empty_or_invalid_response"] += 1
                    continue
                if any(EMAIL.search(t) or SSN.search(t) for t in [prompt, *texts]):
                    exclusions["email_or_ssn_pattern"] += 1
                    continue
                key = normalized(prompt)
                if key in seen:
                    exclusions["duplicate_normalized_prompt"] += 1
                    continue
                seen.add(key)
                prompt_id = hashlib.sha256(key.encode("utf-8")).hexdigest()
                rows = []
                for model, family in MODELS.items():
                    candidate = selected[model][0]
                    rows.append(dict(prompt_id=prompt_id, LLM_name=family, LLM_Input=prompt,
                                     LLM_output=candidate["text"], model_id=model, domain=domain(prompt),
                                     source_id=source_id, upstream_split=upstream,
                                     decoding_method=candidate.get("decoding_method", "unknown")))
                groups.append(rows)
    return groups, dict(exclusions), dict(inventory), dict(sources)


def validate(splits):
    seen = set()
    for name, groups in splits.items():
        ids = {g[0]["prompt_id"] for g in groups}
        if seen.intersection(ids):
            raise ValueError(f"Prompt leakage into {name}")
        seen.update(ids)
        counts = Counter(row["LLM_name"] for g in groups for row in g)
        if set(counts) != set(MODELS.values()) or len(set(counts.values())) != 1:
            raise ValueError(f"Unbalanced or empty split: {name}")
        for group in groups:
            if (len(group) != 3 or len({r["LLM_Input"] for r in group}) != 1
                    or len({r["prompt_id"] for r in group}) != 1
                    or {r["LLM_name"] for r in group} != set(MODELS.values())):
                raise ValueError("Each prompt must have all three families")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, default=ROOT / "data/raw/mixinstruct")
    parser.add_argument("--output", type=Path, default=ROOT / "data/processed")
    parser.add_argument("--max-prompts", type=int, default=6000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.max_prompts < 20:
        parser.error("--max-prompts must be at least 20")
    groups, exclusions, inventory, sources = read_groups(args.raw)
    rng = random.Random(args.seed)
    rng.shuffle(groups)
    eligible = len(groups)
    groups = groups[:args.max_prompts]
    if len(groups) < 20:
        raise ValueError(f"Too few complete prompt groups: {len(groups)}. Models found: {inventory}")
    n_train, n_val = int(len(groups) * .70), int(len(groups) * .15)
    splits = {"train": groups[:n_train], "validation": groups[n_train:n_train+n_val], "test": groups[n_train+n_val:]}
    validate(splits)
    args.output.mkdir(parents=True, exist_ok=True)
    summary = {"seed": args.seed, "selected_models": MODELS, "eligible_prompts": eligible,
               "selected_prompts": len(groups), "exclusions": exclusions, "upstream_model_inventory": inventory,
               "upstream_source_counts": sources, "allowed_sources": sorted(ALLOWED_SOURCES),
               "selected_source_prompt_counts": dict(Counter(g[0]["source_id"].split("/")[0] for g in groups)),
               "split_policy": "Pooled upstream val/test; new 70/15/15 prompt-group split",
               "domain_policy": "Ordered keyword heuristic: coding, math, writing, other; approximate annotations",
               "prompt_only_accuracy_ceiling": 1/3, "splits": {}}
    for name, subset in splits.items():
        rows = [row for group in subset for row in group]
        rng.shuffle(rows)
        path = args.output / f"{name}.csv"
        with path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=FIELDS)
            writer.writeheader()
            writer.writerows(rows)
        summary["splits"][name] = {"prompts": len(subset), "rows": len(rows),
                                 "family_counts": dict(Counter(r["LLM_name"] for r in rows)),
                                 "domain_prompt_counts": dict(Counter(g[0]["domain"] for g in subset)),
                                 "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    # Count repeated response text across splits; common short answers are retained.
    response_sets = {name: {normalized(r["LLM_output"]) for g in subset for r in g} for name, subset in splits.items()}
    summary["shared_normalized_response_strings"] = {
        f"{a}_{b}": len(response_sets[a] & response_sets[b])
        for a,b in (("train","validation"),("train","test"),("validation","test"))}
    download_manifest = args.raw / "download_manifest.json"
    if download_manifest.exists():
        summary["download"] = json.loads(download_manifest.read_text(encoding="utf-8"))
    summary_path = ROOT / "results/dataset_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"Selected {len(groups):,} shared prompts; {len(groups)*3:,} response records.")
    for name, details in summary["splits"].items():
        print(f"{name}: {details['rows']:,} rows; families={details['family_counts']}; domains={details['domain_prompt_counts']}")
    print(f"Exclusions: {exclusions}")
    print("PASS: balanced classes, complete triplets, and no normalized prompt overlap.")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
