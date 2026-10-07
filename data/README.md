# Dataset: three-family subset of MixInstruct

## Sources and attribution

MixInstruct was released by Dongfu Jiang, Xiang Ren, and Bill Yuchen Lin for
LLM-Blender (ACL 2023). Dataset revision:
`c7f77a4ef0515a99d1752c6387c22366d6922da6`.

- Dataset: https://huggingface.co/datasets/llm-blender/mix-instruct
- Paper: https://aclanthology.org/2023.acl-long.792/
- Code: https://github.com/yuchenlin/LLM-Blender

The dataset card labels the collection MIT. Its underlying prompt sources
retain their own terms. Our allowlist includes only `unified_chip2` (LAION
OIG; the publisher identifies volunteer-authored material as Apache 2.0) and
`dolly_15k` (Databricks Dolly-15K, CC BY-SA 3.0). We exclude ShareGPT,
InstructionWild/itwgpt4, and unclassified LAION records from the prepared data.

- OIG provenance: https://laion.ai/blog/oig-dataset/
- Dolly provenance: https://huggingface.co/datasets/databricks/databricks-dolly-15k

This repository distributes preparation code and aggregate metadata, not the
upstream prompts or responses. Raw and processed text files are ignored by Git.
Do not assume the collection's MIT label replaces underlying content licenses.

## Selected models

| Family label | Exact candidate model ID |
| --- | --- |
| ChatGLM | chatglm-6b |
| MPT | mpt-7b-instruct |
| Flan-T5 | flan-t5-xxl |

Use `candidates[].text` as the response and `candidates[].model` as its source.
The upstream top-level `output` is a reference answer, not a selected model's
response. Scores and pairwise judgments are never classifier inputs.

These are specific older checkpoints. Classification performance should not be
presented as a result about all versions of a family or today's proprietary LLMs.

## Construction and splitting

1. Download the pinned upstream validation and test JSONL files (about 148 MiB
   combined) and the original dataset card. Record SHA-256 checksums.
2. Pool those files as a sampling source. Their original names are provenance;
   this project defines new authorship train/validation/test splits.
3. Construct the prompt from nonempty `instruction` and `input`, joined by one
   newline. Preserve component text and candidate responses without rewriting.
4. Apply the source allowlist, require exactly one nonempty response from each
   selected model, and exclude complete groups matching email or SSN patterns.
   This simple pattern screen is not comprehensive anonymization.
5. Deduplicate prompts using Unicode NFKC, case folding, and collapsed whitespace.
   Keep the first complete group; use the normalized prompt's SHA-256 as ID.
   Normalization is for grouping, not replacement of the original feature text.
6. Shuffle with seed 42 and select up to 6,000 complete prompt groups.
7. Split groups 70/15/15, then expand them into three labeled response records.
   Each split has identical prompts for each family and balanced labels.

Expected with the default sample: 18,000 rows total; train 12,600,
validation 2,700, test 2,700. The actual counts and exclusions are recorded in
`results/dataset_summary.json`. No prompt ID can occur in multiple splits.

## Fields

`LLM_name`, `LLM_Input`, and `LLM_output` satisfy the assignment schema.
Additional fields: `prompt_id`, exact `model_id`, heuristic `domain`, `source_id`,
`upstream_split`, and upstream `decoding_method` when available.

Domain categories are ordered keyword heuristics: coding, math, writing, other.
They are approximate and imbalanced. Verify examples before a domain experiment;
small domains may not support reliable generalization claims. 'Other' is a
catch-all rather than a verified factual-QA category.

## Controls and limitations

A deterministic prompt-only classifier cannot exceed 1/3 accuracy on these
complete, balanced triplets: each identical prompt appears once per family.
Macro-F1 need not equal 1/3. This is a control for label leakage, not a failed
experiment. Metadata fields must never enter the classifier features.

Normalized response strings can recur across different prompts and splits,
especially common short answers. Their overlap is counted in the summary.
Model size, architecture, tuning, sampling, response length, truncation, and
self-identifying phrases can all contribute to apparent fingerprints.

## Reproduce (from repository root)

```powershell
.\.venv\Scripts\python.exe scripts/download_dataset.py
.\.venv\Scripts\python.exe scripts/prepare_dataset.py
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```
