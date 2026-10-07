# Dataset documentation

Status: not collected yet.

Required record fields:

- `LLM_name`: generating LLM family (classification label).
- `LLM_Input`: exact user prompt.
- `LLM_output`: exact model response.

Additional metadata to retain when available:

- `prompt_id`, `domain`, exact model identifier, source, collection date,
  generation parameters, and usage/license information.

Before training, document sources and redistribution rights, family counts,
domain counts, deduplication, missing records, and the prompt-grouped split.
Raw data is ignored by Git until its redistribution rights have been checked.
Do not insert simulated responses as if they were real LLM outputs.
