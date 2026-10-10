# Reproduce the analysis

From the repository root:

```powershell
.\.venv\Scripts\python.exe scripts/analyze_results.py
```

This script reads saved test predictions and the training CSV. It does not load checkpoints, run new inference, train models, or change the six experiment results. It verifies test confusion matrices, accuracy, common example identities, and the training CSV checksum before writing aggregate tables to `results/analysis`.

The paired comparison counts examples that become correct or incorrect when the prompt is added. These are descriptive counts, not independent-sample significance tests. Each prompt has three dependent author responses.

RQ4 is addressed through descriptive training-only fingerprints: character and Unicode word-token counts, normalized line breaks and punctuation, digit percentage, lexical diversity, and author-associated token frequencies. Ratios use at least one token/character as denominator for empty responses. The token-frequency ranking uses additive-one smoothing against the shared vocabulary, with at least 25 family-specific occurrences across at least 10 responses. It includes no model feature attribution or causal claim. Language and length confound these statistics.

The subgroup table uses existing heuristic domain and source metadata. It is not a held-out-domain experiment; optional RQ3 remains unperformed. Minority domains are too small for strong conclusions.

Keep the six model training source files unchanged to preserve the experiment runner's source checksums. Analysis is a standalone script and does not add modules under `src`.
