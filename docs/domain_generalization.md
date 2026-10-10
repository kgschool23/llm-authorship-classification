# RQ3: held-out coding experiment

Run `python scripts/run_domain_experiments.py --device cuda` from the repository root.

This is a separate, predefined response-only experiment for CNN and bidirectional LSTM. It removes every heuristically labeled coding prompt from the original training and validation splits, pools the coding prompts across all original splits as a held-domain test, and preserves noncoding original test prompts as the control. The script checks complete balanced triplets and prompt disjointness. It trains both models before scoring either evaluation set, with the original fixed hyperparameters and validation macro-F1 checkpoint selection.

Expected sizes for this project's dataset:

| Split | Prompts | Responses |
| --- | ---: | ---: |
| Noncoding training | 4,144 | 12,432 |
| Noncoding validation | 893 | 2,679 |
| Noncoding control test | 888 | 2,664 |
| Coding held-domain test | 75 | 225 |

These runs have their own vocabularies fitted only to noncoding training responses. They do not reuse weights from RQ1/RQ2. The original models and results remain unchanged. Training remains governed by its existing source checksums; this patch adds no `src` files.

The held test includes prompts that were training examples in the earlier RQ1/RQ2 experiment. They are unseen by the new RQ3 models, which is the relevant leakage boundary. The coding domain was selected based on available sample size rather than test accuracy. No hyperparameter changes are selected using the new test results.

Report coding transfer and control metrics for both architectures, per-family recall/F1, and the sample sizes. A gap may reflect task-specific cues, response-length differences, vocabulary differences, or source composition. Domain labels are keyword heuristics, control domains are mixed, coding has only 75 independent prompt groups, and one seed limits conclusions. Optional RQ3 is exploratory domain transfer, not a claim that all fingerprints generalize.

The existing RQ4 analysis separately describes training-response length, formatting, vocabulary, and lexical diversity. It does not claim causal model attribution.
