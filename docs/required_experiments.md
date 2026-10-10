# RQ1 and RQ2 experiment plan

Run the fixed CNN/LSTM x prompt-only/response-only/prompt-plus-response grid.
Default settings are seed 42, Adam learning rate 0.001, batch size 32, maximum
20 epochs, and patience 3 for validation macro-F1. No hyperparameter search is
performed. The six settings are fixed before test evaluation.

## Run

```powershell
.\.venv\Scripts\python.exe scripts/run_experiments.py --device cuda
```

The runner has three phases:

1. Train all six models and select each checkpoint by validation macro-F1.
2. Evaluate all six selected checkpoints using the same reserved test split.
3. Create the comparison table, grouped bar chart, and prompt contribution deltas.

Run names are `rq12_<architecture>_<input_mode>_seed42`. The existing subset check
folders are not included. Each vocabulary is fitted on training features for
that input mode and is identical across the two architectures.

The runner writes an experiment plan with dataset, configuration, and source
identities. Repeating the command skips completed training/evaluation after
checking those identities. Interrupted training folders are preserved and cause
an explicit error; training cannot resume from optimizer/RNG state because the
saved checkpoint contains model weights only. To restart an interrupted suite,
choose a new `--run-prefix`; do not alter settings based on test scores.

## Outputs

`results/comparison/rq12/` contains:

- `experiment_plan.json`: fixed experiment definitions and integrity hashes.
- `results.csv`, `results.md`, `results.json`: all six validation/test outcomes.
- `comparison.png`: test accuracy and macro-F1 by input mode and architecture.
- `prompt_deltas.csv`: prompt-plus-response minus response-only differences.

Per-run checkpoints, histories, class metrics, predictions, and confusion matrices
remain in `results/runs/`. Checkpoints are excluded from Git. Prediction CSVs
contain labels, IDs, probabilities, and domain/source metadata; original text is
not redistributed.

## Interpretation

RQ1 uses the two response-only rows. RQ2 compares all three modes separately
for each architecture. Prompt-only is a matched-prompt control: every prompt has
one example per family, so a deterministic prompt-only prediction has 1/3
accuracy. Its macro-F1 can differ from 1/3 depending on prediction distribution.

A prompt-plus-response gain suggests contextual information helped on this
sample. A decrease is also a valid finding. Length/truncation, finite training
samples, and optimization may affect the difference. One seed is used in this
stage, so results are descriptive and are not evidence of statistical significance.
Do not tune using test scores or omit weaker rows.
