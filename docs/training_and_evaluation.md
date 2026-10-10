# Training and evaluation protocol

## Default settings

| Setting | Value |
| --- | --- |
| Seed | 42 |
| Optimizer | Adam |
| Learning rate | 0.001 |
| Batch size | 32 |
| Maximum epochs | 20 |
| Early-stopping patience | 3 epochs |
| Selection criterion | Highest validation macro-F1 |
| Gradient clipping | Global norm 1.0 |
| Numerical precision | Float32 |
| Data-loader workers | 0 (Windows compatible) |

The training script reads only training and validation CSVs. Vocabulary is fitted
on the selected training examples only. Complete three-family prompt groups are
preserved when subsampling. Training/validation prompt overlap is rejected.

Random seeds are set for Python, NumPy, PyTorch, and CUDA. The training loader uses
a seeded generator. Deterministic algorithms are requested, cuDNN benchmarking
is disabled, and the cuBLAS workspace setting is established before importing
PyTorch. These controls do not promise identical results across hardware,
PyTorch releases, or platforms.

## Selection and early stopping

Evaluate validation examples after every training epoch with dropout disabled
and without gradient recording. Save a checkpoint only when validation macro-F1
strictly improves. Ties preserve the earlier checkpoint. Stop after three
consecutive nonimprovements, or after 20 epochs. Reload the best checkpoint
before saving final validation metrics and predictions.

Training loss and F1 in learning curves aggregate predictions made during
parameter updates. Validation scores are computed using the fixed model at the
end of each epoch. Because dropout is disabled for validation and the training
model changes during an epoch, training and validation curves are not directly
comparable as estimates of a fixed checkpoint's generalization gap.

Accuracy and macro-F1 are distinct. With balanced three-family classes, a
constant majority prediction has accuracy 1/3 and macro-F1 1/6. A uniformly
random prediction has expected accuracy 1/3. Macro-F1 is the unweighted mean of
per-family F1 scores. All metrics always include the same three labels, even if
one is never predicted.

## Final test evaluation

Use `scripts/evaluate.py` only after the experiment settings have been fixed.
It loads the saved vocabulary and selected checkpoint, verifies model and
preprocessing checksums, and computes predictions without fitting anything.
For test evaluation, it rejects overlap with training/validation prompt IDs.
It also rejects subset check runs as final test experiments. The script refuses
to overwrite an existing evaluation marker.

Test scores must not determine hyperparameters, stopping, model selection, or
which checkpoints to report. Report all six required experiments, including
prompt-only controls. Training is a separate command from test evaluation.

## Saved artifacts

Each run under `results/runs/<run-name>/` contains:

- `manifest.json`: settings, software/hardware, source and dataset checksums,
  data sizes, vocabulary statistics, runtime, and best epoch.
- `config.json`, `vocabulary.json`, and `prompt_ids.json`.
- `best.pt`: selected model state, excluded from Git.
- `history.csv` and `learning_curves.png`.
- `validation_metrics.json`, `validation_predictions.csv`, and confusion plot.
- `validation_baseline.json`.

The evaluation script creates corresponding split metrics, predictions, baseline,
confusion plot, and evaluation metadata. Predictions contain IDs, family labels,
and probabilities, not the original text.

A run directory must be new. To repeat an experiment, use a different run name.
Subset runs are explicitly marked in the manifest and are implementation checks,
not the main project results.

## Two-epoch GPU checks

```powershell
.\.venv\Scripts\python.exe scripts/train.py --architecture cnn --input-mode output_only --device cuda --epochs 2 --max-train-prompts 128 --max-validation-prompts 32 --run-name pipeline-check-cnn
.\.venv\Scripts\python.exe scripts/train.py --architecture lstm --input-mode output_only --device cuda --epochs 2 --max-train-prompts 64 --max-validation-prompts 16 --run-name pipeline-check-lstm
```

Full experiment orchestration and comparison tables are added in the next stage.

References:

- https://docs.pytorch.org/docs/2.9/notes/randomness.html
- https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html
