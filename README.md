# Who Wrote It? Identifying LLMs from Their Responses

CMPSC 448 midterm project, Penn State University, Fall 2026.

**Author and solo team leader:** Kyle Goodman  
**Repository:** https://github.com/kgschool23/llm-authorship-classification

## Goal

Compare a text CNN and an LSTM recurrent neural network for identifying the
LLM family that generated a response. The dataset will include at least three
LLM families with documented provenance and reasonably balanced classes.

## Research questions

- RQ1: Can response text identify its generating LLM family?
- RQ2: How do prompt-only, response-only, and prompt-plus-response inputs compare?
- RQ3 (extra credit): Do fingerprints generalize to held-out task domains?
- RQ4 (extra credit): Which linguistic or structural features distinguish families?

## Current status

Dataset preparation, preprocessing, CNN, LSTM, training, and evaluation are
implemented. Unit tests and short training runs verify the pipeline.
Full experiments and final classification results are pending.

## Planned evaluation

For each of CNN and LSTM, train separate prompt-only, response-only, and
prompt-plus-response classifiers. Report accuracy, macro-F1, per-class metrics,
and confusion matrices, alongside a chance/majority baseline.

Prefer shared prompts across families. Keep all examples with the same prompt
in a single split, including normalized prompt duplicates. Fit preprocessing
and vocabulary using training data only. Select models using validation data;
reserve test data for final evaluation. Document any missing prompt-family pairs.

## Layout

- `configs/`: initial experiment settings.
- `data/`: dataset documentation, raw records, and processed splits.
- `src/`: preprocessing, models, training, and evaluation code.
- `scripts/`: command-line utilities.
- `results/`: metrics and figures; large checkpoints are excluded from Git.
- `reports/`: final report PDF and editable source.

## Environment setup (Windows PowerShell)

```powershell
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe scripts/check_environment.py
```

Verified local setup: Windows 11, Python 3.14.4, NVIDIA RTX 4070 SUPER,
PyTorch 2.9.1+cu128. GPU forward computation and backpropagation passed.

```powershell
.\.venv\Scripts\python.exe -m pip install "torch==2.9.1" --index-url https://download.pytorch.org/whl/cu128
.\.venv\Scripts\python.exe scripts/download_dataset.py
.\.venv\Scripts\python.exe scripts/prepare_dataset.py
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

See `data/README.md` for provenance, source filters, labels, and split policy.
The scripts produce 18,000 records under the default settings, with all responses
to a prompt kept in the same split. Raw and processed text remain local.
Record resolved dependency versions with the actual experiment results.

## Reproducibility

Initial random seed: 42. Training and evaluation commands will be added as they
are implemented. Results and claims in the final report must come from recorded
experiments. Data provenance, licenses, collection settings, and limitations
will be documented in `data/README.md`.

## Model verification

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe scripts/smoke_test_models.py --device cuda
```

Read `docs/model_design.md` for the architecture, feature budgets, and padding
policy. The smoke check fits vocabularies on training features only and performs
one gradient step per combination; it does not evaluate validation/test data.

## Training and evaluation

See `docs/training_and_evaluation.md` for optimizer settings, early stopping,
metrics, and the test protocol. A short implementation check can be run with:

```powershell
.\.venv\Scripts\python.exe scripts/train.py --architecture cnn --input-mode output_only --device cuda --epochs 2 --max-train-prompts 128 --max-validation-prompts 32 --run-name pipeline-check-cnn
```

Training never evaluates test data. Best checkpoints are selected by validation
macro-F1. Each run saves its settings, source checksums, vocabulary, checkpoint,
learning curves, validation metrics, baseline, and predictions. Subset checks
are labeled in the manifest and cannot be evaluated as final test runs.
