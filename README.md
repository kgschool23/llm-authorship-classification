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

Project scaffold only. No dataset has been collected and no experiments have
been run. Configuration values are initial choices, not tuned results.

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

Install the appropriate PyTorch build for the available hardware before training.
The default training dependency file is `requirements-training.txt`. A CUDA
installation command will be documented once the training hardware is confirmed.
Record resolved dependency versions with the actual experiment results.

## Reproducibility

Initial random seed: 42. Training and evaluation commands will be added as they
are implemented. Results and claims in the final report must come from recorded
experiments. Data provenance, licenses, collection settings, and limitations
will be documented in `data/README.md`.
