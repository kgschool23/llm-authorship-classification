# Experiment analysis

All six test prediction files reproduce their recorded confusion matrices and accuracies.

## RQ1: response-only classification

| Model | Accuracy | Macro-F1 | Errors / 2,700 |
| --- | ---: | ---: | ---: |
| CNN | 96.04% | 0.9605 | 107 |
| LSTM | 95.93% | 0.9594 | 110 |

## RQ2: adding the prompt

| Model | Both correct | Both wrong | Helped | Harmed | Net accuracy change |
| --- | ---: | ---: | ---: | ---: | ---: |
| CNN | 2540 | 59 | 48 | 53 | -0.19 pp |
| LSTM | 2562 | 35 | 75 | 28 | +1.74 pp |

Prompt-only inputs are identical within each three-author prompt group. A deterministic prediction therefore gets exactly one of three responses correct. Its macro-F1 depends on predicted-class frequencies and is not necessarily 0.3333.

The LSTM combined model is the highest-scoring reported test configuration (97.67% accuracy, 0.9767 macro-F1). This is descriptive comparison across predeclared runs; no test-based retuning or significance claim is made.

## Errors and learning curves

| Configuration | Validation minus test accuracy |
| --- | ---: |
| CNN / input_only | +0.00 pp |
| CNN / output_only | +1.22 pp |
| CNN / input_output | +0.67 pp |
| LSTM / input_only | +0.00 pp |
| LSTM / output_only | +1.22 pp |
| LSTM / input_output | +0.00 pp |

Class-specific precision and recall are in `per_class.csv`. The confusion-matrix rows are true labels and columns predicted labels, in ChatGLM, Flan-T5, MPT order. Training curves describe optimization; checkpoint selection used validation macro-F1 only.

## RQ4: descriptive response fingerprints

These statistics use only the 12,600 training responses. They characterize author-associated style and vocabulary; they do not establish which features the CNN or LSTM uses.

| Family | Median characters | Median word tokens | Median newlines / 100 words | Median punctuation / 100 words |
| --- | ---: | ---: | ---: | ---: |
| ChatGLM | 856.5 | 150.0 | 3.55 | 16.36 |
| Flan-T5 | 36.0 | 7.0 | 0.00 | 6.45 |
| MPT | 1113.0 | 208.0 | 2.39 | 15.05 |

Distinctive tokens are ranked by additive-one-smoothed log frequency in one family versus the other two, requiring at least 25 occurrences across at least 10 of that family's training responses. They are exploratory vocabulary associations, not model attribution. Unicode `\w+` units are tokenizer counts, not linguistic word counts; languages and response lengths affect them. Raw responses are not exported.

- ChatGLM: casting, tranquility, align, rustle, ultimately, rustling, noting, bustling, chirping, additionally
- Flan-T5: len, yes, click, inside, input, outside, town, blue, scene, fish
- MPT: printf, char, stdio, scanf, nearly, funny, edited, director, cin, shouldn

## Scope and limitations

- One seed, one dataset, three model families; no statistical significance or generalization to unseen model families is claimed.
- Responses share prompts within a split. Response rows are dependent within prompt groups; splits are disjoint by normalized prompt.
- Models received fixed token budgets, so longer responses were truncated as recorded in their manifests.
- Domain labels are keyword heuristics and minority groups are small. `subgroups.csv` is descriptive evaluation on the existing test split. It is not a held-out-domain experiment and does not answer optional RQ3.
- Fingerprint summaries describe the training set; they neither select checkpoints nor change the fixed six-run test results.
