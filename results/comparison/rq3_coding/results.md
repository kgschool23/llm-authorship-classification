# RQ3: transfer to held-out coding

| Model | Evaluation | Prompts | Responses | Accuracy | Macro-F1 |
| --- | --- | ---: | ---: | ---: | ---: |
| CNN | in_domain | 888 | 2664 | 96.40% | 0.9638 |
| CNN | held_domain | 75 | 225 | 96.89% | 0.9689 |
| LSTM | in_domain | 888 | 2664 | 95.87% | 0.9590 |
| LSTM | held_domain | 75 | 225 | 96.00% | 0.9602 |

CNN: held-domain minus in-domain accuracy = +0.49 percentage points.
LSTM: held-domain minus in-domain accuracy = +0.13 percentage points.

Separate response-only models were trained with the held domain absent from training and validation. Both test sets are disjoint from those new training/validation prompt groups. The held test pools that domain across the original splits; RQ1/RQ2 models and results are unchanged.

Domain labels are heuristic, the held-domain sample is small, and only one seed was used. The control set is a mixture of non-held domains. These results are exploratory, not proof of universal domain invariance or a statistically significant difference.
