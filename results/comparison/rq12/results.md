| Model | Input | Best epoch | Validation accuracy | Test accuracy | Test macro-F1 |
| --- | --- | ---: | ---: | ---: | ---: |
| CNN | input_only | 5 | 33.33% | 33.33% | 0.3273 |
| CNN | output_only | 14 | 97.26% | 96.04% | 0.9605 |
| CNN | input_output | 2 | 96.52% | 95.85% | 0.9584 |
| LSTM | input_only | 2 | 33.33% | 33.33% | 0.3148 |
| LSTM | output_only | 6 | 97.15% | 95.93% | 0.9594 |
| LSTM | input_output | 8 | 97.67% | 97.67% | 0.9767 |

Uniform-random expected accuracy: 33.33%.
Balanced constant-class baseline: accuracy 33.33%, macro-F1 0.1667.
Prompt-only accuracy is a matched-prompt control; expected value is exactly 33.33%.
One seed was run. Differences are descriptive; no significance claim is made.
