# Preprocessing and model design

## Shared feature pipeline

Inputs are prompt-only, response-only, or prompt plus response. Only text fields
enter the models; source IDs, model IDs, domains, and decoding metadata do not.
A Unicode regex emits words and punctuation separately, with an explicit newline
token. Text is lowercased by default. This preserves punctuation, code fences,
and line structure, but does not preserve capitalization or exact whitespace.

Each input mode gets a vocabulary fitted on its training features only. Both
architectures use the same vocabulary and examples for that mode. The vocabulary
contains up to 20,000 tokens, including PAD, UNK, SEP, EMPTY, and NL. Equal-frequency
tokens are sorted alphabetically for deterministic token IDs. Validation/test
words not found in the fitted vocabulary become UNK.

Prompt and response budgets are 96 and 256 tokens. Combined features contain a
bounded prompt, SEP, and bounded response (up to 353 tokens). Separate budgets
prevent a long prompt from displacing the entire response. Dynamic batches use
right padding, and each example's true length is retained.

## Text CNN

Trainable 128-dimensional embedding -> parallel Conv1d kernels of width 3, 4,
and 5, with 128 filters each -> ReLU -> masked global max pooling -> concatenate
384 features -> dropout -> three output logits.

The convolution operates over token positions, with embedding coordinates as
channels. It can detect local patterns analogous to three-, four-, and five-token
phrases. Padding-only or partly padded windows beyond the true sequence boundary
are excluded from pooling. For inputs shorter than a kernel, a single
zero-padded window is retained. The PAD embedding stays zero and has no gradient.

## Bidirectional LSTM (the required RNN)

Trainable 128-dimensional embedding -> one bidirectional LSTM layer with 128
hidden units in each direction -> concatenate the final forward/backward hidden
states (256 features) -> dropout -> three output logits.

Packed sequences exclude padded time steps. The final states represent the
actual token sequence rather than its artificial padding. Bidirectionality is
appropriate because an entire response is available during classification.

Both architectures apply embedding and final-feature dropout with probability
0.3. They return raw logits for cross-entropy loss; softmax is not part of the
training forward pass. Label order is ChatGLM, Flan-T5, MPT.

## Verification

Unit tests cover unseen words, separate truncation budgets, metadata exclusion,
empty text, vocabulary serialization, short sequences, finite gradients,
zero padding gradients, right-padding invariance, and unsorted batch ordering.

The real-data smoke script runs one forward/backward/optimizer step for each
of the six architecture/input combinations using a training batch. It writes
`results/model_smoke_test.json`. These checks are not accuracy measurements or
completed training. Training and evaluation are implemented in the next stage.

## References

- PyTorch Conv1d: https://docs.pytorch.org/docs/2.9/generated/torch.nn.Conv1d.html
- PyTorch LSTM: https://docs.pytorch.org/docs/2.9/generated/torch.nn.LSTM.html
- Packed sequences: https://docs.pytorch.org/docs/2.9/generated/torch.nn.utils.rnn.pack_padded_sequence.html
