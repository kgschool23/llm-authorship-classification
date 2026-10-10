"""Text CNN and bidirectional LSTM; both return unnormalized class logits."""
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pack_padded_sequence

from .text import PAD_ID


class TextCNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim=128, filters_per_kernel=128,
                 kernel_sizes=(3, 4, 5), dropout=.3, num_classes=3):
        super().__init__()
        if not kernel_sizes or min(kernel_sizes) < 1:
            raise ValueError("Kernel sizes must be positive")
        self.kernel_sizes = tuple(kernel_sizes)
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=PAD_ID)
        self.dropout = nn.Dropout(dropout)
        self.convolutions = nn.ModuleList(nn.Conv1d(embedding_dim, filters_per_kernel, k) for k in kernel_sizes)
        self.classifier = nn.Linear(filters_per_kernel * len(kernel_sizes), num_classes)

    def forward(self, tokens, lengths):
        # Support short inputs even when they contain fewer tokens than a kernel.
        if tokens.size(1) < max(self.kernel_sizes):
            tokens = F.pad(tokens, (0, max(self.kernel_sizes)-tokens.size(1)), value=PAD_ID)
        embedded = self.dropout(self.embedding(tokens)).transpose(1, 2)
        pooled = []
        for kernel, convolution in zip(self.kernel_sizes, self.convolutions):
            activations = F.relu(convolution(embedded))
            valid_windows = (lengths.to(tokens.device)-kernel+1).clamp(min=1)
            positions = torch.arange(activations.size(2), device=tokens.device)
            mask = positions.unsqueeze(0) >= valid_windows.unsqueeze(1)
            activations = activations.masked_fill(mask.unsqueeze(1), float("-inf"))
            pooled.append(activations.amax(dim=2))
        return self.classifier(self.dropout(torch.cat(pooled, dim=1)))


class TextLSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim=128, hidden_dim=128,
                 num_layers=1, bidirectional=True, dropout=.3, num_classes=3):
        super().__init__()
        self.directions = 2 if bidirectional else 1
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=PAD_ID)
        self.dropout = nn.Dropout(dropout)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=num_layers,
                            batch_first=True, bidirectional=bidirectional,
                            dropout=dropout if num_layers > 1 else 0.)
        self.classifier = nn.Linear(hidden_dim*self.directions, num_classes)

    def forward(self, tokens, lengths):
        embedded = self.dropout(self.embedding(tokens))
        packed = pack_padded_sequence(embedded, lengths.detach().cpu(), batch_first=True, enforce_sorted=False)
        _, (hidden, _) = self.lstm(packed)
        final_states = hidden[-self.directions:].transpose(0, 1).reshape(tokens.size(0), -1)
        return self.classifier(self.dropout(final_states))


def build_model(architecture, vocab_size, config, num_classes=3):
    shared = dict(vocab_size=vocab_size, embedding_dim=config["embedding_dim"],
                  dropout=config["dropout"], num_classes=num_classes)
    if architecture == "cnn":
        return TextCNN(**shared, **config["cnn"])
    if architecture == "lstm":
        return TextLSTM(**shared, **config["lstm"])
    raise ValueError(f"Unknown architecture: {architecture}")


def parameter_count(model):
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
