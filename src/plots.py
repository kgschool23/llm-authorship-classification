"""Headless plots for experiment artifacts."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .data import LABELS


def plot_history(history, path):
    epochs = [row["epoch"] for row in history]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), constrained_layout=True)
    for split, color in (("train", "#2563eb"), ("validation", "#d97706")):
        axes[0].plot(epochs, [row[f"{split}_loss"] for row in history], label=split.title(), color=color)
        axes[1].plot(epochs, [row[f"{split}_macro_f1"] for row in history], label=split.title(), color=color)
    axes[0].set(ylabel="Cross-entropy loss", xlabel="Epoch", title="Loss")
    axes[1].set(ylabel="Macro-F1", xlabel="Epoch", title="Classification performance", ylim=(0, 1))
    for axis in axes:
        axis.legend()
        axis.grid(alpha=.2)
        axis.set_xticks(epochs)
    fig.savefig(Path(path), dpi=180)
    plt.close(fig)


def plot_confusion(metrics, path, title="Confusion matrix"):
    matrix = np.asarray(metrics["confusion_matrix"])
    fig, axis = plt.subplots(figsize=(5.6, 4.6), constrained_layout=True)
    image = axis.imshow(matrix, cmap="Blues")
    fig.colorbar(image, ax=axis, label="Responses")
    axis.set(xticks=range(len(LABELS)), yticks=range(len(LABELS)),
             xticklabels=LABELS, yticklabels=LABELS, xlabel="Predicted family", ylabel="True family", title=title)
    for i in range(len(LABELS)):
        for j in range(len(LABELS)):
            axis.text(j, i, str(matrix[i,j]), ha="center", va="center",
                      color="white" if matrix[i,j] > matrix.max()/2 else "black")
    fig.savefig(Path(path), dpi=180)
    plt.close(fig)
