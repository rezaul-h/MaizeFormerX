"""
General figure-building helpers for experiment outputs.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.utils.io import ensure_dir


def plot_bar_comparison(
    labels: list[str],
    values: list[float],
    save_path: str | Path,
    title: str,
    ylabel: str,
) -> None:
    save_path = Path(save_path)
    ensure_dir(save_path)

    plt.figure(figsize=(10, 5))
    x = np.arange(len(labels))
    plt.bar(x, values)
    plt.xticks(x, labels, rotation=45, ha="right")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_heatmap(
    matrix: np.ndarray,
    x_labels: list[str],
    y_labels: list[str],
    save_path: str | Path,
    title: str,
    cmap: str = "Blues",
) -> None:
    save_path = Path(save_path)
    ensure_dir(save_path)

    plt.figure(figsize=(7, 6))
    plt.imshow(matrix, cmap=cmap, aspect="auto")
    plt.colorbar()
    plt.xticks(range(len(x_labels)), x_labels, rotation=45, ha="right")
    plt.yticks(range(len(y_labels)), y_labels)
    plt.title(title)

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            plt.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center", fontsize=8)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()