#!/usr/bin/env python3
import csv
import json
from pathlib import Path
from typing import List, Optional

import numpy as np


def default_channel_names(n: int) -> List[str]:
    return [f"канал[{i}]" for i in range(n)]


def load_channel_names(meta_path: Optional[str], n_channels: int) -> List[str]:
    labels = []
    if meta_path:
        path = Path(meta_path)
        if path.exists():
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                if isinstance(data, dict):
                    if isinstance(data.get("stream"), dict):
                        labels = data["stream"].get("channel_names") or []
                    if not labels:
                        channels = data.get("channels") or []
                        if isinstance(channels, list):
                            labels = [str(ch.get("label", "")).strip() for ch in channels if isinstance(ch, dict)]
            except Exception:
                labels = []

    labels = [str(x).strip() for x in labels if str(x).strip()]
    if len(labels) != n_channels:
        return default_channel_names(n_channels)
    return labels


def shift_matrix_rows_for_display(matrix: np.ndarray) -> np.ndarray:
    shifted = np.array(matrix, copy=True)
    if shifted.ndim != 2 or shifted.shape[0] <= 1:
        return shifted

    shifted[1:] = np.array([np.roll(row, 1) for row in shifted[1:]])
    return shifted


def format_matrix_text(
    matrix: np.ndarray,
    labels: Optional[List[str]] = None,
    decimals: int = 3,
    max_rows: int = 20,
    max_cols: int = 20,
) -> str:
    n_rows, n_cols = matrix.shape
    row_indices = list(range(min(n_rows, max_rows)))
    col_indices = list(range(min(n_cols, max_cols)))
    row_labels = labels if labels and len(labels) == n_rows else default_channel_names(n_rows)
    col_labels = row_labels

    col_width = max(8, max(len(col_labels[j]) for j in col_indices))
    value_width = max(8, decimals + 4)
    first_width = max(8, max(len(row_labels[i]) for i in row_indices))

    header = ' ' * (first_width + 1)
    header += ' '.join(f"{col_labels[j]:>{max(col_width, value_width)}}" for j in col_indices)
    lines = [header]

    for i in row_indices:
        parts = [f"{row_labels[i]:>{first_width}}"]
        for j in col_indices:
            value = matrix[i, j]
            if np.isnan(value):
                text = 'не_число'
            elif np.isinf(value):
                text = 'беск' if value > 0 else '-беск'
            else:
                text = f"{value:.{decimals}f}"
            parts.append(f"{text:>{max(col_width, value_width)}}")
        lines.append(' '.join(parts))

    if n_rows > max_rows or n_cols > max_cols:
        lines.append(
            f"... матрица обрезана: показано {len(row_indices)}x{len(col_indices)} из {n_rows}x{n_cols}"
        )

    return '\n'.join(lines)


def save_matrix_csv(path: Path, matrix: np.ndarray, labels: Optional[List[str]] = None) -> None:
    labels = labels if labels and len(labels) == matrix.shape[0] else default_channel_names(matrix.shape[0])
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['канал'] + labels)
        for label, row in zip(labels, matrix):
            writer.writerow([label] + [float(x) if np.isfinite(x) else str(x) for x in row])


def save_matrix_heatmap(path: Path, matrix: np.ndarray, labels: Optional[List[str]], title: str) -> None:
    import matplotlib.pyplot as plt

    labels = labels if labels and len(labels) == matrix.shape[0] else default_channel_names(matrix.shape[0])
    path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 8))
    image = ax.imshow(matrix, aspect='auto', vmin=-1.0, vmax=1.0)
    ax.set_title(title)
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=90)
    ax.set_yticklabels(labels)
    fig.colorbar(image, ax=ax)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def compare_matrices(a: np.ndarray, b: np.ndarray, threshold_small: float = 0.01, threshold_large: float = 0.05) -> dict:
    if a.shape != b.shape:
        raise ValueError(f"Формы матриц различаются: {a.shape} vs {b.shape}")

    diff = a - b
    abs_diff = np.abs(diff)
    finite = np.isfinite(abs_diff)
    off_diag_mask = ~np.eye(a.shape[0], dtype=bool)
    off_diag = abs_diff[off_diag_mask]
    off_diag_finite = off_diag[np.isfinite(off_diag)]

    return {
        "shape": list(a.shape),
        "max_abs_diff": float(np.nanmax(abs_diff)) if np.any(finite) else None,
        "mean_abs_diff": float(np.nanmean(abs_diff)) if np.any(finite) else None,
        "off_diag_max_abs_diff": float(np.nanmax(off_diag_finite)) if off_diag_finite.size else None,
        "off_diag_mean_abs_diff": float(np.nanmean(off_diag_finite)) if off_diag_finite.size else None,
        "off_diag_fraction_absdiff_le_0_01": (
            float(np.mean(off_diag_finite <= threshold_small)) if off_diag_finite.size else None
        ),
        "off_diag_fraction_absdiff_le_0_05": (
            float(np.mean(off_diag_finite <= threshold_large)) if off_diag_finite.size else None
        ),
    }
