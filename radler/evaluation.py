"""Evaluation, diagnostics, and ablation helpers for RADLER."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix


WIDTH_VALUES = np.array([25.0, 50.0, 75.0, 100.0])


def selected_iou(iou_matrix: np.ndarray, predicted_classes: np.ndarray) -> np.ndarray:
    """Return per-image IoU selected by the predicted width class."""
    iou_matrix = np.asarray(iou_matrix, dtype=float)
    predicted_classes = np.asarray(predicted_classes, dtype=int)
    return iou_matrix[np.arange(len(predicted_classes)), predicted_classes]


def summarize_selector(iou_matrix: np.ndarray, target_classes: np.ndarray, predicted_classes: np.ndarray) -> dict:
    """Compute compact selector diagnostics used in the paper."""
    iou_matrix = np.asarray(iou_matrix, dtype=float)
    target_classes = np.asarray(target_classes, dtype=int)
    predicted_classes = np.asarray(predicted_classes, dtype=int)
    selected = selected_iou(iou_matrix, predicted_classes)
    oracle = iou_matrix.max(axis=1)
    return {
        "target_counts": np.bincount(target_classes, minlength=iou_matrix.shape[1]).tolist(),
        "selected_counts": np.bincount(predicted_classes, minlength=iou_matrix.shape[1]).tolist(),
        "exact_accuracy": float(accuracy_score(target_classes, predicted_classes)),
        "average_width": float(WIDTH_VALUES[predicted_classes].mean()),
        "mean_iou": float(selected.mean()),
        "oracle_iou": float(oracle.mean()),
        "oracle_regret": float((oracle - selected).mean()),
        "confusion_matrix": confusion_matrix(
            target_classes,
            predicted_classes,
            labels=list(range(iou_matrix.shape[1])),
        ),
    }


def bootstrap_ci(values: np.ndarray, seed: int = 123, n_boot: int = 10000) -> tuple[float, float]:
    """Bootstrap 95% confidence interval for the mean."""
    rng = np.random.default_rng(seed)
    values = np.asarray(values, dtype=float)
    samples = values[rng.integers(0, len(values), size=(n_boot, len(values)))]
    low, high = np.percentile(samples.mean(axis=1), [2.5, 97.5])
    return float(low), float(high)


def sign_flip_pvalue(values: np.ndarray, seed: int = 123, n_perm: int = 20000) -> float:
    """Two-sided paired sign-flip permutation p-value for mean difference."""
    rng = np.random.default_rng(seed)
    values = np.asarray(values, dtype=float)
    observed = abs(values.mean())
    signs = rng.choice([-1.0, 1.0], size=(n_perm, len(values)))
    null = abs((signs * values).mean(axis=1))
    return float((np.count_nonzero(null >= observed) + 1.0) / (n_perm + 1.0))


def paired_difference_report(
    candidate: np.ndarray,
    baseline: np.ndarray,
    seed: int = 123,
) -> dict:
    """Return mean paired difference, bootstrap CI, and sign-flip p-value."""
    diff = np.asarray(candidate, dtype=float) - np.asarray(baseline, dtype=float)
    low, high = bootstrap_ci(diff, seed=seed)
    return {
        "mean_difference": float(diff.mean()),
        "ci95_low": low,
        "ci95_high": high,
        "p_value": sign_flip_pvalue(diff, seed=seed),
    }


def random_matched_distribution(
    iou_matrix: np.ndarray,
    predicted_classes: np.ndarray,
    seed: int = 123,
    n_iter: int = 5000,
) -> dict:
    """Simulate random width selection with RADLER's selected-width distribution."""
    rng = np.random.default_rng(seed)
    predicted_classes = np.asarray(predicted_classes, dtype=int)
    probabilities = np.bincount(predicted_classes, minlength=iou_matrix.shape[1]) / len(predicted_classes)
    means = []
    widths = []
    for _ in range(n_iter):
        sampled = rng.choice(np.arange(iou_matrix.shape[1]), size=len(predicted_classes), p=probabilities)
        means.append(selected_iou(iou_matrix, sampled).mean())
        widths.append(WIDTH_VALUES[sampled].mean())
    low, high = np.percentile(means, [2.5, 97.5])
    return {
        "mean_iou": float(np.mean(means)),
        "ci95_low": float(low),
        "ci95_high": float(high),
        "mean_width": float(np.mean(widths)),
    }
