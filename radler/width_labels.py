"""Generate relative-to-best width labels for RADLER selector training."""

from __future__ import annotations

import numpy as np


def select_narrowest_acceptable_width(
    ious: np.ndarray,
    widths: np.ndarray | list[float] = (0.25, 0.50, 0.75, 1.0),
    tolerance: float = 0.10,
) -> float:
    """Return the smallest width within a relative IoU-regret tolerance.

    A width is acceptable when ``(IoU_max - IoU_width) / IoU_max <= tolerance``.
    If multiple widths satisfy the tolerance, RADLER intentionally chooses the
    narrowest one.
    """
    ious = np.asarray(ious, dtype=float)
    widths = np.asarray(widths, dtype=float)
    if ious.shape[0] != widths.shape[0]:
        raise ValueError("ious and widths must have the same length.")
    best = float(np.max(ious))
    if best <= 0:
        return float(widths[int(np.argmax(ious))])
    acceptable = np.flatnonzero((best - ious) / best <= tolerance)
    return float(widths[int(acceptable[0])])


def make_width_labels(
    iou_matrix: np.ndarray,
    widths: np.ndarray | list[float] = (0.25, 0.50, 0.75, 1.0),
    tolerance: float = 0.10,
) -> np.ndarray:
    """Generate one width label per image from an image-by-width IoU matrix."""
    return np.asarray(
        [
            select_narrowest_acceptable_width(row, widths=widths, tolerance=tolerance)
            for row in np.asarray(iou_matrix, dtype=float)
        ],
        dtype=float,
    )


def width_to_class(labels: np.ndarray, widths: np.ndarray | list[float] = (0.25, 0.50, 0.75, 1.0)) -> np.ndarray:
    """Map width values to integer classes 0..N-1."""
    widths = np.asarray(widths, dtype=float)
    index = {float(width): idx for idx, width in enumerate(widths)}
    return np.asarray([index[float(label)] for label in labels], dtype=int)


def class_to_width(classes: np.ndarray, widths: np.ndarray | list[float] = (0.25, 0.50, 0.75, 1.0)) -> np.ndarray:
    """Map integer width classes back to width values."""
    widths = np.asarray(widths, dtype=float)
    return widths[np.asarray(classes, dtype=int)]
