"""Energy aggregation helpers for RADLER operating-point figures."""

from __future__ import annotations

import numpy as np


def aggregate_energy_mAh(
    selected_classes: np.ndarray,
    per_image_energy_uAh: np.ndarray | list[float],
    selector_energy_uAh: float = 0.0,
) -> float:
    """Aggregate final-test-set energy in mAh.

    Parameters
    ----------
    selected_classes:
        Integer width class per image.
    per_image_energy_uAh:
        Per-image network energy for each width class, in microampere-hours.
    selector_energy_uAh:
        Optional per-image selector overhead, also in microampere-hours.
    """
    selected_classes = np.asarray(selected_classes, dtype=int)
    per_image_energy_uAh = np.asarray(per_image_energy_uAh, dtype=float)
    network_energy = per_image_energy_uAh[selected_classes].sum()
    selector_energy = selector_energy_uAh * len(selected_classes)
    return float((network_energy + selector_energy) / 1000.0)
