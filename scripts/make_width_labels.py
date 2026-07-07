#!/usr/bin/env python3
"""Generate RADLER width labels from per-image IoU values."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from radler.width_labels import make_width_labels, width_to_class


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path, help="CSV with one row per image and one IoU column per width.")
    parser.add_argument("--output", required=True, type=Path, help="Output CSV with added width_label and width_class columns.")
    parser.add_argument("--iou-columns", nargs=4, default=["iou_025", "iou_050", "iou_075", "iou_100"])
    parser.add_argument("--tolerance", type=float, required=True, help="Relative IoU-regret tolerance, e.g. 0.10 for 10%%.")
    args = parser.parse_args()

    frame = pd.read_csv(args.input)
    labels = make_width_labels(frame[args.iou_columns].to_numpy(), tolerance=args.tolerance)
    frame["width_label"] = labels
    frame["width_class"] = width_to_class(labels)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output, index=False)


if __name__ == "__main__":
    main()
