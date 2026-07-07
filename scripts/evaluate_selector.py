#!/usr/bin/env python3
"""Evaluate a trained RADLER selector on held-out feature/IoU artifacts."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from radler.evaluation import paired_difference_report, selected_iou, summarize_selector
from radler.selector import load_selector, predict_width_classes


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selector", required=True, type=Path)
    parser.add_argument("--features", required=True, type=Path)
    parser.add_argument("--labels", required=True, type=Path)
    parser.add_argument("--label-column", default="width_class")
    parser.add_argument("--iou-columns", nargs=4, default=["iou_025", "iou_050", "iou_075", "iou_100"])
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    payload = load_selector(args.selector)
    features = pd.read_pickle(args.features) if args.features.suffix in {".pkl", ".pickle"} else pd.read_csv(args.features)
    labels_frame = pd.read_pickle(args.labels) if args.labels.suffix in {".pkl", ".pickle"} else pd.read_csv(args.labels)
    target = labels_frame[args.label_column].to_numpy(dtype=int)
    iou_matrix = labels_frame[args.iou_columns].to_numpy(dtype=float)
    predicted = predict_width_classes(payload, features)

    selected = selected_iou(iou_matrix, predicted)
    full_width = iou_matrix[:, -1]
    report = summarize_selector(iou_matrix, target, predicted)
    report["paired_vs_full_width"] = paired_difference_report(selected, full_width)
    report["confusion_matrix"] = np.asarray(report["confusion_matrix"]).tolist()
    report["predicted_classes"] = predicted.tolist()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
