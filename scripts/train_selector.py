#!/usr/bin/env python3
"""Train the RADLER SVM width selector from feature and label tables."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from radler.selector import save_selector, train_svm_selector


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", required=True, type=Path, help="CSV/PKL feature table.")
    parser.add_argument("--labels", required=True, type=Path, help="CSV/PKL table containing the width class labels.")
    parser.add_argument("--label-column", default="width_class")
    parser.add_argument("--drop-columns", nargs="*", default=["index", "width_label", "width_class"])
    parser.add_argument("--output", required=True, type=Path, help="Path for the trained selector .joblib file.")
    parser.add_argument("--metadata", type=Path, help="Optional JSON metadata output.")
    parser.add_argument("--cv-splits", type=int, default=5)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--n-jobs", type=int, default=None)
    args = parser.parse_args()

    features = pd.read_pickle(args.features) if args.features.suffix in {".pkl", ".pickle"} else pd.read_csv(args.features)
    labels_frame = pd.read_pickle(args.labels) if args.labels.suffix in {".pkl", ".pickle"} else pd.read_csv(args.labels)
    labels = labels_frame[args.label_column].to_numpy(dtype=int)
    feature_columns = [column for column in features.columns if column not in set(args.drop_columns)]
    result = train_svm_selector(
        features[feature_columns],
        labels,
        cv_splits=args.cv_splits,
        random_state=args.seed,
        n_jobs=args.n_jobs,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_selector(result, args.output)
    metadata = {
        "best_params": result.best_params,
        "cv_score": result.cv_score,
        "train_accuracy": result.train_accuracy,
        "feature_columns": result.feature_columns,
    }
    if args.metadata:
        args.metadata.parent.mkdir(parents=True, exist_ok=True)
        args.metadata.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    else:
        print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
