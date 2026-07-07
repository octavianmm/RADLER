"""SVM selector training and inference utilities."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC


DEFAULT_PARAM_GRID = {
    "selector__estimator__C": [1, 10, 100, 1000],
    "selector__estimator__kernel": ["rbf", "poly", "sigmoid"],
    "selector__estimator__gamma": ["scale", "auto"],
    "selector__estimator__class_weight": [None, "balanced"],
}


@dataclass(frozen=True)
class SelectorTrainingResult:
    model: Pipeline
    best_params: dict
    cv_score: float
    train_accuracy: float
    feature_columns: list[str]


def train_svm_selector(
    features: pd.DataFrame,
    labels: Iterable[int],
    *,
    cv_splits: int = 5,
    random_state: int = 123,
    param_grid: dict | None = None,
    n_jobs: int | None = None,
) -> SelectorTrainingResult:
    """Train RADLER's one-vs-rest SVM selector.

    MinMax scaling is fitted inside the cross-validation pipeline to avoid
    train/validation leakage during hyperparameter selection.
    """
    features = features.apply(pd.to_numeric, errors="coerce").fillna(0.0)
    labels = np.asarray(labels, dtype=int)
    pipeline = Pipeline(
        [
            ("scaler", MinMaxScaler()),
            ("selector", OneVsRestClassifier(SVC(random_state=random_state))),
        ]
    )
    cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=random_state)
    search = GridSearchCV(
        estimator=pipeline,
        param_grid=DEFAULT_PARAM_GRID if param_grid is None else param_grid,
        scoring="accuracy",
        cv=cv,
        n_jobs=n_jobs,
        refit=True,
    )
    search.fit(features, labels)
    train_predictions = search.best_estimator_.predict(features)
    return SelectorTrainingResult(
        model=search.best_estimator_,
        best_params=dict(search.best_params_),
        cv_score=float(search.best_score_),
        train_accuracy=float(accuracy_score(labels, train_predictions)),
        feature_columns=list(features.columns),
    )


def save_selector(result: SelectorTrainingResult, path: str | Path) -> None:
    """Persist a trained selector and its feature columns."""
    payload = {
        "model": result.model,
        "best_params": result.best_params,
        "cv_score": result.cv_score,
        "train_accuracy": result.train_accuracy,
        "feature_columns": result.feature_columns,
    }
    joblib.dump(payload, path)


def load_selector(path: str | Path) -> dict:
    """Load a selector payload created by :func:`save_selector`."""
    return joblib.load(path)


def predict_width_classes(selector_payload: dict, features: pd.DataFrame) -> np.ndarray:
    """Predict width classes using a saved selector payload."""
    columns = selector_payload["feature_columns"]
    features = features[columns].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    return np.asarray(selector_payload["model"].predict(features), dtype=int)
