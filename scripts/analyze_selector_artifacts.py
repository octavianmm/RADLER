#!/usr/bin/env python3
"""Generate reproducible paper analyses for the RADLER IEEE Access manuscript.

The script reads the stored adaptation artifacts produced by the RADLER codebase
and writes compact CSV/LaTeX tables for selector diagnostics, uncertainty
analysis, simple baselines, and feature/classifier ablations. It intentionally
does not retrain the segmentation backbones.

Expected repository layout, relative to --repo:
  adaptation/garage/<case>/{train,test}_features.pickle
  adaptation/garage/<case>/{train,test}_features_with_new_labels.pickle
"""

from __future__ import annotations

import argparse
import csv
import math
import pickle
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.multiclass import OneVsRestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


warnings.filterwarnings("ignore")

WIDTH_LABELS = ["25", "50", "75", "100"]
WIDTH_VALUES = np.array([25.0, 50.0, 75.0, 100.0])
METRIC_COLUMNS = ["precision_025", "precision_050", "precision_075", "precision_100"]
DROP_COLUMNS = ["index", "max_precision_model", *METRIC_COLUMNS]


@dataclass(frozen=True)
class Case:
    label: str
    short: str
    artifact_dir: str
    classifier: str


CASES = [
    Case("AgriAdapt SSU-Net", "agriadapt_ssu", "geok_squeeze_final", "trained_classifier_geoksqueeze_100.sav"),
    Case("AgriAdapt SU-Net", "agriadapt_su", "geok_slim_final", "trained_classifier_geokslim_10.sav"),
    Case("Tobacco SSU-Net", "tobacco_ssu", "tobacco_squeeze_final", "trained_classifier_tobaccosqueeze_01.sav"),
    Case("Tobacco SU-Net", "tobacco_su", "tobacco_slim_final", "trained_classifier_tobaccoslim_1.sav"),
]


FEATURE_GROUPS = {
    "full context": None,
    "correlation-pruned": "corr_pruned",
    "brightness only": ["mean_brightness"],
    "spectral": [
        "mean_brightness",
        "std_brightness",
        "max_brightness",
        "min_brightness",
        "no_bins",
        "contrast_hue_hist",
        "std_hue_arc",
        "contrast",
        "mean_saturation",
        "std_saturation",
        "max_saturation",
    ],
    "morphological": ["keypoints"],
    "texture": [
        "glcm_contrast_1",
        "glcm_contrast_2",
        "glcm_contrast_3",
        "glcm_contrast_4",
        "glcm_correlation_1",
        "glcm_correlation_2",
        "glcm_correlation_3",
        "glcm_correlation_4",
        "glcm_dissimilarity_1",
        "glcm_dissimilarity_2",
        "glcm_dissimilarity_3",
        "glcm_dissimilarity_4",
        "glcm_asm_1",
        "glcm_asm_2",
        "glcm_asm_3",
        "glcm_asm_4",
        "glcm_energy_1",
        "glcm_energy_2",
        "glcm_energy_3",
        "glcm_energy_4",
        "glcm_homogeneity_1",
        "glcm_homogeneity_2",
        "glcm_homogeneity_3",
        "glcm_homogeneity_4",
    ],
    "wavelet": ["dwt_decomposition_1", "dwt_decomposition_2", "dwt_decompositon_3"],
}


def as_float_frame(df: pd.DataFrame) -> pd.DataFrame:
    return df.apply(pd.to_numeric, errors="coerce").fillna(0.0)


def load_case(repo: Path, case: Case):
    case_dir = repo / "adaptation" / "garage" / case.artifact_dir
    train_full = as_float_frame(pd.read_pickle(case_dir / "train_features.pickle").drop(columns=["index"], errors="ignore"))
    test_full = as_float_frame(pd.read_pickle(case_dir / "test_features.pickle").drop(columns=["index"], errors="ignore"))
    train_labeled = pd.read_pickle(case_dir / "train_features_with_new_labels.pickle")
    test_labeled = pd.read_pickle(case_dir / "test_features_with_new_labels.pickle")
    y_train = train_labeled["max_precision_model"].to_numpy(dtype=int)
    y_test = test_labeled["max_precision_model"].to_numpy(dtype=int)
    iou_test = test_labeled[METRIC_COLUMNS].astype(float).to_numpy()
    classifier = pickle.load(open(case_dir / case.classifier, "rb"))
    return case_dir, train_full, test_full, y_train, y_test, iou_test, classifier


def scaled(train_x: pd.DataFrame, test_x: pd.DataFrame):
    scaler = MinMaxScaler().fit(train_x.values)
    return (
        pd.DataFrame(scaler.transform(train_x.values), columns=train_x.columns),
        pd.DataFrame(scaler.transform(test_x.values), columns=test_x.columns),
    )


def classifier_columns(classifier, test_x: pd.DataFrame):
    if hasattr(classifier, "feature_names_in_"):
        cols = list(classifier.feature_names_in_)
        if set(cols).issubset(set(test_x.columns)):
            return cols
    return list(test_x.columns)


def selected_iou(iou: np.ndarray, pred: np.ndarray) -> np.ndarray:
    return iou[np.arange(len(pred)), pred]


def bootstrap_ci(values: np.ndarray, rng: np.random.Generator, n_boot: int = 10000):
    values = np.asarray(values, dtype=float)
    samples = values[rng.integers(0, len(values), size=(n_boot, len(values)))]
    return np.percentile(samples.mean(axis=1), [2.5, 97.5])


def sign_flip_pvalue(values: np.ndarray, rng: np.random.Generator, n_perm: int = 20000):
    values = np.asarray(values, dtype=float)
    observed = abs(values.mean())
    signs = rng.choice([-1.0, 1.0], size=(n_perm, len(values)))
    null = abs((signs * values).mean(axis=1))
    return (np.count_nonzero(null >= observed) + 1.0) / (n_perm + 1.0)


def width_distribution(pred: np.ndarray) -> str:
    counts = np.bincount(pred, minlength=4)
    return "/".join(str(int(x)) for x in counts)


def metric_summary(iou: np.ndarray, pred: np.ndarray):
    vals = selected_iou(iou, pred)
    return WIDTH_VALUES[pred].mean(), vals.mean() * 100.0


def make_svc_from_saved(classifier):
    estimator = getattr(classifier, "estimator", classifier)
    params = estimator.get_params()
    svc = SVC(
        C=params.get("C", 1.0),
        kernel=params.get("kernel", "rbf"),
        gamma=params.get("gamma", "scale"),
        degree=params.get("degree", 3),
        class_weight=params.get("class_weight", None),
        random_state=params.get("random_state", 100),
    )
    return OneVsRestClassifier(svc)


def correlation_pruned_columns(train_x: pd.DataFrame, threshold: float = 0.85):
    corr = train_x.corr(numeric_only=True).abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    drop = {column for column in upper.columns if any(upper[column] > threshold)}
    keep = [column for column in train_x.columns if column not in drop]
    return keep


def train_predict(model, train_x: pd.DataFrame, y_train: np.ndarray, test_x: pd.DataFrame):
    x_train, x_test = scaled(train_x, test_x)
    model.fit(x_train, y_train)
    return np.asarray(model.predict(x_test), dtype=int)


def write_csv(path: Path, rows: Iterable[dict], fieldnames: list[str]):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_table(path: Path, caption: str, label: str, tabular_spec: str, header: str, rows: list[str]):
    with open(path, "w") as f:
        f.write("\\begin{table*}[!tb]\n")
        f.write("\\centering\n\\scriptsize\n")
        f.write("\\setlength{\\tabcolsep}{3pt}\n")
        f.write("\\renewcommand{\\arraystretch}{1.05}\n")
        f.write(f"\\begin{{tabular}}{{{tabular_spec}}}\n")
        f.write("\\toprule\n")
        f.write(header + "\\\\\n")
        f.write("\\midrule\n")
        f.write("\n".join(rows))
        f.write("\n\\bottomrule\n")
        f.write("\\end{tabular}\n")
        f.write(f"\\caption{{{caption}}}\n")
        f.write(f"\\label{{{label}}}\n")
        f.write("\\end{table*}\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path.cwd(), help="RADLER repository root")
    parser.add_argument("--out", type=Path, default=Path("results/paper_analysis_outputs"), help="Output directory")
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    diagnostics_rows = []
    stats_rows = []
    baseline_rows = []
    ablation_rows = []
    classifier_rows = []
    confusion_rows = []
    hyper_rows = []
    training_label_rows = []

    for case in CASES:
        _, train_x, test_x, y_train, y_test, iou_test, classifier = load_case(args.repo, case)
        train_scaled, test_scaled = scaled(train_x, test_x)
        cols = classifier_columns(classifier, test_scaled)
        pred = np.asarray(classifier.predict(test_scaled[cols]), dtype=int)
        radler_width, radler_iou = metric_summary(iou_test, pred)
        static_iou = iou_test.mean(axis=0) * 100.0
        full_iou = static_iou[3]
        best_static_idx = int(np.argmax(static_iou))
        best_static_iou = static_iou[best_static_idx]
        oracle_iou = iou_test.max(axis=1).mean() * 100.0
        oracle_pred = iou_test.argmax(axis=1)

        estimator = getattr(classifier, "estimator", classifier)
        params = estimator.get_params()
        hyper_rows.append(
            {
                "Setting": case.label,
                "Classifier": case.classifier,
                "C": params.get("C"),
                "kernel": params.get("kernel"),
                "gamma": params.get("gamma"),
                "degree": params.get("degree"),
                "class_weight": params.get("class_weight"),
                "features": len(cols),
            }
        )

        training_label_rows.append(
            {
                "Setting": case.label,
                "n_train": len(y_train),
                "target_25_50_75_100": width_distribution(y_train),
            }
        )

        diagnostics_rows.append(
            {
                "Setting": case.label,
                "n_test": len(pred),
                "target_25_50_75_100": width_distribution(y_test),
                "selected_25_50_75_100": width_distribution(pred),
                "selector_exact_accuracy_pct": f"{accuracy_score(y_test, pred) * 100.0:.1f}",
                "avg_width_pct": f"{radler_width:.2f}",
                "radler_iou_pct": f"{radler_iou:.2f}",
                "full_width_iou_pct": f"{full_iou:.2f}",
                "oracle_iou_pct": f"{oracle_iou:.2f}",
                "oracle_regret_iou_points": f"{oracle_iou - radler_iou:.2f}",
            }
        )

        cm = confusion_matrix(y_test, pred, labels=[0, 1, 2, 3])
        for row_idx, width in enumerate(WIDTH_LABELS):
            confusion_rows.append(
                {
                    "Setting": case.label,
                    "Target": width,
                    "Pred25": int(cm[row_idx, 0]),
                    "Pred50": int(cm[row_idx, 1]),
                    "Pred75": int(cm[row_idx, 2]),
                    "Pred100": int(cm[row_idx, 3]),
                }
            )

        radler_vals = selected_iou(iou_test, pred) * 100.0
        full_vals = iou_test[:, 3] * 100.0
        best_static_vals = iou_test[:, best_static_idx] * 100.0
        oracle_vals = iou_test.max(axis=1) * 100.0
        diff = radler_vals - full_vals
        ci = bootstrap_ci(diff, rng)
        pvalue = sign_flip_pvalue(diff, rng)
        stats_rows.append(
            {
                "Setting": case.label,
                "Comparison": "RADLER - full width",
                "mean_diff_iou_points": f"{diff.mean():.3f}",
                "ci95_low": f"{ci[0]:.3f}",
                "ci95_high": f"{ci[1]:.3f}",
                "paired_permutation_p": f"{pvalue:.4f}",
            }
        )

        closest_static_idx = int(np.argmin(np.abs(WIDTH_VALUES - radler_width)))
        matched_probs = np.bincount(pred, minlength=4) / len(pred)
        random_means = []
        random_widths = []
        for _ in range(5000):
            rand_pred = rng.choice(np.arange(4), size=len(pred), p=matched_probs)
            random_widths.append(WIDTH_VALUES[rand_pred].mean())
            random_means.append(selected_iou(iou_test, rand_pred).mean() * 100.0)
        random_ci = np.percentile(random_means, [2.5, 97.5])

        baseline_rows.extend(
            [
                {
                    "Setting": case.label,
                    "Baseline": "RADLER SVM selector",
                    "AvgWidth": f"{radler_width:.2f}",
                    "IoU": f"{radler_iou:.2f}",
                    "Notes": "reported operating point",
                },
                {
                    "Setting": case.label,
                    "Baseline": "closest fixed width",
                    "AvgWidth": f"{WIDTH_VALUES[closest_static_idx]:.2f}",
                    "IoU": f"{static_iou[closest_static_idx]:.2f}",
                    "Notes": f"fixed {WIDTH_LABELS[closest_static_idx]}%",
                },
                {
                    "Setting": case.label,
                    "Baseline": "best fixed width",
                    "AvgWidth": f"{WIDTH_VALUES[best_static_idx]:.2f}",
                    "IoU": f"{best_static_iou:.2f}",
                    "Notes": f"fixed {WIDTH_LABELS[best_static_idx]}%",
                },
                {
                    "Setting": case.label,
                    "Baseline": "random matched distribution",
                    "AvgWidth": f"{np.mean(random_widths):.2f}",
                    "IoU": f"{np.mean(random_means):.2f}",
                    "Notes": f"95% CI {random_ci[0]:.2f}--{random_ci[1]:.2f}",
                },
            ]
        )

        corr_cols = correlation_pruned_columns(train_x)
        for group_name, requested_cols in FEATURE_GROUPS.items():
            if requested_cols == "corr_pruned":
                cols_for_group = corr_cols
            elif requested_cols is None:
                cols_for_group = list(train_x.columns)
            else:
                cols_for_group = [c for c in requested_cols if c in train_x.columns]
            if not cols_for_group:
                continue
            model = make_svc_from_saved(classifier)
            group_pred = train_predict(model, train_x[cols_for_group], y_train, test_x[cols_for_group])
            group_width, group_iou = metric_summary(iou_test, group_pred)
            ablation_rows.append(
                {
                    "Setting": case.label,
                    "FeatureSet": group_name,
                    "Features": len(cols_for_group),
                    "AvgWidth": f"{group_width:.2f}",
                    "IoU": f"{group_iou:.2f}",
                    "SelectorAcc": f"{accuracy_score(y_test, group_pred) * 100.0:.1f}",
                }
            )

        alt_models = {
            "SVM (reported hyperparameters)": make_svc_from_saved(classifier),
            "logistic regression": LogisticRegression(max_iter=5000, random_state=100, multi_class="auto"),
            "k-nearest neighbors": KNeighborsClassifier(n_neighbors=5),
            "decision tree": DecisionTreeClassifier(max_depth=5, random_state=100),
            "random forest": RandomForestClassifier(
                n_estimators=300,
                max_depth=15,
                criterion="entropy",
                max_features="sqrt",
                class_weight="balanced",
                random_state=1000,
            ),
        }
        for model_name, model in alt_models.items():
            alt_pred = train_predict(model, train_x, y_train, test_x)
            alt_width, alt_iou = metric_summary(iou_test, alt_pred)
            classifier_rows.append(
                {
                    "Setting": case.label,
                    "Classifier": model_name,
                    "AvgWidth": f"{alt_width:.2f}",
                    "IoU": f"{alt_iou:.2f}",
                    "SelectorAcc": f"{accuracy_score(y_test, alt_pred) * 100.0:.1f}",
                }
            )

    write_csv(out / "selector_diagnostics.csv", diagnostics_rows, list(diagnostics_rows[0].keys()))
    write_csv(out / "selector_hyperparameters.csv", hyper_rows, list(hyper_rows[0].keys()))
    write_csv(out / "training_label_distributions.csv", training_label_rows, list(training_label_rows[0].keys()))
    write_csv(out / "paired_uncertainty.csv", stats_rows, list(stats_rows[0].keys()))
    write_csv(out / "simple_baselines.csv", baseline_rows, list(baseline_rows[0].keys()))
    write_csv(out / "feature_ablation.csv", ablation_rows, list(ablation_rows[0].keys()))
    write_csv(out / "classifier_ablation.csv", classifier_rows, list(classifier_rows[0].keys()))
    write_csv(out / "selector_confusion.csv", confusion_rows, list(confusion_rows[0].keys()))

    write_table(
        out / "selector_diagnostics_table.tex",
        "Selector diagnostics for the reported RADLER operating points. Target and selected widths are reported as counts for 25/50/75/100 percent widths.",
        "tab:selector_diagnostics",
        "lrrrrr",
        "\\textbf{Setting} & \\textbf{Target widths} & \\textbf{Selected widths} & \\textbf{Exact acc.} & \\textbf{Avg. width} & \\textbf{Oracle regret}",
        [
            f"{r['Setting']} & {r['target_25_50_75_100']} & {r['selected_25_50_75_100']} & {r['selector_exact_accuracy_pct']} & {r['avg_width_pct']} & {r['oracle_regret_iou_points']}\\\\"
            for r in diagnostics_rows
        ],
    )

    write_table(
        out / "training_label_distributions_table.tex",
        "Training target-width label distributions generated from the labeled training artifacts. Counts are reported for 25/50/75/100 percent widths.",
        "tab:training_label_distributions",
        "lrr",
        "\\textbf{Setting} & \\textbf{$n_{train}$} & \\textbf{Target widths}",
        [
            f"{r['Setting']} & {r['n_train']} & {r['target_25_50_75_100']}\\\\"
            for r in training_label_rows
        ],
    )

    stat_rows_for_table = [r for r in stats_rows if r["Comparison"] == "RADLER - full width"]
    write_table(
        out / "paired_uncertainty_table.tex",
        "Paired per-image IoU differences for RADLER against the full-width baseline. Positive values favor RADLER; intervals are 95 percent bootstrap confidence intervals.",
        "tab:paired_uncertainty",
        "llrrrr",
        "\\textbf{Setting} & \\textbf{Comparison} & \\textbf{Mean diff.} & \\textbf{95\\% CI low} & \\textbf{95\\% CI high} & \\textbf{$p$}",
        [
            f"{r['Setting']} & {r['Comparison'].replace('RADLER - ', '')} & {r['mean_diff_iou_points']} & {r['ci95_low']} & {r['ci95_high']} & {r['paired_permutation_p']}\\\\"
            for r in stat_rows_for_table
        ],
    )

    write_table(
        out / "simple_baselines_table.tex",
        "Simple adaptive and fixed-width baselines derived from the same per-image width-IoU artifacts.",
        "tab:simple_baselines",
        "llrrl",
        "\\textbf{Setting} & \\textbf{Method} & \\textbf{Avg. width} & \\textbf{IoU} & \\textbf{Notes}",
        [
            f"{r['Setting']} & {r['Baseline']} & {r['AvgWidth']} & {r['IoU']} & {r['Notes']}\\\\"
            for r in baseline_rows
        ],
    )

    write_table(
        out / "feature_ablation_table.tex",
        "Feature-group ablation for the SVM selector. Each row retrains the selector with the indicated feature subset while keeping the segmentation backbones fixed.",
        "tab:feature_ablation",
        "llrrr",
        "\\textbf{Setting} & \\textbf{Feature set} & \\textbf{Features} & \\textbf{Avg. width} & \\textbf{IoU}",
        [
            f"{r['Setting']} & {r['FeatureSet']} & {r['Features']} & {r['AvgWidth']} & {r['IoU']}\\\\"
            for r in ablation_rows
        ],
    )

    write_table(
        out / "classifier_ablation_table.tex",
        "Alternative lightweight selector models trained on the same contextual features and optimal-width labels.",
        "tab:classifier_ablation",
        "llrrr",
        "\\textbf{Setting} & \\textbf{Selector} & \\textbf{Avg. width} & \\textbf{IoU} & \\textbf{Exact acc.}",
        [
            f"{r['Setting']} & {r['Classifier']} & {r['AvgWidth']} & {r['IoU']} & {r['SelectorAcc']}\\\\"
            for r in classifier_rows
        ],
    )

    write_table(
        out / "selector_confusion_table.tex",
        "Selector confusion matrices for the reported RADLER operating points. Rows are target widths and columns are selected widths.",
        "tab:selector_confusion",
        "llrrrr",
        "\\textbf{Setting} & \\textbf{Target} & \\textbf{Pred. 25} & \\textbf{Pred. 50} & \\textbf{Pred. 75} & \\textbf{Pred. 100}",
        [
            f"{r['Setting']} & {r['Target']} & {r['Pred25']} & {r['Pred50']} & {r['Pred75']} & {r['Pred100']}\\\\"
            for r in confusion_rows
        ],
    )

    print(f"Wrote paper analysis outputs to {out.resolve()}")


if __name__ == "__main__":
    main()
