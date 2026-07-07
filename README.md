# RADLER

This repository contains the code used to reproduce the software-side experiments
and analyses for:

**RADLER: Context-Aware Slimmable Model Selection for Energy-Efficient Real-Time UAV Weed Detection**

The repository is intentionally code-only and data-free. It includes the RADLER
feature extraction, relative-width labeling, SVM selector, slimmable backbone
definitions, analysis utilities, and plotting scripts used for the paper. It
does **not** include public datasets, trained segmentation checkpoints, cached
feature/IoU tables, trained SVM files, or raw Jetson/Monsoon energy logs.

## What Is Included

- `radler/features.py`: contextual image feature extraction used by the selector.
- `radler/width_labels.py`: relative-to-best IoU tolerance labeling.
- `radler/selector.py`: one-vs-rest SVM selector training and persistence.
- `radler/evaluation.py`: selector diagnostics, paired uncertainty, and baselines.
- `radler/energy.py`: aggregate final-test-set energy calculations.
- `radler/models/`: slimmable SU-Net and SSU-Net backbone definitions reused from SqueezeSlimU-Net.
- `radler/segmentation.py`: external-dataset loaders, per-width segmentation metrics, and FPS benchmarking helpers.
- `scripts/analyze_selector_artifacts.py`: reproduces selector diagnostics, training-label distributions, paired tests against the full-width baseline, simple baselines, feature ablations, and lightweight-classifier ablations from cached artifacts.
- `scripts/regenerate_tradeoff_figures.py`: regenerates the IoU-energy trade-off plots used in the manuscript.
- `scripts/make_radler_overview_figure.py`: regenerates the RADLER workflow figure.
- `scripts/extract_features_from_folder.py`: extracts contextual features from image folders.
- `scripts/make_width_labels.py`: creates RADLER width labels from per-width IoU tables.
- `scripts/train_selector.py`: trains the SVM selector from feature and label tables.
- `scripts/evaluate_selector.py`: evaluates a trained selector on held-out features/IoUs.
- `scripts/train_segmentation_backbone.py`: trains a slimmable SU-Net or SSU-Net backbone on external image/mask folders.
- `scripts/evaluate_segmentation_widths.py`: evaluates a checkpoint at each static width.
- `scripts/benchmark_segmentation_widths.py`: measures per-width latency and FPS with synthetic RGB inputs.

## What Is Not Included

The following are intentionally excluded to keep the repository publishable and
data-free:

- public weed datasets;
- image files and masks;
- `.pt`, `.pth`, `.ckpt`, `.onnx`, `.sav`, `.joblib`, `.pickle`, `.pkl`, `.npz`, `.npy`, `.csv` experiment artifacts;
- Jetson/Monsoon raw energy logs;
- generated paper PDFs and manuscript build artifacts.

## Expected Artifact Layout

For full reproduction of the paper analysis, place external artifacts under
`artifacts/` with this structure:

```text
artifacts/
  adaptation/garage/
    geok_squeeze_final/
      train_features.pickle
      test_features.pickle
      train_features_with_new_labels.pickle
      test_features_with_new_labels.pickle
      trained_classifier_geoksqueeze_100.sav
    geok_slim_final/
      ...
    tobacco_squeeze_final/
      ...
    tobacco_slim_final/
      ...
```

The expected case names and artifact files are listed in
`configs/paper_cases.json`.

## Quick Start

Install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Run selector diagnostics from cached artifacts:

```bash
python scripts/analyze_selector_artifacts.py \
  --repo artifacts \
  --out results/paper_analysis_outputs
```

Train a selector from feature and width-label tables:

```bash
python scripts/train_selector.py \
  --features artifacts/features/train_features.csv \
  --labels artifacts/labels/train_width_labels.csv \
  --output artifacts/selectors/radler_selector.joblib \
  --metadata results/radler_selector_metadata.json
```

Evaluate a selector:

```bash
python scripts/evaluate_selector.py \
  --selector artifacts/selectors/radler_selector.joblib \
  --features artifacts/features/test_features.csv \
  --labels artifacts/labels/test_width_labels_and_ious.csv \
  --output results/radler_selector_eval.json
```

Evaluate a slimmable segmentation checkpoint on external YOLO-style labels:

```bash
python scripts/evaluate_segmentation_widths.py \
  --architecture squeeze \
  --checkpoint artifacts/checkpoints/agriadapt_ssu.pt \
  --images data/agriadapt/test/images \
  --labels data/agriadapt/test/labels \
  --label-format yolo \
  --image-size 512 \
  --output results/agriadapt_ssu_width_metrics.csv
```

Benchmark per-width latency/FPS:

```bash
python scripts/benchmark_segmentation_widths.py \
  --architecture squeeze \
  --checkpoint artifacts/checkpoints/agriadapt_ssu.pt \
  --image-size 512 \
  --device auto \
  --output results/agriadapt_ssu_width_fps.csv
```

Regenerate manuscript figures:

```bash
python scripts/make_radler_overview_figure.py
python scripts/regenerate_tradeoff_figures.py
```

## Data Sources

- AgriAdapt Weed Detection dataset: see the project repository cited in the paper.
- Tobacco Aerial Dataset: see the original dataset source cited in the paper.

The paper uses public datasets through externally prepared split files, feature
tables, per-width IoU tables, trained selectors, segmentation checkpoints, and
energy logs. When those derived artifacts cannot be redistributed directly, the
scripts and templates in this repository document the expected layout and can be
used to regenerate them from the public datasets and locally trained backbones.

For YOLO-style labels, the evaluation/training scripts expect:

```text
split/
  images/
    frame_0001.jpg
  labels/
    frame_0001.txt
```

For class-valued masks, such as Tobacco-style `data/` and `maskref/` folders,
pass `--label-format mask --images <data-dir> --masks <mask-dir>`.

## Reproducibility Notes

- The reported RADLER selector uses the full 39-feature contextual vector after
  scaling. Correlation-pruned and feature-group variants are diagnostic ablations.
- The width-label tolerance is relative IoU regret:
  `(IoU_max - IoU_width) / IoU_max <= delta`.
- The selector is a one-vs-rest SVM trained on MinMax-scaled contextual features.
- Jetson measurements in the paper are vision-pipeline measurements, not full UAV
  mission battery measurements.
- Backbone training follows slimmable-network practice by accumulating gradients
  over widths in descending order before each optimizer update.

## License

No open-source license has been selected yet. Until a license is added, reuse is
limited to the permissions granted by the repository owners.
