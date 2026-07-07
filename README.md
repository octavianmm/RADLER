# RADLER

**RADLER: Context-Aware Slimmable Model Selection for Energy-Efficient Real-Time UAV Weed Detection**

This repository provides the code and configuration files used for the software experiments and analyses presented in the paper above. The release focuses on the RADLER selector and analysis pipeline. Public datasets, trained segmentation checkpoints, derived experiment artifacts, and raw hardware-measurement logs are not redistributed.

## Reproduction Scope

Directly supported by this repository, with user-supplied local inputs where needed:

- Contextual feature extraction from image folders.
- Generation of width labels using the relative IoU-regret criterion.
- Training, saving, and loading of the one-vs-rest SVM selector.
- Selector evaluation, paired confidence intervals and significance tests, and diagnostic ablations.
- Aggregation of per-image energy measurements over a test set.
- Regeneration of the workflow figure and IoU-energy trade-off figures.

Reproducible after downloading the public datasets and training or supplying compatible slimmable segmentation checkpoints:

- SU-Net and SSU-Net segmentation training.
- Per-width segmentation evaluation for 25%, 50%, 75%, and 100% widths.
- Latency/FPS benchmarking of trained or supplied checkpoints.

Not directly reproducible from the public repository alone:

- The exact paper results that depend on cached IoU tables, trained checkpoints, trained selectors, locally prepared split files, and raw Jetson/Monsoon measurement logs.

## Repository Structure

### Core Package

- `radler/features.py`: contextual image feature extraction used by the selector.
- `radler/width_labels.py`: generation of width labels using the relative IoU-regret criterion.
- `radler/selector.py`: training, saving, and loading of the one-vs-rest SVM selector.
- `radler/evaluation.py`: selector diagnostics, paired confidence intervals and significance tests, and adaptive baselines.
- `radler/energy.py`: aggregation of per-image energy measurements over a test set.
- `radler/segmentation.py`: dataset loaders for externally downloaded datasets, per-width segmentation metrics, and FPS benchmarking helpers.
- `radler/models/`: slimmable SU-Net and SSU-Net backbone definitions reused from SqueezeSlimU-Net.

### Scripts

- `scripts/extract_features_from_folder.py`: extracts contextual features from image folders.
- `scripts/make_width_labels.py`: creates RADLER width labels from per-width IoU tables.
- `scripts/train_selector.py`: trains the SVM selector from feature and label tables.
- `scripts/evaluate_selector.py`: evaluates a trained selector on held-out features and IoUs.
- `scripts/train_segmentation_backbone.py`: trains a slimmable SU-Net or SSU-Net backbone.
- `scripts/evaluate_segmentation_widths.py`: evaluates all widths of a trained slimmable segmentation checkpoint.
- `scripts/benchmark_segmentation_widths.py`: measures model-only per-width latency and FPS with synthetic RGB inputs.
- `scripts/analyze_selector_artifacts.py`: reproduces selector diagnostics, uncertainty analysis, simple baselines, feature ablations, classifier ablations, and training-label summaries from cached paper artifacts.
- `scripts/regenerate_tradeoff_figures.py`: regenerates the IoU-energy trade-off plots used in the manuscript.
- `scripts/make_radler_overview_figure.py`: regenerates the RADLER workflow figure.

### Configuration

- `configs/paper_cases.json`: legacy paper-case artifact names used by `scripts/analyze_selector_artifacts.py`.
- `configs/feature_groups.json`: feature-group definitions used for diagnostic ablations.
- `configs/energy_measurements_template.csv`: template for per-width energy measurements.

### Documentation

- `docs/CURATION_NOTES.md`: notes on how this public release was curated.
- `data/README.md`, `artifacts/README.md`, and `results/README.md`: placeholder directories for local, ignored inputs and outputs.

### Artifacts and Results

- `data/`: place downloaded datasets here, or keep datasets outside the repository.
- `artifacts/`: place local checkpoints, feature tables, IoU tables, trained selectors, and energy logs here.
- `results/`: default destination for generated tables, JSON summaries, and figures.

## Excluded Data and Artifacts

The following files are not distributed in this repository because they contain datasets, large derived artifacts, trained models, or hardware-measurement outputs:

- Dataset files, image folders, masks, and dataset archives.
- Trained segmentation checkpoints and serialized SVM selectors used in the paper.
- Cached feature tables, per-width IoU tables, generated NumPy arrays, and generated CSV outputs used in the paper.
- Raw Jetson/Monsoon power-measurement logs.
- Generated manuscript PDFs and LaTeX build artifacts.

File extensions are not excluded by themselves. The scripts can create CSV, pickle, NumPy, JSON, checkpoint, and figure outputs locally; those local outputs should normally remain uncommitted unless they are intentionally published as a separate data/model release.

## Expected Artifact Layout

For new work, prefer clear case names such as:

```text
artifacts/
  paper_cases/
    agriadapt_ssu_net/
    agriadapt_su_net/
    tobacco_ssu_net/
    tobacco_su_net/
```

The paper-analysis script still expects the legacy layout recorded in `configs/paper_cases.json`:

```text
artifacts/
  adaptation/
    garage/
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

These are legacy experiment-directory names inherited from the original project. In this release, `garage` is the adaptation-artifact workspace, `geok_*` denotes the AgriAdapt cases, `squeeze` denotes SSU-Net, and `slim` denotes SU-Net.

| Human-readable case | Configured directory | Configured selector file |
| --- | --- | --- |
| AgriAdapt SSU-Net | `geok_squeeze_final` | `trained_classifier_geoksqueeze_100.sav` |
| AgriAdapt SU-Net | `geok_slim_final` | `trained_classifier_geokslim_10.sav` |
| Tobacco SSU-Net | `tobacco_squeeze_final` | `trained_classifier_tobaccosqueeze_01.sav` |
| Tobacco SU-Net | `tobacco_slim_final` | `trained_classifier_tobaccoslim_1.sav` |

## Prerequisites

- Python versions are not pinned in `requirements.txt`. This release was checked in a Python 3.9.6 environment; the Jetson measurements reported in the paper used Python 3.8.10 and PyTorch 1.12.0 on the embedded device.
- The public code was checked on macOS 26.5.1. The embedded measurements in the paper were collected on an NVIDIA Jetson Nano running Ubuntu 20.04.6 LTS.
- CUDA is optional for feature extraction, selector training, selector evaluation, and paper-analysis scripts. CUDA is useful for segmentation training/evaluation and is selected automatically by scripts using `--device auto` when available.
- Jetson latency and power profiling require a separate embedded environment and external power-measurement hardware. The generic benchmark helper in this repository is not a substitute for controlled Jetson/Monsoon profiling.
- The repository itself is small, about 500 KB in this curated release. Disk use after setup depends on the downloaded datasets, trained checkpoints, cached IoU tables, and generated local artifacts.

## Installation and Basic Usage

### Installation

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### Feature Extraction

```bash
python scripts/extract_features_from_folder.py \
  --image-dir data/agriadapt/train/images \
  --output artifacts/features/agriadapt_train_features.csv
```

The output includes `image_path` for traceability. When training a selector, either remove nonnumeric columns from the feature table or pass them through `--drop-columns`.

### Width-Label Generation

```bash
python scripts/make_width_labels.py \
  --input artifacts/ious/agriadapt_train_width_ious.csv \
  --output artifacts/labels/agriadapt_train_width_labels.csv \
  --iou-columns iou_025 iou_050 iou_075 iou_100 \
  --tolerance 0.10
```

The tolerance is relative IoU regret: `(IoU_max - IoU_width) / IoU_max <= tolerance`.

### Selector Training

```bash
python scripts/train_selector.py \
  --features artifacts/features/agriadapt_train_features.csv \
  --labels artifacts/labels/agriadapt_train_width_labels.csv \
  --label-column width_class \
  --drop-columns image_path index width_label width_class \
  --output artifacts/selectors/agriadapt_ssu_selector.joblib \
  --metadata results/agriadapt_ssu_selector_metadata.json \
  --cv-splits 5 \
  --seed 123
```

### Selector Evaluation

```bash
python scripts/evaluate_selector.py \
  --selector artifacts/selectors/agriadapt_ssu_selector.joblib \
  --features artifacts/features/agriadapt_test_features.csv \
  --labels artifacts/labels/agriadapt_test_width_labels_and_ious.csv \
  --label-column width_class \
  --iou-columns iou_025 iou_050 iou_075 iou_100 \
  --output results/agriadapt_ssu_selector_eval.json
```

### Segmentation Training and Evaluation

Train a slimmable backbone:

```bash
python scripts/train_segmentation_backbone.py \
  --architecture ssu \
  --train-images data/agriadapt/train/images \
  --train-labels data/agriadapt/train/labels \
  --val-images data/agriadapt/val/images \
  --val-labels data/agriadapt/val/labels \
  --label-format yolo \
  --image-size 512 \
  --epochs 100 \
  --batch-size 8 \
  --output artifacts/checkpoints/agriadapt_ssu.pt \
  --history results/agriadapt_ssu_history.json
```

Evaluate all widths of a trained slimmable segmentation checkpoint:

```bash
python scripts/evaluate_segmentation_widths.py \
  --architecture ssu \
  --checkpoint artifacts/checkpoints/agriadapt_ssu.pt \
  --images data/agriadapt/test/images \
  --labels data/agriadapt/test/labels \
  --label-format yolo \
  --image-size 512 \
  --output results/agriadapt_ssu_width_metrics.csv
```

Both YOLO labels and class-valued masks are supported. For YOLO annotations, use `--label-format yolo --labels <label-dir>`. For class masks, use `--label-format mask --masks <mask-dir>`; mask values listed in `--mask-weed-values` are treated as weed pixels.

### Latency Benchmarking

```bash
python scripts/benchmark_segmentation_widths.py \
  --architecture ssu \
  --checkpoint artifacts/checkpoints/agriadapt_ssu.pt \
  --image-size 512 \
  --batch-size 1 \
  --warmup 10 \
  --runs 100 \
  --device auto \
  --output results/agriadapt_ssu_width_fps.csv
```

This benchmark uses synthetic RGB inputs and measures model forward-pass latency only. It does not include image decoding, dataset loading, feature extraction, SVM selection, or hardware power measurement. The Jetson results reported in the paper were obtained separately with external measurement hardware.

### Paper-Analysis Scripts

```bash
python scripts/analyze_selector_artifacts.py \
  --repo artifacts \
  --out results/paper_analysis_outputs \
  --seed 123
```

This command requires the cached paper artifacts under `artifacts/adaptation/garage/<case>/`.

### Figure Regeneration

```bash
mkdir -p figs
python scripts/make_radler_overview_figure.py
python scripts/regenerate_tradeoff_figures.py
```

The overview-figure script requires `rsvg-convert` from librsvg. The trade-off figures are regenerated from the operating points hard-coded in `scripts/regenerate_tradeoff_figures.py`.

## Minimal End-to-End Example for a New Dataset

```bash
# 1. Prepare local image and annotation folders under data/<dataset>/.
# 2. Train or provide a compatible slimmable checkpoint.
python scripts/train_segmentation_backbone.py --architecture ssu --train-images data/my_dataset/train/images --train-labels data/my_dataset/train/labels --val-images data/my_dataset/val/images --val-labels data/my_dataset/val/labels --label-format yolo --image-size 512 --output artifacts/checkpoints/my_dataset_ssu.pt

# 3. Evaluate all widths and create a per-width metric table.
python scripts/evaluate_segmentation_widths.py --architecture ssu --checkpoint artifacts/checkpoints/my_dataset_ssu.pt --images data/my_dataset/train/images --labels data/my_dataset/train/labels --label-format yolo --image-size 512 --output artifacts/ious/my_dataset_train_width_ious.csv

# 4. Extract contextual features.
python scripts/extract_features_from_folder.py --image-dir data/my_dataset/train/images --output artifacts/features/my_dataset_train_features.csv

# 5. Generate width labels for a chosen tolerance.
python scripts/make_width_labels.py --input artifacts/ious/my_dataset_train_width_ious.csv --output artifacts/labels/my_dataset_train_width_labels.csv --tolerance 0.10

# 6. Train and evaluate the selector.
python scripts/train_selector.py --features artifacts/features/my_dataset_train_features.csv --labels artifacts/labels/my_dataset_train_width_labels.csv --drop-columns image_path index width_label width_class --output artifacts/selectors/my_dataset_selector.joblib
python scripts/evaluate_selector.py --selector artifacts/selectors/my_dataset_selector.joblib --features artifacts/features/my_dataset_test_features.csv --labels artifacts/labels/my_dataset_test_width_labels_and_ious.csv --output results/my_dataset_selector_eval.json
```

After selector evaluation, use `scripts/aggregate_energy.py` if you have per-width energy measurements and per-image selected-width classes.

## File Formats

Feature table:

| image_path | mean_brightness | std_brightness | ... | glcm_homogeneity_4 |
| --- | ---: | ---: | ---: | ---: |
| `data/agriadapt/train/images/frame_0001.jpg` | 118.2 | 31.4 | ... | 0.42 |

Required content: one row per image, the 39 default contextual feature columns from `radler/features.py`, and optionally `image_path`. Feature columns used for training should be numeric.

Per-width IoU table:

| image_id | iou_025 | iou_050 | iou_075 | iou_100 |
| --- | ---: | ---: | ---: | ---: |
| `frame_0001` | 0.521 | 0.552 | 0.558 | 0.561 |

Required content: one row per image and four IoU columns. The default column names are `iou_025`, `iou_050`, `iou_075`, and `iou_100`; alternative names can be passed with `--iou-columns`.

Width-label table:

| image_id | iou_025 | iou_050 | iou_075 | iou_100 | width_label | width_class |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `frame_0001` | 0.521 | 0.552 | 0.558 | 0.561 | 0.50 | 1 |

Required content: a `width_class` column with classes `0, 1, 2, 3` corresponding to `25%, 50%, 75%, 100%`. `scripts/make_width_labels.py` adds `width_label` and `width_class` to the input IoU table.

Selector evaluation output:

| field | meaning |
| --- | --- |
| `target_counts` | Number of target labels for 25/50/75/100% widths. |
| `selected_counts` | Number of predicted selections for 25/50/75/100% widths. |
| `exact_accuracy` | Exact width-label classification accuracy. |
| `average_width` | Mean selected width in percent. |
| `mean_iou` | Mean IoU obtained by the selected widths. |
| `paired_vs_full_width` | Paired IoU difference, confidence interval, and sign-flip p value against full width. |

Energy table:

| case | width_class | width_percent | per_image_energy_uAh | notes |
| --- | ---: | ---: | ---: | --- |
| `agriadapt_ssu` | 0 | 25 | 1.15 | local measurement |

Required content for `scripts/aggregate_energy.py`: `width_class` and `per_image_energy_uAh`. The selection table must contain a `selected_class` column unless another name is passed with `--selected-class-column`.

## Data Sources

- AgriAdapt Weed Detection dataset: https://gitlab.fri.uni-lj.si/lrk/agriadapt. The paper uses the predefined project split: 363 training images, 84 validation images, and 50 final test images.
- Tobacco Aerial Dataset: https://doi.org/10.1016/j.atech.2022.100142 and https://www.sciencedirect.com/science/article/pii/S277237552200106X. The paper uses campaign/partition 2, with 936 images split into 748 training images and 188 final test images using the fixed image-level split associated with the code.

For YOLO-style labels, the scripts expect:

```text
split/
  images/
    frame_0001.jpg
  labels/
    frame_0001.txt
```

For class-valued masks, pass `--label-format mask --images <image-dir> --masks <mask-dir>`.

The paper experiments use the public datasets together with derived artifacts such as split definitions, contextual-feature tables, per-width IoU tables, trained selectors, segmentation checkpoints, and energy measurements. These derived artifacts are not all redistributed. The repository provides the scripts, configuration files, and directory templates required to regenerate them when the underlying datasets and checkpoints are available.

## Paper Configurations

The selector uses MinMax scaling and one-vs-rest SVM classification. Hyperparameter tuning uses 5-fold stratified cross-validation with shuffling and random seed 123. The search grid is implemented in `radler/selector.py`.

| Case | Split definition | Reported tolerance | SVM kernel | C | Gamma | Class weighting | Legacy artifact directory |
| --- | --- | --- | --- | ---: | --- | --- | --- |
| AgriAdapt SSU-Net | 363 train / 84 validation / 50 test | 0.10 | sigmoid | 10 | auto | none | `geok_squeeze_final` |
| AgriAdapt SU-Net | 363 train / 84 validation / 50 test | 0.01 | sigmoid | 10 | auto | none | `geok_slim_final` |
| Tobacco SSU-Net | campaign 2, 748 train / 188 test | not encoded in `paper_cases.json`; see prepared selector artifact | sigmoid | 1 | scale | none | `tobacco_squeeze_final` |
| Tobacco SU-Net | campaign 2, 748 train / 188 test | 0.001 in the trade-off figure | polynomial | 100 | auto | none | `tobacco_slim_final` |

## Paper Results

- Figure 1: `scripts/make_radler_overview_figure.py`.
- Figures 7-8: `scripts/regenerate_tradeoff_figures.py`.
- Tables 7-10: `scripts/analyze_selector_artifacts.py`.
- Static-width segmentation metrics: `scripts/evaluate_segmentation_widths.py`.
- Latency/FPS: `scripts/benchmark_segmentation_widths.py`.

The exact manuscript tables require the cached feature tables, width-label tables, trained selectors, segmentation checkpoints, and local measurement artifacts described above.

## Methodological Notes

- The reported RADLER selector uses the full 39-dimensional contextual feature vector after MinMax scaling.
- Correlation-pruned and single-feature-group variants are evaluated only as diagnostic ablations.
- The tolerance parameter represents relative IoU regret: `(IoU_max - IoU_width) / IoU_max <= tolerance`.
- The selector is a one-vs-rest SVM trained to predict the narrowest acceptable width label.
- Backbone training follows slimmable-network practice by accumulating gradients over widths in descending order before each optimizer update.
- The reported Jetson measurements characterize the vision pipeline and should not be interpreted as full UAV mission-level battery measurements.

## Backbone Source

SU-Net and SSU-Net are reused from the SqueezeSlimU-Net work:

```bibtex
@article{machidon2025squeezeslimunet,
  author  = {Machidon, Alina L. and Krasovec, Andraz and Pejovic, Veljko and Machidon, Octavian M.},
  title   = {SqueezeSlimU-Net: An Adaptive and Efficient Segmentation Architecture for Real-Time UAV Weed Detection},
  journal = {IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing},
  year    = {2025},
  volume  = {18},
  pages   = {5749--5764},
  doi     = {10.1109/JSTARS.2025.3536175}
}
```

RADLER adds the contextual width-selection layer, relative IoU-regret labeling, operating-point analysis, selector diagnostics, and embedded energy evaluation.

## Citation

Use the final IEEE Access citation once the paper is published. Until then:

```bibtex
@article{machidon2026radler,
  author  = {Machidon, Alina L. and Krasovec, Andraz and Igret, Ioana C. and Pejovic, Veljko and Machidon, Octavian M.},
  title   = {RADLER: Context-Aware Slimmable Model Selection for Energy-Efficient Real-Time UAV Weed Detection},
  journal = {IEEE Access},
  year    = {2026},
  note    = {Manuscript under review}
}
```

## Limitations of This Release

- Exact paper reproduction requires external datasets and derived artifacts that are not distributed in this repository.
- Raw Jetson/Monsoon logs are not included.
- The repository does not reproduce a full UAV flight pipeline.
- The benchmark helper measures model latency and is not a replacement for controlled hardware power profiling.

## License

No open-source license is currently provided. The source code may be viewed, but reuse, modification, and redistribution require permission from the copyright holders.

## Contact

For reproducibility questions, contact Octavian M. Machidon at `octavian.machidon@fri.uni-lj.si` or open a GitHub Issue in this repository.
