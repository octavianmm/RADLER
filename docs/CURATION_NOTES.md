# Curation Notes

This folder was prepared as a code-only public release for the RADLER paper.

## Source Scope

Included code covers the experiments described in the manuscript:

- slimmable SU-Net and SSU-Net model definitions reused from SqueezeSlimU-Net;
- generic training, per-width evaluation, and FPS benchmarking scripts for those
  slimmable segmentation backbones;
- RADLER contextual feature extraction;
- relative IoU-regret width-label generation;
- SVM width-selector training and inference;
- selector diagnostics, paired tests, simple baselines, feature ablations, and classifier ablations;
- aggregate energy calculations;
- manuscript figure regeneration scripts.

Excluded material:

- dataset images and masks;
- trained segmentation checkpoints;
- trained selector files;
- feature tables and per-width IoU tables;
- Jetson/Monsoon raw logs;
- manuscript LaTeX sources and generated PDFs.

## Historical Artifact Names

Some analysis scripts expect columns named `precision_025`, `precision_050`,
`precision_075`, and `precision_100` because the original working artifacts used
that naming convention. In the revised manuscript these columns are interpreted
as the per-width segmentation quality values used for IoU-regret/selector
diagnostics. If new artifacts are generated with clearer names such as
`iou_025`, either rename the columns or pass the desired column names to the
standalone scripts that expose `--iou-columns`.

## Missing Cluster-Only Material

The original research working tree referenced in notes was
`/home/omachidon/work/RADLER_revision/agriadapt`. Teleport access was not
available during this curation pass, so this release does not include any
cluster-only data artifacts. The public code is nevertheless structured around
the artifact names listed in `configs/paper_cases.json`, so those files can be
added separately without changing the code.

The accessible GitLab source tree and its `develop` and
`alinamachidon-main-patch-1fee` branches were checked for code-only material.
Useful ideas from branch-only training, evaluation, profiling, and Tobacco
inference scripts were folded into cleaned, path-independent release scripts.
Dataset files, checkpoints, cached KNN/SVM artifacts, generated PDFs, and raw
results remain excluded.
