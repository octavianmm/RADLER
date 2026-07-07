#!/usr/bin/env python3
"""Extract RADLER contextual features from a folder of images."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--include-wavelet", action="store_true")
    args = parser.parse_args()

    import cv2
    from radler.features import extract_context_features

    rows = []
    for image_path in sorted(args.image_dir.rglob("*")):
        if image_path.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Could not read image: {image_path}")
        row = {"image_path": str(image_path)}
        row.update(extract_context_features(image, include_wavelet=args.include_wavelet))
        rows.append(row)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.output, index=False)


if __name__ == "__main__":
    main()
