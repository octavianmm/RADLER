#!/usr/bin/env python3
"""Evaluate SU-Net/SSU-Net checkpoints at all slimmable widths."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from radler import settings
from radler.segmentation import (
    SegmentationFolderDataset,
    build_segmentation_model,
    evaluate_widths,
    load_state_dict,
    resolve_device,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--architecture", required=True, choices=["slim", "squeeze", "su", "ssu"])
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--images", required=True, type=Path)
    parser.add_argument("--labels", type=Path, help="YOLO label directory.")
    parser.add_argument("--masks", type=Path, help="Class-mask directory.")
    parser.add_argument("--label-format", choices=["yolo", "mask"], default="yolo")
    parser.add_argument("--image-size", nargs="+", type=int, default=[512])
    parser.add_argument("--weed-class", type=int, default=1)
    parser.add_argument("--mask-weed-values", nargs="+", type=int, default=[2, 255])
    parser.add_argument("--widths", nargs="+", type=float, default=settings.WIDTHS)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    dataset = SegmentationFolderDataset(
        images_dir=args.images,
        labels_dir=args.labels,
        masks_dir=args.masks,
        label_format=args.label_format,
        image_size=args.image_size,
        weed_class=args.weed_class,
        mask_weed_values=args.mask_weed_values,
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    model = build_segmentation_model(args.architecture, out_channels=2, device=device)
    load_state_dict(model, args.checkpoint, device=device)
    rows = evaluate_widths(model, loader, widths=args.widths, device=device)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.suffix.lower() == ".json":
        args.output.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    else:
        pd.DataFrame(rows).to_csv(args.output, index=False)


if __name__ == "__main__":
    main()
