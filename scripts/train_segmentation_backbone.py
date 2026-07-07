#!/usr/bin/env python3
"""Train a slimmable SU-Net/SSU-Net segmentation backbone."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from radler import settings
from radler.segmentation import (
    SegmentationFolderDataset,
    build_segmentation_model,
    evaluate_widths,
    resolve_device,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--architecture", required=True, choices=["slim", "squeeze", "su", "ssu"])
    parser.add_argument("--train-images", required=True, type=Path)
    parser.add_argument("--train-labels", type=Path)
    parser.add_argument("--train-masks", type=Path)
    parser.add_argument("--val-images", required=True, type=Path)
    parser.add_argument("--val-labels", type=Path)
    parser.add_argument("--val-masks", type=Path)
    parser.add_argument("--label-format", choices=["yolo", "mask"], default="yolo")
    parser.add_argument("--image-size", nargs="+", type=int, default=[512])
    parser.add_argument("--weed-class", type=int, default=1)
    parser.add_argument("--mask-weed-values", nargs="+", type=int, default=[2, 255])
    parser.add_argument("--widths", nargs="+", type=float, default=settings.WIDTHS)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--class-weights", nargs=2, type=float, default=[1.0, 1.0])
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--history", type=Path)
    return parser.parse_args()


def build_dataset(args: argparse.Namespace, split: str) -> SegmentationFolderDataset:
    return SegmentationFolderDataset(
        images_dir=getattr(args, f"{split}_images"),
        labels_dir=getattr(args, f"{split}_labels"),
        masks_dir=getattr(args, f"{split}_masks"),
        label_format=args.label_format,
        image_size=args.image_size,
        weed_class=args.weed_class,
        mask_weed_values=args.mask_weed_values,
    )


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    train_loader = DataLoader(
        build_dataset(args, "train"),
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False,
    )
    val_loader = DataLoader(build_dataset(args, "val"), batch_size=1, shuffle=False)

    model = build_segmentation_model(args.architecture, out_channels=2, device=device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    criterion = torch.nn.CrossEntropyLoss(
        weight=torch.tensor(args.class_weights, dtype=torch.float32, device=device)
    )

    best_iou = -np.inf
    history = []
    args.output.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for images, target in train_loader:
            images = images.to(device)
            target = target.to(device)
            optimizer.zero_grad()
            for width in sorted(args.widths, reverse=True):
                if hasattr(model, "set_width"):
                    model.set_width(float(width))
                logits = model(images)
                loss = criterion(logits, target)
                loss.backward()
                losses.append(float(loss.detach().cpu()))
            optimizer.step()

        val_rows = evaluate_widths(model, val_loader, widths=args.widths, device=device)
        val_iou = float(np.nanmean([row["iou"] for row in val_rows]))
        record = {
            "epoch": epoch,
            "train_loss": float(np.mean(losses)),
            "val_mean_iou": val_iou,
            "val_by_width": val_rows,
        }
        history.append(record)
        print(json.dumps(record))

        if val_iou > best_iou:
            best_iou = val_iou
            torch.save(model.state_dict(), args.output)

    if args.history:
        args.history.parent.mkdir(parents=True, exist_ok=True)
        args.history.write_text(json.dumps(history, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
