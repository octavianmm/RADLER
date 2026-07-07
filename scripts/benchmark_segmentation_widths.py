#!/usr/bin/env python3
"""Benchmark per-width SU-Net/SSU-Net latency and FPS."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from radler import settings
from radler.segmentation import (
    benchmark_widths,
    build_segmentation_model,
    load_state_dict,
    resolve_device,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--architecture", required=True, choices=["slim", "squeeze", "su", "ssu"])
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--image-size", nargs="+", type=int, default=[512])
    parser.add_argument("--widths", nargs="+", type=float, default=settings.WIDTHS)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--runs", type=int, default=100)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    model = build_segmentation_model(args.architecture, out_channels=2, device=device)
    if args.checkpoint:
        load_state_dict(model, args.checkpoint, device=device)

    rows = benchmark_widths(
        model,
        image_size=args.image_size,
        widths=args.widths,
        batch_size=args.batch_size,
        warmup=args.warmup,
        runs=args.runs,
        device=device,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.suffix.lower() == ".json":
        args.output.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    else:
        pd.DataFrame(rows).to_csv(args.output, index=False)


if __name__ == "__main__":
    main()
