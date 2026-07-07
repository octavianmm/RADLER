#!/usr/bin/env python3
"""Aggregate per-image width selections into final-test-set energy in mAh."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from radler.energy import aggregate_energy_mAh


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selections", required=True, type=Path, help="CSV with a selected width class column.")
    parser.add_argument("--energy", required=True, type=Path, help="CSV with width_class and per_image_energy_uAh columns.")
    parser.add_argument("--selected-class-column", default="selected_class")
    parser.add_argument("--selector-energy-uAh", type=float, default=0.0)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    selections = pd.read_csv(args.selections)
    energy = pd.read_csv(args.energy).sort_values("width_class")
    total = aggregate_energy_mAh(
        selections[args.selected_class_column].to_numpy(dtype=int),
        energy["per_image_energy_uAh"].to_numpy(dtype=float),
        selector_energy_uAh=args.selector_energy_uAh,
    )
    payload = {
        "n_images": int(len(selections)),
        "aggregate_energy_mAh": total,
        "selector_energy_uAh": args.selector_energy_uAh,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
