"""Segmentation data, evaluation, and benchmarking helpers for RADLER."""

from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from PIL import Image, ImageDraw
from torch.utils.data import Dataset

from radler import settings
from radler.models import SlimSqueezeUNet, SlimUNet


IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff"}


def normalize_image_size(image_size: Iterable[int] | None) -> tuple[int, int] | None:
    """Return a PIL-compatible ``(width, height)`` size tuple."""
    if image_size is None:
        return None
    values = [int(x) for x in image_size]
    if len(values) == 1:
        return values[0], values[0]
    if len(values) == 2:
        return values[0], values[1]
    raise ValueError("image_size must contain one value or two values")


def image_to_tensor(image: Image.Image) -> torch.Tensor:
    """Convert an RGB PIL image to a float tensor with shape ``C,H,W``."""
    array = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(array.transpose(2, 0, 1))


class SegmentationFolderDataset(Dataset):
    """Load external segmentation data without bundling datasets in the repo.

    Supported annotation formats:
    - ``yolo``: one label file per image, with normalized YOLO rectangles or polygons.
    - ``mask``: one class-valued mask image per input image.
    """

    def __init__(
        self,
        images_dir: Path,
        labels_dir: Path | None = None,
        masks_dir: Path | None = None,
        label_format: str = "yolo",
        image_size: Iterable[int] | None = None,
        weed_class: int = 1,
        mask_weed_values: Iterable[int] = (2, 255),
    ) -> None:
        self.images_dir = Path(images_dir)
        self.labels_dir = Path(labels_dir) if labels_dir else None
        self.masks_dir = Path(masks_dir) if masks_dir else None
        self.label_format = label_format
        self.image_size = normalize_image_size(image_size)
        self.weed_class = int(weed_class)
        self.mask_weed_values = {int(x) for x in mask_weed_values}

        if self.label_format not in {"yolo", "mask"}:
            raise ValueError("label_format must be 'yolo' or 'mask'")
        if self.label_format == "yolo" and self.labels_dir is None:
            raise ValueError("labels_dir is required for YOLO labels")
        if self.label_format == "mask" and self.masks_dir is None:
            raise ValueError("masks_dir is required for class-valued masks")

        self.images = sorted(
            p for p in self.images_dir.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES
        )
        if not self.images:
            raise ValueError(f"No images found in {self.images_dir}")
        if self.masks_dir:
            self._masks_by_stem = {
                p.stem: p
                for p in self.masks_dir.iterdir()
                if p.suffix.lower() in IMAGE_SUFFIXES
            }
        else:
            self._masks_by_stem = {}

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        image_path = self.images[index]
        image = Image.open(image_path).convert("RGB")
        if self.image_size:
            image = image.resize(self.image_size, Image.Resampling.BILINEAR)

        width, height = image.size
        if self.label_format == "yolo":
            target = self._load_yolo_mask(image_path.stem, width, height)
        else:
            target = self._load_class_mask(image_path.stem, width, height)
        return image_to_tensor(image), torch.as_tensor(target, dtype=torch.long)

    def _load_yolo_mask(self, stem: str, width: int, height: int) -> np.ndarray:
        mask = np.zeros((height, width), dtype=np.int64)
        label_path = self.labels_dir / f"{stem}.txt"
        if not label_path.exists():
            return mask

        for raw_line in label_path.read_text(encoding="utf-8").splitlines():
            parts = raw_line.strip().split()
            if len(parts) < 5:
                continue
            class_id = int(float(parts[0]))
            if class_id != self.weed_class:
                continue
            coords = [float(x) for x in parts[1:]]
            if len(coords) == 4:
                self._paint_yolo_box(mask, coords, width, height)
            elif len(coords) >= 6 and len(coords) % 2 == 0:
                self._paint_yolo_polygon(mask, coords, width, height)
        return mask

    @staticmethod
    def _paint_yolo_box(
        mask: np.ndarray,
        coords: list[float],
        width: int,
        height: int,
    ) -> None:
        center_x, center_y, box_width, box_height = coords
        left = max(0, int(math.floor((center_x - box_width / 2.0) * width)))
        right = min(width, int(math.ceil((center_x + box_width / 2.0) * width)))
        top = max(0, int(math.floor((center_y - box_height / 2.0) * height)))
        bottom = min(height, int(math.ceil((center_y + box_height / 2.0) * height)))
        if right > left and bottom > top:
            mask[top:bottom, left:right] = 1

    @staticmethod
    def _paint_yolo_polygon(
        mask: np.ndarray,
        coords: list[float],
        width: int,
        height: int,
    ) -> None:
        points = [
            (coords[i] * width, coords[i + 1] * height)
            for i in range(0, len(coords), 2)
        ]
        polygon_mask = Image.new("L", (width, height), 0)
        ImageDraw.Draw(polygon_mask).polygon(points, outline=1, fill=1)
        mask[np.asarray(polygon_mask, dtype=bool)] = 1

    def _load_class_mask(self, stem: str, width: int, height: int) -> np.ndarray:
        mask_path = self._masks_by_stem.get(stem)
        if mask_path is None:
            raise FileNotFoundError(f"No mask found for image stem '{stem}'")
        mask_image = Image.open(mask_path)
        if mask_image.size != (width, height):
            mask_image = mask_image.resize((width, height), Image.Resampling.NEAREST)
        array = np.asarray(mask_image)
        if array.ndim == 3:
            array = array[..., 0]
        return np.isin(array.astype(int), list(self.mask_weed_values)).astype(np.int64)


def build_segmentation_model(
    architecture: str,
    out_channels: int = 2,
    device: str | torch.device | None = None,
) -> torch.nn.Module:
    """Build one of the slimmable segmentation backbones used in the paper."""
    name = architecture.lower()
    if name in {"slim", "su", "su-net", "sunet"}:
        model = SlimUNet(out_channels=out_channels)
    elif name in {"squeeze", "ssu", "ssu-net", "squeezeslim", "squeezeslimunet"}:
        model = SlimSqueezeUNet(out_channels=out_channels)
    else:
        raise ValueError(f"Unknown architecture: {architecture}")
    if device is not None:
        model = model.to(device)
    return model


def load_state_dict(model: torch.nn.Module, checkpoint: Path, device: str | torch.device) -> None:
    """Load a plain or wrapped PyTorch state dict into ``model``."""
    payload = torch.load(Path(checkpoint), map_location=device)
    if isinstance(payload, dict) and "state_dict" in payload:
        payload = payload["state_dict"]
    if isinstance(payload, dict) and all(str(k).startswith("module.") for k in payload):
        payload = {str(k)[7:]: v for k, v in payload.items()}
    model.load_state_dict(payload)


def resolve_device(device: str) -> torch.device:
    """Resolve ``auto`` to CUDA when available, otherwise CPU."""
    if device == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


class SegmentationMetricAccumulator:
    """Accumulate per-image binary weed-segmentation metrics."""

    def __init__(self, positive_class: int = 1) -> None:
        self.positive_class = positive_class
        self.values: dict[str, list[float]] = {
            "iou": [],
            "precision": [],
            "recall": [],
            "f1": [],
            "accuracy": [],
        }

    def update(self, logits: torch.Tensor, target: torch.Tensor) -> None:
        predicted = torch.argmax(logits.detach(), dim=1)
        if target.ndim == 4:
            target = torch.argmax(target, dim=1)
        target = target.detach()

        for pred_i, target_i in zip(predicted, target):
            pred_pos = pred_i == self.positive_class
            target_pos = target_i == self.positive_class
            tp = torch.count_nonzero(pred_pos & target_pos).item()
            fp = torch.count_nonzero(pred_pos & ~target_pos).item()
            fn = torch.count_nonzero(~pred_pos & target_pos).item()
            tn = torch.count_nonzero(~pred_pos & ~target_pos).item()

            union = tp + fp + fn
            precision_den = tp + fp
            recall_den = tp + fn
            f1_den = 2 * tp + fp + fn
            total = tp + fp + fn + tn

            self.values["iou"].append(np.nan if union == 0 else tp / union)
            self.values["precision"].append(
                np.nan if precision_den == 0 else tp / precision_den
            )
            self.values["recall"].append(np.nan if recall_den == 0 else tp / recall_den)
            self.values["f1"].append(np.nan if f1_den == 0 else (2 * tp) / f1_den)
            self.values["accuracy"].append((tp + tn) / total)

    def compute(self) -> dict[str, float]:
        return {
            key: float(np.nanmean(value)) if value else float("nan")
            for key, value in self.values.items()
        }


@torch.no_grad()
def evaluate_widths(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    widths: Iterable[float] = settings.WIDTHS,
    device: str | torch.device = "cpu",
) -> list[dict[str, float]]:
    """Evaluate a slimmable segmentation model at each requested width."""
    device = torch.device(device)
    model.to(device)
    model.eval()
    rows = []
    for width in widths:
        if hasattr(model, "set_width"):
            model.set_width(float(width))
        accumulator = SegmentationMetricAccumulator()
        for images, target in loader:
            images = images.to(device)
            target = target.to(device)
            logits = model(images)
            accumulator.update(logits, target)
        rows.append({"width": float(width), **accumulator.compute()})
    return rows


@torch.no_grad()
def benchmark_widths(
    model: torch.nn.Module,
    image_size: Iterable[int],
    widths: Iterable[float] = settings.WIDTHS,
    batch_size: int = 1,
    warmup: int = 10,
    runs: int = 100,
    device: str | torch.device = "cpu",
) -> list[dict[str, float]]:
    """Benchmark per-width latency and FPS with synthetic RGB inputs."""
    device = torch.device(device)
    width_px, height_px = normalize_image_size(image_size)
    model.to(device)
    model.eval()
    x = torch.rand(batch_size, 3, height_px, width_px, device=device)
    rows = []
    is_cuda = device.type == "cuda"

    for width in widths:
        if hasattr(model, "set_width"):
            model.set_width(float(width))
        for _ in range(warmup):
            model(x)
        if is_cuda:
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)

        start = time.perf_counter()
        for _ in range(runs):
            model(x)
        if is_cuda:
            torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - start

        latency_ms = elapsed * 1000.0 / runs
        fps = batch_size * runs / elapsed
        row = {
            "width": float(width),
            "batch_size": int(batch_size),
            "latency_ms": float(latency_ms),
            "fps": float(fps),
        }
        if is_cuda:
            row["peak_memory_mb"] = float(torch.cuda.max_memory_allocated(device) / 1e6)
        rows.append(row)
    return rows
