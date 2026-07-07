"""Slimmable segmentation backbone definitions reused from SqueezeSlimU-Net."""

from .slim_squeeze_unet import SlimSqueezeUNet
from .slim_unet import SlimUNet

__all__ = ["SlimSqueezeUNet", "SlimUNet"]
