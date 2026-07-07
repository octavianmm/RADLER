"""Contextual image features used by RADLER's lightweight selector.

The default extractor returns the 39-feature contextual vector used by the
reported RADLER operating points. Optional wavelet descriptors can be enabled
for diagnostic ablations, but they are not part of the default deployed vector.
"""

from __future__ import annotations

from collections import OrderedDict
from typing import Mapping

import cv2
import numpy as np
import pandas as pd
from skimage import color, img_as_ubyte
from skimage.feature import graycomatrix, graycoprops


DEFAULT_FEATURE_COLUMNS = [
    "mean_brightness",
    "std_brightness",
    "max_brightness",
    "min_brightness",
    "no_bins",
    "contrast_hue_hist",
    "std_hue_arc",
    "contrast",
    "mean_saturation",
    "std_saturation",
    "max_saturation",
    "min_saturation",
    "keypoints",
    "ExG_ExR",
    "CIVE_index",
    "glcm_contrast_1",
    "glcm_contrast_2",
    "glcm_contrast_3",
    "glcm_contrast_4",
    "glcm_correlation_1",
    "glcm_correlation_2",
    "glcm_correlation_3",
    "glcm_correlation_4",
    "glcm_dissimilarity_1",
    "glcm_dissimilarity_2",
    "glcm_dissimilarity_3",
    "glcm_dissimilarity_4",
    "glcm_asm_1",
    "glcm_asm_2",
    "glcm_asm_3",
    "glcm_asm_4",
    "glcm_energy_1",
    "glcm_energy_2",
    "glcm_energy_3",
    "glcm_energy_4",
    "glcm_homogeneity_1",
    "glcm_homogeneity_2",
    "glcm_homogeneity_3",
    "glcm_homogeneity_4",
]

WAVELET_FEATURE_COLUMNS = [
    "dwt_decomposition_1",
    "dwt_decomposition_2",
    "dwt_decomposition_3",
]


def _as_uint8_bgr(image: np.ndarray) -> np.ndarray:
    image = np.asarray(image)
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("Expected an HxWx3 RGB/BGR image array.")
    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)
    return image


def _brightness_features(image: np.ndarray) -> Mapping[str, float]:
    yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
    y, _, _ = cv2.split(yuv.astype(float))
    return {
        "mean_brightness": float(np.mean(y)),
        "std_brightness": float(np.std(y)),
        "max_brightness": float(np.max(y)),
        "min_brightness": float(np.min(y)),
    }


def _hue_saturation_features(image: np.ndarray) -> Mapping[str, float]:
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv.astype(float))
    hist = cv2.calcHist([hsv], [0], None, [20], [0, 256])
    threshold = 0.01 * float(np.max(hist))
    significant_bins = float(np.sum(hist >= threshold))
    sorted_hist = np.sort(hist.ravel())
    hue_contrast = float(sorted_hist[-1] - sorted_hist[-2]) if len(sorted_hist) > 1 else 0.0
    contrast = 0.0 if np.max(v) <= np.min(v) else float(np.std(v) / (np.max(v) - np.min(v)))
    return {
        "no_bins": significant_bins,
        "contrast_hue_hist": hue_contrast,
        "std_hue_arc": float(np.std(h)),
        "contrast": contrast,
        "mean_saturation": float(np.mean(s)),
        "std_saturation": float(np.std(s)),
        "max_saturation": float(np.max(s)),
        "min_saturation": float(np.min(s)),
    }


def _keypoint_count(image: np.ndarray) -> float:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    try:
        detector = cv2.SIFT_create()
    except AttributeError:
        detector = cv2.ORB_create()
    return float(len(detector.detect(gray, None)))


def _vegetation_features(image: np.ndarray) -> Mapping[str, float]:
    red = image[:, :, 0].astype(float)
    green = image[:, :, 1].astype(float)
    blue = image[:, :, 2].astype(float)
    excess_green = 2 * green - red - blue
    excess_red = 1.4 * red - green
    exg_exr = excess_green - excess_red
    _, binary_exg_exr = cv2.threshold(exg_exr, 0, 255, cv2.THRESH_BINARY)
    cive = red * 0.441 - green * 0.811 + blue * 0.385 + 18.78745
    _, binary_cive = cv2.threshold(
        cive.astype(np.uint8), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    return {
        "ExG_ExR": float(np.count_nonzero(binary_exg_exr) / binary_exg_exr.size),
        "CIVE_index": float(np.count_nonzero(binary_cive) / binary_cive.size),
    }


def _glcm_features(image: np.ndarray) -> Mapping[str, float]:
    gray = img_as_ubyte(color.rgb2gray(image))
    bins = np.array([0, 16, 32, 48, 64, 80, 96, 112, 128, 144, 160, 176, 192, 208, 224, 240, 255])
    digitized = np.digitize(gray, bins)
    glcm = graycomatrix(
        digitized,
        distances=[1],
        angles=[0, np.pi / 4, np.pi / 2, 3 * np.pi / 4],
        levels=int(digitized.max()) + 1,
        symmetric=True,
        normed=True,
    )
    features: OrderedDict[str, float] = OrderedDict()
    for prop in ["contrast", "correlation", "dissimilarity", "ASM", "energy", "homogeneity"]:
        values = np.ravel(graycoprops(glcm, prop))
        normalized_name = "asm" if prop == "ASM" else prop
        for idx, value in enumerate(values, start=1):
            features[f"glcm_{normalized_name}_{idx}"] = float(value)
    return features


def _wavelet_features(image: np.ndarray) -> Mapping[str, float]:
    try:
        import pywt
    except ImportError:
        return {name: 0.0 for name in WAVELET_FEATURE_COLUMNS}
    gray = color.rgb2gray(image)
    coeffs = pywt.wavedec2(gray, wavelet="haar", level=1)
    _, (horizontal, vertical, diagonal) = coeffs
    return {
        "dwt_decomposition_1": float(np.mean(np.abs(horizontal))),
        "dwt_decomposition_2": float(np.mean(np.abs(vertical))),
        "dwt_decomposition_3": float(np.mean(np.abs(diagonal))),
    }


def extract_context_features(image: np.ndarray, include_wavelet: bool = False) -> OrderedDict[str, float]:
    """Return RADLER contextual features for one image.

    Parameters
    ----------
    image:
        HxWx3 image array. The original pipeline used OpenCV-style BGR arrays.
    include_wavelet:
        Add optional wavelet descriptors for diagnostic feature-group ablations.
    """
    image = _as_uint8_bgr(image)
    features: OrderedDict[str, float] = OrderedDict()
    features.update(_brightness_features(image))
    features.update(_hue_saturation_features(image))
    features["keypoints"] = _keypoint_count(image)
    features.update(_vegetation_features(image))
    features.update(_glcm_features(image))
    if include_wavelet:
        features.update(_wavelet_features(image))
    return features


def extract_feature_frame(images: list[np.ndarray], include_wavelet: bool = False) -> pd.DataFrame:
    """Extract contextual features for a list of images."""
    rows = [extract_context_features(image, include_wavelet=include_wavelet) for image in images]
    return pd.DataFrame(rows)
