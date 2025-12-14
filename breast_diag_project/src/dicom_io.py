"""Utilities for reading DICOM images and preparing tensors."""
from __future__ import annotations

from typing import List, Tuple

import numpy as np
import pydicom


def read_dicom_header(path: str) -> pydicom.dataset.Dataset:
    """Read a DICOM file header without pixel data."""

    return pydicom.dcmread(path, stop_before_pixels=True, force=True)


def _sort_paths_by_instance_number(image_paths: List[str]) -> List[Tuple[int, str]]:
    sorted_paths = []
    for path in image_paths:
        ds = pydicom.dcmread(path, stop_before_pixels=False, force=True)
        instance_number = int(getattr(ds, "InstanceNumber", 0))
        sorted_paths.append((instance_number, path, ds))
    sorted_paths.sort(key=lambda x: x[0])
    return [(inst, path) for inst, path, _ in sorted_paths]


def load_series_pixel_array(image_paths: List[str]) -> np.ndarray:
    """Load and stack a DICOM series into a numpy array.

    Args:
        image_paths: List of file paths belonging to the same series.

    Returns:
        Numpy array of shape [1, D, H, W] with dtype float32.
    """

    if not image_paths:
        raise ValueError("No image paths provided for series loading")

    sorted_with_paths = _sort_paths_by_instance_number(image_paths)
    slices = []
    for _, path in sorted_with_paths:
        ds = pydicom.dcmread(path, stop_before_pixels=False, force=True)
        pixel_array = ds.pixel_array.astype(np.float32)
        slices.append(pixel_array)
    volume = np.stack(slices, axis=0)  # [D, H, W]
    volume = volume[np.newaxis, ...]  # [1, D, H, W]
    return volume


def zscore_normalize(volume: np.ndarray) -> np.ndarray:
    """Z-score normalize the input volume."""

    mean = volume.mean()
    std = volume.std()
    if std == 0:
        return volume - mean
    return (volume - mean) / std


def minmax_normalize(volume: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Min-max normalize the input volume to [0, 1]."""

    vmin = volume.min()
    vmax = volume.max()
    return (volume - vmin) / (vmax - vmin + eps)


__all__ = ["read_dicom_header", "load_series_pixel_array", "zscore_normalize", "minmax_normalize"]
