"""Utilities for reading DICOM images and preparing tensors."""
from __future__ import annotations

from typing import List, Tuple

import numpy as np
import pydicom


def read_dicom_header(path: str) -> pydicom.dataset.Dataset:
    """Read a DICOM file header without pixel data."""

    return pydicom.dcmread(path, stop_before_pixels=True, force=True)


def _sort_paths_by_instance_number(image_paths: List[str]) -> List[Tuple[int, str, pydicom.dataset.Dataset]]:
    sorted_paths = []
    for path in image_paths:
        ds = pydicom.dcmread(path, stop_before_pixels=False, force=True)
        instance_number = int(getattr(ds, "InstanceNumber", 0))
        sorted_paths.append((instance_number, path, ds))
    sorted_paths.sort(key=lambda x: x[0])
    return sorted_paths


def _estimate_spacing(sorted_datasets: List[pydicom.dataset.Dataset]) -> Tuple[float, float, float]:
    first = sorted_datasets[0]
    try:
        pixel_spacing = getattr(first, "PixelSpacing", [1.0, 1.0])
        row_spacing, col_spacing = float(pixel_spacing[0]), float(pixel_spacing[1])
    except Exception:
        row_spacing = col_spacing = 1.0

    try:
        if len(sorted_datasets) > 1:
            positions = [np.array(getattr(ds, "ImagePositionPatient", [0, 0, i])) for i, ds in enumerate(sorted_datasets)]
            deltas = [abs(pos[2] - positions[0][2]) for pos in positions[1:]]
            slice_spacing = float(np.median(deltas)) if deltas else float(getattr(first, "SliceThickness", 1.0))
        else:
            slice_spacing = float(getattr(first, "SliceThickness", 1.0))
    except Exception:
        slice_spacing = 1.0

    return (slice_spacing, row_spacing, col_spacing)


def load_series_pixel_array(image_paths: List[str], return_spacing: bool = False):
    """Load and stack a DICOM series into a numpy array.

    Args:
        image_paths: List of file paths belonging to the same series.
        return_spacing: When True, also return (z, y, x) spacing tuple.

    Returns:
        Numpy array of shape [1, D, H, W] with dtype float32, optionally with spacing.
    """

    if not image_paths:
        raise ValueError("No image paths provided for series loading")

    sorted_with_paths = _sort_paths_by_instance_number(image_paths)
    slices = []
    datasets = []
    for _, path, ds in sorted_with_paths:
        pixel_array = ds.pixel_array.astype(np.float32)
        slices.append(pixel_array)
        datasets.append(ds)
    volume = np.stack(slices, axis=0)  # [D, H, W]
    volume = volume[np.newaxis, ...]  # [1, D, H, W]

    if return_spacing:
        spacing = _estimate_spacing(datasets)
        return volume, spacing
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
