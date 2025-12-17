"""Dataset and DataModule for breast diagnosis training."""
from __future__ import annotations

import json
import math
import random
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import pytorch_lightning as pl

from breast_diag_project.src import dicom_io, sr_parser
from breast_diag_project.src.config import ExperimentConfig


def _zscore_tensor(tensor: torch.Tensor) -> torch.Tensor:
    mean = tensor.mean()
    std = tensor.std()
    if std == 0:
        # Avoid division by zero when the volume is constant.
        return tensor - mean
    return (tensor - mean) / std


def resample_to_spacing(
    tensor: torch.Tensor,
    current_spacing: Sequence[float] | None,
    target_spacing: Sequence[float] | None,
) -> torch.Tensor:
    if current_spacing is None or target_spacing is None:
        return tensor

    if len(current_spacing) != 3 or len(target_spacing) != 3:
        return tensor

    # Compute scaling factor per axis to convert current voxel spacing.
    scale = torch.tensor(current_spacing, dtype=torch.float32) / torch.tensor(
        target_spacing, dtype=torch.float32
    )
    _, depth, height, width = tensor.shape
    new_shape = (
        max(1, int(round(depth * scale[0].item()))),
        max(1, int(round(height * scale[1].item()))),
        max(1, int(round(width * scale[2].item()))),
    )

    tensor_5d = tensor.unsqueeze(0)
    resampled = F.interpolate(
        tensor_5d,
        size=new_shape,
        mode="trilinear",
        align_corners=False,
    )
    return resampled.squeeze(0)


def _compute_isotropic_spacing(
    spacing: Sequence[float] | None, method: str = "median"
) -> Sequence[float] | None:
    if spacing is None or len(spacing) != 3:
        return None

    arr = np.asarray(spacing, dtype=np.float32)
    if method == "min":
        iso = float(np.min(arr))
    elif method == "max":
        iso = float(np.max(arr))
    else:
        iso = float(np.median(arr))
    return (iso, iso, iso)


def preprocess_volume(
    volume: np.ndarray,
    spacing: Sequence[float] | None,
    clip_percentiles: Sequence[float] | None,
    resample_isotropic: bool,
    isotropic_method: str = "median",
    target_spacing_override: Sequence[float] | None = None,
    resize_shape: Sequence[int] | None = None,
) -> torch.Tensor:
    """Standardize a volume using clipping, resampling, and z-score normalization."""

    vol = volume.astype(np.float32, copy=False)
    if clip_percentiles is not None and len(clip_percentiles) == 2:
        low, high = np.percentile(vol, clip_percentiles)
        vol = np.clip(vol, low, high)
    vol = np.nan_to_num(vol)
    tensor = torch.from_numpy(vol)
    if tensor.ndim == 3:
        # Ensure channel dimension exists for downstream interpolation steps.
        tensor = tensor.unsqueeze(0)

    target_spacing = None
    if target_spacing_override is not None and len(target_spacing_override) == 3:
        target_spacing = target_spacing_override
    elif resample_isotropic:
        target_spacing = _compute_isotropic_spacing(spacing, isotropic_method)

    tensor = resample_to_spacing(tensor, spacing, target_spacing)
    if resize_shape is not None and len(resize_shape) == 3:
        resize_size = tuple(int(max(1, d)) for d in resize_shape)
        tensor = F.interpolate(
            tensor.unsqueeze(0), size=resize_size, mode="trilinear", align_corners=False
        ).squeeze(0)
    tensor = _zscore_tensor(tensor)
    return tensor.float()


def _random_rotate_3d(tensor: torch.Tensor, degrees: Sequence[float]) -> torch.Tensor:
    if len(degrees) != 2:
        return tensor

    angle = random.uniform(degrees[0], degrees[1]) * random.choice([-1, 1])
    radians = math.radians(angle)
    cos, sin = math.cos(radians), math.sin(radians)
    rotation = tensor.new_tensor(
        [[cos, -sin, 0, 0], [sin, cos, 0, 0], [0, 0, 1, 0]], dtype=torch.float32
    )
    grid = F.affine_grid(
        rotation.unsqueeze(0),
        size=(1, tensor.shape[0], tensor.shape[1], tensor.shape[2], tensor.shape[3]),
        align_corners=False,
    )
    rotated = F.grid_sample(
        tensor.unsqueeze(0), grid, mode="bilinear", padding_mode="border", align_corners=False
    )
    return rotated.squeeze(0)


def _elastic_deform_3d(
    tensor: torch.Tensor, alpha: float = 1.5, sigma: float = 8.0
) -> torch.Tensor:
    device = tensor.device
    _, depth, height, width = tensor.shape
    displacement = torch.randn((1, 3, depth, height, width), device=device, dtype=tensor.dtype)
    kernel_size = max(1, int(round(sigma)))
    if kernel_size % 2 == 0:
        kernel_size += 1
    padding = kernel_size // 2
    displacement = F.avg_pool3d(displacement, kernel_size=kernel_size, stride=1, padding=padding)
    displacement = displacement * (alpha / max(depth, height, width))

    z_lin = torch.linspace(-1, 1, depth, device=device, dtype=tensor.dtype)
    y_lin = torch.linspace(-1, 1, height, device=device, dtype=tensor.dtype)
    x_lin = torch.linspace(-1, 1, width, device=device, dtype=tensor.dtype)
    zz, yy, xx = torch.meshgrid(z_lin, y_lin, x_lin, indexing="ij")
    base_grid = torch.stack((xx, yy, zz), dim=-1)
    disp = displacement.squeeze(0).permute(1, 2, 3, 0)
    norm_factors = torch.tensor([width / 2, height / 2, depth / 2], device=device, dtype=tensor.dtype)
    disp = disp / norm_factors
    grid = base_grid + disp

    deformed = F.grid_sample(
        tensor.unsqueeze(0), grid.unsqueeze(0), mode="bilinear", padding_mode="border", align_corners=False
    )
    return deformed.squeeze(0)


def _intensity_augment(tensor: torch.Tensor, gamma_range: Sequence[float], noise_std: float) -> torch.Tensor:
    if len(gamma_range) != 2:
        gamma_range = (0.9, 1.1)
    gamma = random.uniform(gamma_range[0], gamma_range[1])
    signed = torch.sign(tensor)
    tensor = signed * torch.pow(tensor.abs() + 1e-6, gamma)
    if noise_std > 0:
        tensor = tensor + torch.randn_like(tensor) * float(noise_std)
    return tensor


def build_randaugment3d(augment_config: Dict[str, Any]) -> Optional[Callable[[torch.Tensor], torch.Tensor]]:
    if not augment_config or not augment_config.get("use_randaugment", False):
        return None

    num_layers = int(augment_config.get("num_layers", 0))
    degrees = augment_config.get("rotation_degrees", (5, 15))
    elastic_cfg = augment_config.get("elastic", {})
    intensity_cfg = augment_config.get("intensity", {})
    gamma_range = intensity_cfg.get("gamma_range", (0.9, 1.1))
    noise_std = float(intensity_cfg.get("noise_std", 0.0))

    transforms: list[Callable[[torch.Tensor], torch.Tensor]] = []

    if augment_config.get("flip_lr", True):
        transforms.append(lambda t: torch.flip(t, dims=[-1]))

    transforms.append(lambda t: _random_rotate_3d(t, degrees))
    transforms.append(
        lambda t: _elastic_deform_3d(
            t,
            alpha=float(elastic_cfg.get("alpha", 1.5)),
            sigma=float(elastic_cfg.get("sigma", 8.0)),
        )
    )
    transforms.append(lambda t: _intensity_augment(t, gamma_range, noise_std))

    def _augment(tensor: torch.Tensor) -> torch.Tensor:
        if not transforms or num_layers <= 0:
            return tensor
        chosen = random.sample(transforms, k=min(num_layers, len(transforms)))
        for fn in chosen:
            tensor = fn(tensor)
        return tensor

    return _augment


class BreastDiagnosisDataset(Dataset):
    """Dataset returning (image_tensor, target_tensor) pairs."""

    def __init__(self, manifest_df: pd.DataFrame, split: str, config: ExperimentConfig):
        self.manifest_df = manifest_df.reset_index(drop=True)
        self.split = split
        self.config = config
        self.label_config = config.labels
        self.cache_images = bool(config.data.get("cache_images", False))
        self.preprocess_cfg = config.data.get("preprocessing", {})
        self.clip_percentiles: Sequence[float] | None = tuple(
            self.preprocess_cfg.get("clip_percentiles", (1.0, 99.0))
        )
        self.resample_isotropic = bool(self.preprocess_cfg.get("resample_to_isotropic", True))
        self.isotropic_method = str(self.preprocess_cfg.get("isotropic_method", "median"))
        target_override = self.preprocess_cfg.get("target_spacing_override")
        self.target_spacing_override: Sequence[float] | None = (
            tuple(target_override) if isinstance(target_override, (list, tuple)) else None
        )
        resize_shape = self.preprocess_cfg.get("resize_shape")
        self.resize_shape: Sequence[int] | None = (
            tuple(int(x) for x in resize_shape) if isinstance(resize_shape, (list, tuple)) else None
        )
        self.augment_fn = build_randaugment3d(config.data.get("augmentations", {})) if split == "train" else None
        self._pt_cache: Dict[int, Dict[str, Any]] = {}

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.manifest_df)

    def _build_tensor_from_pt(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        if self.cache_images and idx in self._pt_cache:
            cached = self._pt_cache[idx]
            return cached["image"], cached["label"]

        row = self.manifest_df.iloc[idx]
        pt_path = Path(row["pt_path"]) if "pt_path" in row else None
        if not pt_path or not pt_path.exists():
            raise FileNotFoundError(f"Missing cached PT example for index {idx}: {pt_path}")

        data = torch.load(pt_path, map_location="cpu")
        image_tensor = data["image"].float()
        label_tensor = data.get("label")
        if label_tensor is None and "target" in data:
            label_tensor = torch.tensor(float(data["target"]), dtype=torch.float32)
        if label_tensor is None:
            raise KeyError(f"Cached PT example missing label: {pt_path}")

        if self.cache_images:
            self._pt_cache[idx] = {"image": image_tensor, "label": label_tensor}

        return image_tensor, label_tensor

    def __getitem__(self, idx: int):  # type: ignore[override]
        if "pt_path" not in self.manifest_df.columns:
            raise KeyError(
                "Manifest rows must include 'pt_path' for offline preprocessed examples."
            )

        image_tensor, target_tensor = self._build_tensor_from_pt(idx)
        if self.split == "train" and self.augment_fn is not None:
            image_tensor = self.augment_fn(image_tensor)
        return image_tensor, target_tensor


class BreastDiagnosisDataModule(pl.LightningDataModule):
    """LightningDataModule managing splits and loaders."""

    def __init__(self, config: ExperimentConfig):
        super().__init__()
        self.config = config
        self.batch_size = int(config.training.get("batch_size", 1))
        self.num_workers = int(config.data.get("num_workers", 0))
        self.train_df: Optional[pd.DataFrame] = None
        self.val_df: Optional[pd.DataFrame] = None
        self.test_df: Optional[pd.DataFrame] = None
        self.interim_dir = config.interim_dir
        self.preprocess_cfg = config.data.get("preprocessing", {})
        self.clip_percentiles: Sequence[float] | None = tuple(
            self.preprocess_cfg.get("clip_percentiles", (1.0, 99.0))
        )
        self.resample_isotropic = bool(self.preprocess_cfg.get("resample_to_isotropic", True))
        self.isotropic_method = str(self.preprocess_cfg.get("isotropic_method", "median"))
        target_override = self.preprocess_cfg.get("target_spacing_override")
        self.target_spacing_override: Sequence[float] | None = (
            tuple(target_override) if isinstance(target_override, (list, tuple)) else None
        )
        resize_shape = self.preprocess_cfg.get("resize_shape")
        self.resize_shape: Sequence[int] | None = (
            tuple(int(x) for x in resize_shape) if isinstance(resize_shape, (list, tuple)) else None
        )

    def setup(self, stage: Optional[str] = None):  # type: ignore[override]
        manifest_path = self.config.manifest_path
        df = pd.read_csv(manifest_path)

        # Ensure image_paths column is parsed into lists.
        df["image_paths"] = df["image_paths"].apply(lambda x: json.loads(x) if isinstance(x, str) else x)

        df = self._materialize_pt_cache(df)

        splits = self.config.data.get("train_val_test_split", [0.7, 0.15, 0.15])
        if len(splits) != 3 or not np.isclose(sum(splits), 1.0):
            raise ValueError("train_val_test_split must have three values summing to 1.0")

        rng = np.random.RandomState(42)
        indices = rng.permutation(len(df))
        n_train = int(len(df) * splits[0])
        n_val = int(len(df) * splits[1])

        train_idx = indices[:n_train]
        val_idx = indices[n_train : n_train + n_val]
        test_idx = indices[n_train + n_val :]

        self.train_df = df.iloc[train_idx].reset_index(drop=True)
        self.val_df = df.iloc[val_idx].reset_index(drop=True)
        self.test_df = df.iloc[test_idx].reset_index(drop=True)

    def _materialize_pt_cache(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create PT files for each manifest row if missing and record their paths."""

        pt_dir = self.interim_dir
        pt_dir.mkdir(parents=True, exist_ok=True)

        def _build_pt_path(row: pd.Series) -> Path:
            patient_id = str(row.get("patient_id", "patient")) or "patient"
            study_uid = str(row.get("study_uid", "study")) or "study"
            safe_study = study_uid.replace(".", "").replace(" ", "")
            filename = f"{patient_id}__{safe_study}.pt"
            return pt_dir / filename

        pt_paths = []
        for _, row in df.iterrows():
            pt_path = _build_pt_path(row)
            if not pt_path.exists():
                example = self._convert_row_to_pt(row)
                torch.save(example, pt_path)
            pt_paths.append(str(pt_path))

        df = df.copy()
        df["pt_path"] = pt_paths
        return df

    def _convert_row_to_pt(self, row: pd.Series) -> Dict[str, Any]:
        """Load a manifest row's DICOMs, standardize, and package as a PT example."""

        image_paths = row["image_paths"]
        if isinstance(image_paths, str):
            image_paths = json.loads(image_paths)

        volume, spacing = dicom_io.load_series_pixel_array(image_paths, return_spacing=True)
        image_tensor = preprocess_volume(
            volume,
            spacing,
            self.clip_percentiles,
            self.resample_isotropic,
            self.isotropic_method,
            self.target_spacing_override,
            self.resize_shape,
        )

        label = sr_parser.parse_sr_to_label(row["sr_path"], self.config.labels)
        label_tensor = torch.tensor(float(label.get("target", 0.0)), dtype=torch.float32)

        return {
            "image": image_tensor,
            "label": label_tensor,
            "patient_id": row.get("patient_id", ""),
            "study_uid": row.get("study_uid", ""),
            "sr_path": row.get("sr_path", ""),
            "image_paths": image_paths,
            "label_fields": label,
            "spacing": spacing,
        }

    def train_dataloader(self) -> DataLoader:  # type: ignore[override]
        assert self.train_df is not None, "DataModule not setup"
        dataset = BreastDiagnosisDataset(self.train_df, split="train", config=self.config)
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self) -> DataLoader:  # type: ignore[override]
        assert self.val_df is not None, "DataModule not setup"
        dataset = BreastDiagnosisDataset(self.val_df, split="val", config=self.config)
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

    def test_dataloader(self) -> DataLoader:  # type: ignore[override]
        assert self.test_df is not None, "DataModule not setup"
        dataset = BreastDiagnosisDataset(self.test_df, split="test", config=self.config)
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)


__all__ = ["BreastDiagnosisDataset", "BreastDiagnosisDataModule"]
