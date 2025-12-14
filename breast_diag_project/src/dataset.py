"""Dataset and DataModule for breast diagnosis training."""
from __future__ import annotations

import json
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
import pytorch_lightning as pl

from . import dicom_io, sr_parser
from .config import ExperimentConfig


class BreastDiagnosisDataset(Dataset):
    """Dataset returning (image_tensor, target_tensor) pairs."""

    def __init__(self, manifest_df: pd.DataFrame, split: str, config: ExperimentConfig):
        self.manifest_df = manifest_df.reset_index(drop=True)
        self.split = split
        self.config = config
        self.label_config = config.labels
        self.cache_images = bool(config.data.get("cache_images", False))
        self._image_cache: Dict[int, torch.Tensor] = {}

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.manifest_df)

    def _load_image(self, idx: int) -> torch.Tensor:
        if self.cache_images and idx in self._image_cache:
            return self._image_cache[idx]

        row = self.manifest_df.iloc[idx]
        image_paths = row["image_paths"]
        if isinstance(image_paths, str):
            image_paths = json.loads(image_paths)

        volume = dicom_io.load_series_pixel_array(image_paths)
        volume = dicom_io.zscore_normalize(volume)
        tensor = torch.from_numpy(volume)

        if self.cache_images:
            self._image_cache[idx] = tensor
        return tensor

    def _load_label(self, idx: int) -> torch.Tensor:
        row = self.manifest_df.iloc[idx]
        label = sr_parser.parse_sr_to_label(row["sr_path"], self.label_config)
        target = torch.tensor(float(label.get("target", 0)), dtype=torch.float32)
        return target

    def __getitem__(self, idx: int):  # type: ignore[override]
        image_tensor = self._load_image(idx)
        target_tensor = self._load_label(idx)
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

    def setup(self, stage: Optional[str] = None):  # type: ignore[override]
        manifest_path = self.config.manifest_path
        df = pd.read_csv(manifest_path)

        # Ensure image_paths column is parsed into lists.
        df["image_paths"] = df["image_paths"].apply(lambda x: json.loads(x) if isinstance(x, str) else x)

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
