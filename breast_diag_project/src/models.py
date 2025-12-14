"""Model definitions and factories."""
from __future__ import annotations

from typing import Any, Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl


class Simple3DCNN(pl.LightningModule):
    """A simple configurable 3D CNN for binary classification."""

    def __init__(self, model_config: Dict[str, Any], training_config: Dict[str, Any]):
        super().__init__()
        self.save_hyperparameters({"model": model_config, "training": training_config})
        self.training_config = training_config

        in_channels = int(model_config.get("in_channels", 1))
        num_classes = int(model_config.get("num_classes", 1))
        features: List[int] = list(model_config.get("features", [32, 64, 128]))
        dropout = float(model_config.get("dropout", 0.0))

        layers: List[nn.Module] = []
        current_channels = in_channels
        for feat in features:
            layers.append(nn.Conv3d(current_channels, feat, kernel_size=3, padding=1))
            layers.append(nn.BatchNorm3d(feat))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.MaxPool3d(kernel_size=2, stride=2))
            if dropout > 0:
                layers.append(nn.Dropout3d(dropout))
            current_channels = feat
        self.feature_extractor = nn.Sequential(*layers)
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.classifier = nn.Linear(current_channels, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        x = self.feature_extractor(x)
        x = self.avg_pool(x)
        x = torch.flatten(x, 1)
        logits = self.classifier(x)
        return logits.squeeze(-1)

    def _compute_loss(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return F.binary_cross_entropy_with_logits(logits, targets)

    def training_step(self, batch, batch_idx: int):  # type: ignore[override]
        images, targets = batch
        logits = self(images)
        loss = self._compute_loss(logits, targets)
        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx: int):  # type: ignore[override]
        images, targets = batch
        logits = self(images)
        loss = self._compute_loss(logits, targets)
        self.log("val_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def configure_optimizers(self):
        optimizer_name = str(self.training_config.get("optimizer", "adam"))
        lr = float(self.training_config.get("learning_rate", 1e-3))
        weight_decay = float(self.training_config.get("weight_decay", 0.0))

        if optimizer_name.lower() == "adamw":
            optimizer = torch.optim.AdamW(self.parameters(), lr=lr, weight_decay=weight_decay)
        else:
            optimizer = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)
        return optimizer


def create_model(model_config: Dict[str, Any], training_config: Dict[str, Any]) -> pl.LightningModule:
    """Factory for creating models based on configuration."""

    name = model_config.get("name", "simple_3d_cnn")
    if name == "simple_3d_cnn":
        return Simple3DCNN(model_config, training_config)
    raise ValueError(f"Unknown model name: {name}")


__all__ = ["Simple3DCNN", "create_model"]
