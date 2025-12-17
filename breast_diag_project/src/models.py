"""Model definitions and factories."""
from __future__ import annotations

from typing import Any, Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl


class Simple3DCNN(pl.LightningModule):
    """A simple configurable 3D CNN for classification."""

    def __init__(self, model_config: Dict[str, Any], training_config: Dict[str, Any]):
        super().__init__()
        self.save_hyperparameters({"model": model_config, "training": training_config})
        self.training_config = training_config

        in_channels = int(model_config.get("in_channels", 1))
        self.num_classes = int(model_config.get("num_classes", 1))
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
                # Dropout optionally regularizes each downsampling stage.
                layers.append(nn.Dropout3d(dropout))
            current_channels = feat
        self.feature_extractor = nn.Sequential(*layers)
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.classifier = nn.Linear(current_channels, self.num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        x = self.feature_extractor(x)
        x = self.avg_pool(x)
        x = torch.flatten(x, 1)
        logits = self.classifier(x)
        if self.num_classes == 1:
            return logits.squeeze(-1)
        return logits

    def _compute_loss(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if self.num_classes == 1:
            return F.binary_cross_entropy_with_logits(logits, targets)
        return F.cross_entropy(logits, targets.long())

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


class BasicBlock3D(nn.Module):
    """3D version of the classic ResNet basic block."""

    expansion = 1

    def __init__(self, in_planes: int, planes: int, stride: int = 1, downsample: nn.Module | None = None):
        super().__init__()
        self.conv1 = nn.Conv3d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm3d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(planes, planes, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(planes)
        self.downsample = downsample

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out


class ResNet3D(pl.LightningModule):
    """A ResNet-style architecture for volumetric data."""

    def __init__(self, model_config: Dict[str, Any], training_config: Dict[str, Any]):
        super().__init__()
        self.save_hyperparameters({"model": model_config, "training": training_config})
        self.training_config = training_config

        self.in_channels = int(model_config.get("in_channels", 1))
        self.num_classes = int(model_config.get("num_classes", 1))
        self.layers_config: List[int] = list(model_config.get("layers", [2, 2, 2, 2]))
        self.base_channels = int(model_config.get("base_channels", 64))
        dropout = float(model_config.get("dropout", 0.0))

        self.inplanes = self.base_channels
        self.conv1 = nn.Conv3d(
            self.in_channels, self.base_channels, kernel_size=7, stride=2, padding=3, bias=False
        )
        self.bn1 = nn.BatchNorm3d(self.base_channels)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(self.base_channels, self.layers_config[0])
        self.layer2 = self._make_layer(self.base_channels * 2, self.layers_config[1], stride=2)
        self.layer3 = self._make_layer(self.base_channels * 4, self.layers_config[2], stride=2)
        self.layer4 = self._make_layer(self.base_channels * 8, self.layers_config[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.fc = nn.Linear(self.base_channels * 8 * BasicBlock3D.expansion, self.num_classes)

    def _make_layer(self, planes: int, blocks: int, stride: int = 1) -> nn.Sequential:
        downsample = None
        if stride != 1 or self.inplanes != planes * BasicBlock3D.expansion:
            downsample = nn.Sequential(
                nn.Conv3d(self.inplanes, planes * BasicBlock3D.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm3d(planes * BasicBlock3D.expansion),
            )

        layers = [BasicBlock3D(self.inplanes, planes, stride=stride, downsample=downsample)]
        self.inplanes = planes * BasicBlock3D.expansion
        for _ in range(1, blocks):
            layers.append(BasicBlock3D(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        logits = self.fc(x)
        if self.num_classes == 1:
            return logits.squeeze(-1)
        return logits

    def _compute_loss(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if self.num_classes == 1:
            return F.binary_cross_entropy_with_logits(logits, targets)
        return F.cross_entropy(logits, targets.long())

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
    if name == "resnet3d":
        return ResNet3D(model_config, training_config)
    raise ValueError(f"Unknown model name: {name}")


__all__ = ["BasicBlock3D", "Simple3DCNN", "ResNet3D", "create_model"]
