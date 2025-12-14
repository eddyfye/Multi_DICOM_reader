"""Configuration loading and validation utilities."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

REQUIRED_TOP_LEVEL_KEYS = {"paths", "data", "labels", "model", "training", "trainer", "output"}


@dataclass
class ExperimentConfig:
    """Container for experiment configuration."""

    raw: Dict[str, Any]

    @property
    def paths(self) -> Dict[str, Any]:
        return self.raw["paths"]

    @property
    def data(self) -> Dict[str, Any]:
        return self.raw["data"]

    @property
    def labels(self) -> Dict[str, Any]:
        return self.raw["labels"]

    @property
    def model(self) -> Dict[str, Any]:
        return self.raw["model"]

    @property
    def training(self) -> Dict[str, Any]:
        return self.raw["training"]

    @property
    def trainer(self) -> Dict[str, Any]:
        return self.raw["trainer"]

    @property
    def output(self) -> Dict[str, Any]:
        return self.raw["output"]

    @property
    def manifest_path(self) -> Path:
        return Path(self.data["manifest_path"])

    @property
    def raw_images_dir(self) -> Path:
        return Path(self.paths["raw_images_dir"])

    @property
    def raw_sr_dir(self) -> Path:
        return Path(self.paths["raw_sr_dir"])

    @property
    def processed_dir(self) -> Path:
        return Path(self.paths["processed_dir"])

    @property
    def manifest_filename(self) -> str:
        return str(self.paths["manifest_filename"])


def _validate_required_keys(config: Dict[str, Any]) -> None:
    missing = REQUIRED_TOP_LEVEL_KEYS - config.keys()
    if missing:
        raise KeyError(f"Config missing required top-level keys: {sorted(missing)}")

    if not isinstance(config.get("paths"), dict):
        raise TypeError("Config 'paths' section must be a dictionary")

    for key in ["raw_images_dir", "raw_sr_dir", "processed_dir", "manifest_filename"]:
        if key not in config["paths"]:
            raise KeyError(f"Config paths missing '{key}'")

    if not isinstance(config.get("data"), dict) or "manifest_path" not in config["data"]:
        raise KeyError("Config data section must include 'manifest_path'")


def load_config(path: str) -> ExperimentConfig:
    """Load and validate an experiment configuration JSON file."""

    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with config_path.open("r") as f:
        config = json.load(f)

    _validate_required_keys(config)
    return ExperimentConfig(raw=config)


__all__ = ["ExperimentConfig", "load_config"]
