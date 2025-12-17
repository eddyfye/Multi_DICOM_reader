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
        return self.interim_dir / self.manifest_filename

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
    def interim_dir(self) -> Path:
        if "interim_dir" in self.paths:
            return Path(self.paths["interim_dir"])
        return self.processed_dir.parent / "interim"

    @property
    def output_dir(self) -> Path:
        if "output_dir" in self.paths:
            return Path(self.paths["output_dir"])
        return Path(self.output.get("save_dir", "outputs"))

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
        # Each path entry must exist to locate inputs/outputs on disk.
        if key not in config["paths"]:
            raise KeyError(f"Config paths missing '{key}'")

    if not isinstance(config.get("data"), dict):
        raise KeyError("Config data section must include 'data' dictionary")


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
