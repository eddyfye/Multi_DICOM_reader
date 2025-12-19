"""Configuration loading and validation utilities."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

REQUIRED_TOP_LEVEL_KEYS = {"paths", "data", "model", "training", "trainer", "output"}


@dataclass
class ExperimentConfig:
    """Container for experiment configuration."""

    raw: Dict[str, Any]
    input_dir: Path
    preproc_result_dir: Path
    output_dir_path: Path
    project_name: str

    @property
    def paths(self) -> Dict[str, Any]:
        return self.raw["paths"]

    @property
    def data(self) -> Dict[str, Any]:
        return self.raw["data"]

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
    def project(self) -> str:
        return self.project_name

    @property
    def manifest_path(self) -> Path:
        return self.interim_dir / self.manifest_filename

    @property
    def raw_images_dir(self) -> Path:
        images_subdir = self.paths.get("raw_images_subdir", "images")
        return self.input_dir / images_subdir

    @property
    def raw_sr_dir(self) -> Path:
        sr_subdir = self.paths.get("raw_sr_subdir", "sr")
        return self.input_dir / sr_subdir

    @property
    def interim_dir(self) -> Path:
        return self.preproc_result_dir

    @property
    def output_dir(self) -> Path:
        return self.output_dir_path

    @property
    def manifest_filename(self) -> str:
        return str(self.paths["manifest_filename"])

    @property
    def dicom_filters(self) -> Dict[str, Any]:
        return self.data.get("dicom_filters", {})

    @property
    def image_modality(self) -> str:
        return str(self.dicom_filters.get("image_modality", "MR"))

    @property
    def series_description_keyword(self) -> str:
        return str(self.dicom_filters.get("series_description_keyword", "BLISS_AUTO"))

    @property
    def sr_modality(self) -> str:
        return str(self.dicom_filters.get("sr_modality", "SR"))

    @property
    def sr_labels_keyword(self) -> str:
        return str(self.dicom_filters.get("sr_labels_keyword", "assessment category"))


def _validate_required_keys(config: Dict[str, Any]) -> None:
    missing = REQUIRED_TOP_LEVEL_KEYS - config.keys()
    if missing:
        raise KeyError(f"Config missing required top-level keys: {sorted(missing)}")

    if not isinstance(config.get("paths"), dict):
        raise TypeError("Config 'paths' section must be a dictionary")

    if "manifest_filename" not in config["paths"]:
        raise KeyError("Config paths must include 'manifest_filename'")

    if not isinstance(config.get("data"), dict):
        raise KeyError("Config data section must include 'data' dictionary")

    if not isinstance(config.get("output"), dict):
        raise TypeError("Config 'output' section must be a dictionary")


def load_config(
    path: str, input_dir: str, preproc_result_dir: str, output_dir: str | None
) -> ExperimentConfig:
    """Load and validate an experiment configuration JSON file."""

    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with config_path.open("r") as f:
        config = json.load(f)

    _validate_required_keys(config)

    output_config = config.setdefault("output", {})
    project_name = str(output_config.get("project_name", "breast_diag_experiment"))
    output_config["project_name"] = project_name
    output_config.setdefault("experiment_name", project_name)

    base_output_dir = Path(output_dir) if output_dir else Path("outputs")
    resolved_output_dir = base_output_dir / project_name
    return ExperimentConfig(
        raw=config,
        input_dir=Path(input_dir),
        preproc_result_dir=Path(preproc_result_dir),
        output_dir_path=resolved_output_dir,
        project_name=project_name,
    )


__all__ = ["ExperimentConfig", "load_config"]
