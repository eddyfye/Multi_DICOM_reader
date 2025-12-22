"""Inference entrypoint for breast diagnosis models."""
from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any, Dict, Sequence

import numpy as np
import torch
import pydicom

from breast_diag_project.src.config import load_config, ExperimentConfig
from breast_diag_project.src.dataset import preprocess_volume
from breast_diag_project.src.dicom_index import collect_image_metadata
from breast_diag_project.src.dicom_io import load_series_pixel_array
from breast_diag_project.src.models import create_model
from breast_diag_project.src.sr_writer import map_logits_to_label, write_sr

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")


def _get_preprocess_settings(config: ExperimentConfig) -> Dict[str, Any]:
    preprocess_cfg = config.data.get("preprocessing", {})
    return {
        "clip_percentiles": preprocess_cfg.get("clip_percentiles", (1.0, 99.0)),
        "resample_isotropic": bool(preprocess_cfg.get("resample_to_isotropic", True)),
        "isotropic_method": preprocess_cfg.get("isotropic_method", "median"),
        "target_spacing_override": preprocess_cfg.get("target_spacing_override"),
        "resize_shape": preprocess_cfg.get("resize_shape"),
        "max_volume_bytes": preprocess_cfg.get("max_volume_bytes"),
        "max_voxels": preprocess_cfg.get("max_voxels"),
        "downsample_on_overflow": preprocess_cfg.get("downsample_on_overflow", True),
    }


def _load_checkpoint(model: torch.nn.Module, checkpoint_path: str, device: torch.device) -> None:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint.get("state_dict", checkpoint)
    model.load_state_dict(state_dict)


def _prepare_input_tensor(
    image_paths: Sequence[str], preprocess_settings: Dict[str, Any]
) -> torch.Tensor:
    volume, spacing = load_series_pixel_array(list(image_paths), return_spacing=True)
    tensor = preprocess_volume(
        volume,
        spacing,
        preprocess_settings["clip_percentiles"],
        preprocess_settings["resample_isotropic"],
        preprocess_settings["isotropic_method"],
        preprocess_settings["target_spacing_override"],
        preprocess_settings["resize_shape"],
        max_volume_bytes=preprocess_settings["max_volume_bytes"],
        max_voxels=preprocess_settings["max_voxels"],
        downsample_on_overflow=preprocess_settings["downsample_on_overflow"],
    )
    return tensor.unsqueeze(0)


def run_inference(
    config_path: str,
    input_dir: str,
    output_dir: str,
    checkpoint_path: str,
    image_modality: str | None = None,
    series_description_keyword: str | None = None,
) -> None:
    config = load_config(config_path, input_dir, output_dir, output_dir)

    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    modality = image_modality if image_modality is not None else config.image_modality
    series_keyword = (
        series_description_keyword
        if series_description_keyword is not None
        else config.series_description_keyword
    )

    logger.info("Scanning DICOM series in %s", input_dir)
    images_map = collect_image_metadata(
        input_dir,
        image_modality=modality or None,
        series_description_keyword=series_keyword or None,
    )
    logger.info("Found %d image series to run inference on", len(images_map))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = create_model(config.model, config.training).to(device)
    _load_checkpoint(model, checkpoint_path, device)
    model.eval()

    preprocess_settings = _get_preprocess_settings(config)

    for (study_uid, series_uid), meta_list in images_map.items():
        image_paths = [m[3] for m in sorted(meta_list, key=lambda x: x[3])]
        patient_ids = {m[0] for m in meta_list if m[0]}
        patient_id = next(iter(patient_ids), "unknown")
        if not image_paths:
            logger.warning("Skipping series %s with no image paths", series_uid)
            continue

        try:
            input_tensor = _prepare_input_tensor(image_paths, preprocess_settings).to(device)
        except ValueError as exc:
            logger.warning("Skipping series %s due to preprocessing error: %s", series_uid, exc)
            continue

        with torch.no_grad():
            logits = model(input_tensor)
        logits_np = logits.detach().cpu().numpy()
        label_fields = map_logits_to_label(np.ravel(logits_np), config)
        label_fields.update({"study_uid": study_uid, "series_uid": series_uid})

        source = pydicom.dcmread(image_paths[0], stop_before_pixels=True, force=True)
        series_output_dir = output_root / patient_id / study_uid
        sr_path = write_sr(series_output_dir, source, label_fields)
        logger.info(
            "Wrote SR for patient %s study %s series %s -> %s",
            patient_id,
            study_uid,
            series_uid,
            sr_path,
        )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run inference for breast diagnosis models")
    parser.add_argument("--config", required=True, help="Path to experiment config JSON")
    parser.add_argument("--input-dir", required=True, help="Directory with DICOM images")
    parser.add_argument("--output-dir", required=True, help="Directory for SR outputs")
    parser.add_argument("--checkpoint", required=True, help="Path to Lightning checkpoint")
    parser.add_argument(
        "--image-modality",
        default=None,
        help="Override DICOM modality filter (e.g., MR). Use '' to disable.",
    )
    parser.add_argument(
        "--series-description",
        default=None,
        help="Override series description keyword filter. Use '' to disable.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    run_inference(
        args.config,
        args.input_dir,
        args.output_dir,
        args.checkpoint,
        image_modality=args.image_modality,
        series_description_keyword=args.series_description,
    )


if __name__ == "__main__":
    main()
