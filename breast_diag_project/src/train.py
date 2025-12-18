"""Training runner using PyTorch Lightning."""
from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
from typing import Any

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger, TensorBoardLogger
from pytorch_lightning.utilities.rank_zero import rank_zero_info

from breast_diag_project.src.config import ExperimentConfig, load_config
from breast_diag_project.src.dataset import BreastDiagnosisDataModule
from breast_diag_project.src.models import create_model


def _tensorboard_available() -> bool:
    """Return whether a TensorBoard-compatible package is installed."""

    return bool(importlib.util.find_spec("tensorboard") or importlib.util.find_spec("tensorboardX"))


def _create_logger(output_config: dict[str, Any], save_dir: Path):
    """Create a Lightning logger with a safe TensorBoard fallback."""

    logger_preference = str(output_config.get("logger", "auto")).lower()
    experiment_name = str(output_config.get("experiment_name", "breast_diag_experiment"))

    if logger_preference == "none":
        return False

    if logger_preference in {"tensorboard", "tb"}:
        if not _tensorboard_available():
            raise ImportError(
                "TensorBoard logging requested but no compatible package was found. "
                "Install 'tensorboard' or 'tensorboardX' or set output.logger to 'csv' or 'auto'."
            )
        return TensorBoardLogger(save_dir=str(save_dir), name=experiment_name)

    if logger_preference == "csv":
        return CSVLogger(save_dir=str(save_dir), name=experiment_name)

    # Auto mode: prefer TensorBoard when available, otherwise default to CSV.
    if _tensorboard_available():
        return TensorBoardLogger(save_dir=str(save_dir), name=experiment_name)
    return CSVLogger(save_dir=str(save_dir), name=experiment_name)


def _configure_matmul_precision(training_config: dict[str, Any]) -> None:
    """Optionally configure matmul precision for Tensor Core usage."""

    precision_setting = training_config.get("matmul_precision")
    if precision_setting:
        # Let users opt into faster matmul kernels when their hardware supports it.
        torch.set_float32_matmul_precision(str(precision_setting))


def _resolve_precision(training_config: dict[str, Any]) -> str:
    """Determine precision settings and emit a rank-zero log message."""

    precision = str(training_config.get("precision", "16-mixed"))
    training_config["precision"] = precision

    amp_enabled = precision.lower() in {"16", "16-mixed", "bf16", "bf16-mixed"}
    if amp_enabled:
        rank_zero_info(f"Training with AMP precision: {precision}")
    else:
        rank_zero_info(
            "AMP is disabled; set training.precision to '16-mixed' or 'bf16-mixed' "
            "in the config to enable mixed precision."
        )

    return precision


def run_training(config: ExperimentConfig) -> None:
    data_module = BreastDiagnosisDataModule(config)
    model = create_model(config.model, config.training)

    _configure_matmul_precision(config.training)
    precision = _resolve_precision(config.training)

    save_dir = config.output_dir
    save_dir.mkdir(parents=True, exist_ok=True)
    monitor_metric = config.output.get("checkpoint_monitor", "val_loss")
    rank_zero_info("Using checkpoint monitor metric: %s", monitor_metric)
    checkpoint_cb = ModelCheckpoint(
        dirpath=save_dir,
        monitor=monitor_metric,
        save_top_k=1,
        mode="min",
    )

    trainer = pl.Trainer(
        accelerator=config.trainer.get("accelerator", "auto"),
        devices=config.trainer.get("devices", "auto"),
        max_epochs=int(config.training.get("max_epochs", 1)),
        precision=precision,
        gradient_clip_val=float(config.training.get("gradient_clip_val", 0.0)),
        log_every_n_steps=int(config.trainer.get("log_every_n_steps", 50)),
        callbacks=[checkpoint_cb],
        default_root_dir=save_dir,
        logger=_create_logger(config.output, save_dir),
    )

    trainer.fit(model, datamodule=data_module)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train breast diagnosis model")
    parser.add_argument(
        "--configdir",
        required=True,
        help="Directory containing the experiment configuration JSON (config.json)",
    )
    parser.add_argument(
        "--preprocresultdir",
        required=True,
        help="Directory containing interim preprocessing results and manifest",
    )
    parser.add_argument(
        "--outputdir",
        required=True,
        help="Directory to store training outputs (checkpoints, logs, metrics)",
    )
    parser.add_argument(
        "--inputdir",
        required=True,
        help=(
            "Root directory containing raw DICOM data; image and SR series are inferred "
            "from metadata rather than fixed subdirectories"
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    config_path = Path(args.configdir) / "config.json"
    config = load_config(config_path, args.inputdir, args.preprocresultdir, args.outputdir)
    run_training(config)


if __name__ == "__main__":
    main()
