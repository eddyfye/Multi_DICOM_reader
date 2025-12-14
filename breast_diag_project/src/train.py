"""Training runner using PyTorch Lightning."""
from __future__ import annotations

import argparse
from pathlib import Path

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

from .config import ExperimentConfig, load_config
from .dataset import BreastDiagnosisDataModule
from .models import create_model


def run_training(config: ExperimentConfig) -> None:
    data_module = BreastDiagnosisDataModule(config)
    model = create_model(config.model, config.training)

    save_dir = Path(config.output.get("save_dir", "outputs"))
    save_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_cb = ModelCheckpoint(
        dirpath=save_dir,
        monitor=config.output.get("checkpoint_monitor", "val_loss"),
        save_top_k=1,
        mode="min",
    )

    trainer = pl.Trainer(
        accelerator=config.trainer.get("accelerator", "auto"),
        devices=config.trainer.get("devices", "auto"),
        max_epochs=int(config.training.get("max_epochs", 1)),
        precision=config.training.get("precision", "32-true"),
        gradient_clip_val=float(config.training.get("gradient_clip_val", 0.0)),
        log_every_n_steps=int(config.trainer.get("log_every_n_steps", 50)),
        callbacks=[checkpoint_cb],
        default_root_dir=save_dir,
    )

    trainer.fit(model, datamodule=data_module)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train breast diagnosis model")
    parser.add_argument("--config", required=True, help="Path to experiment config JSON")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    config = load_config(args.config)
    run_training(config)


if __name__ == "__main__":
    main()
