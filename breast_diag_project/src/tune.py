"""Optuna-based hyperparameter tuning for breast diagnosis models."""
from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import optuna
import pytorch_lightning as pl
from optuna.trial import Trial
from pytorch_lightning.callbacks import ModelCheckpoint

from breast_diag_project.src import build_manifest, train
from breast_diag_project.src.config import ExperimentConfig, load_config
from breast_diag_project.src.dataset import BreastDiagnosisDataModule
from breast_diag_project.src.models import create_model


logger = logging.getLogger(__name__)


def _prepare_manifest(config: ExperimentConfig, build_manifest_flag: bool) -> None:
    images_root = config.raw_images_dir
    sr_root = config.raw_sr_dir
    interim_dir = config.interim_dir
    manifest_path = config.manifest_path
    output_root = config.output_dir

    interim_dir.mkdir(parents=True, exist_ok=True)
    output_root.mkdir(parents=True, exist_ok=True)

    if build_manifest_flag or not manifest_path.exists():
        build_manifest.run(
            str(images_root),
            str(sr_root),
            str(manifest_path),
            image_modality=config.image_modality,
            series_description_keyword=config.series_description_keyword,
            sr_modality=config.sr_modality,
        )


def _suggest_from_space(
    trial: Trial,
    name: str,
    spec: Any,
    default_fn: Callable[[], Any],
) -> Any:
    if spec is None:
        return default_fn()
    if isinstance(spec, Mapping):
        if "value" in spec:
            return spec["value"]
        param_type = str(spec.get("type", "float")).lower()
        if param_type == "categorical":
            choices = spec.get("choices")
            if not isinstance(choices, Sequence):
                raise ValueError(f"Categorical search space for '{name}' must include choices.")
            return trial.suggest_categorical(name, list(choices))
        if param_type == "int":
            return trial.suggest_int(
                name,
                int(spec["low"]),
                int(spec["high"]),
                step=int(spec.get("step", 1)),
                log=bool(spec.get("log", False)),
            )
        return trial.suggest_float(
            name,
            float(spec["low"]),
            float(spec["high"]),
            step=spec.get("step"),
            log=bool(spec.get("log", False)),
        )
    if isinstance(spec, Sequence) and not isinstance(spec, (str, bytes)):
        return trial.suggest_categorical(name, list(spec))
    return spec


def _apply_trial_config(
    config: ExperimentConfig, trial: Trial, search_space: Mapping[str, Any] | None
) -> None:
    model_config = config.model
    training_config = config.training
    search_space = search_space or {}

    training_config["learning_rate"] = _suggest_from_space(
        trial,
        "learning_rate",
        search_space.get("learning_rate"),
        lambda: trial.suggest_float("learning_rate", 1e-4, 5e-3, log=True),
    )
    training_config["weight_decay"] = _suggest_from_space(
        trial,
        "weight_decay",
        search_space.get("weight_decay"),
        lambda: trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True),
    )
    training_config["batch_size"] = _suggest_from_space(
        trial,
        "batch_size",
        search_space.get("batch_size"),
        lambda: trial.suggest_categorical("batch_size", [1, 2, 4]),
    )
    training_config["optimizer"] = _suggest_from_space(
        trial,
        "optimizer",
        search_space.get("optimizer"),
        lambda: trial.suggest_categorical("optimizer", ["adam", "adamw"]),
    )

    model_config["dropout"] = _suggest_from_space(
        trial,
        "dropout",
        search_space.get("dropout"),
        lambda: trial.suggest_float("dropout", 0.0, 0.5),
    )
    model_name = str(model_config.get("name", "simple_3d_cnn")).lower()
    if model_name == "simple_3d_cnn":
        model_config["features"] = _suggest_from_space(
            trial,
            "features",
            search_space.get("features"),
            lambda: trial.suggest_categorical(
                "features",
                [[16, 32, 64], [32, 64, 128], [32, 64, 128, 256]],
            ),
        )
    elif model_name == "resnet3d":
        model_config["base_channels"] = _suggest_from_space(
            trial,
            "base_channels",
            search_space.get("base_channels"),
            lambda: trial.suggest_categorical("base_channels", [32, 64]),
        )
        model_config["layers"] = _suggest_from_space(
            trial,
            "layers",
            search_space.get("layers"),
            lambda: trial.suggest_categorical("layers", [[2, 2, 2, 2], [3, 4, 6, 3]]),
        )


def _configure_trial_output(config: ExperimentConfig, trial: Trial) -> Path:
    trial_dir = config.output_dir / f"trial_{trial.number}"
    trial_dir.mkdir(parents=True, exist_ok=True)
    config.output_dir_path = trial_dir
    config.output["experiment_name"] = f"{config.project}_trial_{trial.number}"
    return trial_dir


def _build_trainer(
    config: ExperimentConfig,
    trial: Trial,
    monitor_metric: str,
    trial_dir: Path,
    enable_pruning: bool,
) -> pl.Trainer:
    callbacks = []
    checkpoint_cb = ModelCheckpoint(
        dirpath=trial_dir,
        monitor=monitor_metric,
        save_top_k=1,
        mode="min",
    )
    callbacks.append(checkpoint_cb)

    if enable_pruning:
        try:
            from optuna.integration import PyTorchLightningPruningCallback
        except ImportError:  # pragma: no cover - optional dependency
            logger.warning("Optuna pruning callback unavailable; proceeding without pruning.")
        else:
            callbacks.append(PyTorchLightningPruningCallback(trial, monitor=monitor_metric))

    train._configure_matmul_precision(config.training)
    precision = train._resolve_precision(config.training)

    return pl.Trainer(
        accelerator=config.trainer.get("accelerator", "auto"),
        devices=config.trainer.get("devices", "auto"),
        max_epochs=int(config.training.get("max_epochs", 1)),
        precision=precision,
        gradient_clip_val=float(config.training.get("gradient_clip_val", 0.0)),
        log_every_n_steps=int(config.trainer.get("log_every_n_steps", 50)),
        callbacks=callbacks,
        default_root_dir=trial_dir,
        logger=train._create_logger(config.output, trial_dir),
    )


def _objective(
    trial: Trial,
    config_path: Path,
    input_dir: str,
    preproc_result_dir: str,
    output_dir: str,
    enable_pruning: bool,
    search_space: Mapping[str, Any] | None,
) -> float:
    config = load_config(str(config_path), input_dir, preproc_result_dir, output_dir)
    _apply_trial_config(config, trial, search_space)

    trial_dir = _configure_trial_output(config, trial)
    data_module = BreastDiagnosisDataModule(config)
    model = create_model(config.model, config.training)
    monitor_metric = str(config.output.get("checkpoint_monitor", "val_loss"))

    trainer = _build_trainer(config, trial, monitor_metric, trial_dir, enable_pruning)
    trainer.fit(model, datamodule=data_module)

    metric_value: Any = trainer.callback_metrics.get(monitor_metric)
    if metric_value is None:
        raise ValueError(f"Monitored metric '{monitor_metric}' not found in trainer metrics.")
    if hasattr(metric_value, "item"):
        return float(metric_value.item())
    return float(metric_value)


def run_tuning(
    config_dir: str,
    input_dir: str,
    preproc_result_dir: str,
    output_dir: str,
    build_manifest_flag: bool,
) -> optuna.Study:
    config_path = Path(config_dir) / "config.json"
    base_config = load_config(str(config_path), input_dir, preproc_result_dir, output_dir)
    _prepare_manifest(base_config, build_manifest_flag)

    tuning_config = base_config.raw.get("tuning", {})
    n_trials = int(tuning_config.get("n_trials", 10))
    study_name = str(tuning_config.get("study_name", "breast_diag_study"))
    storage = tuning_config.get("storage")
    direction = str(tuning_config.get("direction", "minimize"))
    enable_pruning = bool(tuning_config.get("prune", False))
    search_space = tuning_config.get("search_space", {})

    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        direction=direction,
        load_if_exists=True,
    )
    study.optimize(
        lambda trial: _objective(
            trial,
            config_path=config_path,
            input_dir=input_dir,
            preproc_result_dir=preproc_result_dir,
            output_dir=output_dir,
            enable_pruning=enable_pruning,
            search_space=search_space,
        ),
        n_trials=n_trials,
    )
    logger.info("Best trial %s -> value %.4f", study.best_trial.number, study.best_value)
    logger.info("Best params: %s", study.best_trial.params)
    return study


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tune breast diagnosis model with Optuna")
    parser.add_argument(
        "--configdir",
        required=True,
        help="Directory containing the experiment configuration JSON (config.json)",
    )
    parser.add_argument(
        "--inputdir",
        required=True,
        help=(
            "Root directory containing raw DICOM data; image and SR series are inferred "
            "from metadata rather than fixed subdirectories"
        ),
    )
    parser.add_argument(
        "--preprocresultdir",
        required=True,
        help="Directory to store intermediate preprocessing results and manifest",
    )
    parser.add_argument(
        "--outputdir",
        required=True,
        help="Directory to store tuning outputs (checkpoints, logs, metrics)",
    )
    parser.add_argument("--build-manifest", action="store_true", help="Force rebuilding manifest")
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    args = _parse_args()
    run_tuning(
        args.configdir,
        input_dir=args.inputdir,
        preproc_result_dir=args.preprocresultdir,
        output_dir=args.outputdir,
        build_manifest_flag=args.build_manifest,
    )


if __name__ == "__main__":
    main()
