"""Master entrypoint to build manifest and launch training."""
from __future__ import annotations

import argparse
from pathlib import Path

from breast_diag_project.src import build_manifest, train
from breast_diag_project.src.config import load_config


def run_experiment(
    config_path: str,
    input_dir: str,
    preproc_result_dir: str,
    output_dir: str,
    build_manifest_flag: bool = False,
) -> None:
    config = load_config(config_path, input_dir, preproc_result_dir, output_dir)

    images_root = config.raw_images_dir
    sr_root = config.raw_sr_dir
    interim_dir = config.interim_dir
    manifest_path = config.manifest_path
    output_root = config.output_dir

    interim_dir.mkdir(parents=True, exist_ok=True)
    output_root.mkdir(parents=True, exist_ok=True)

    if build_manifest_flag or not manifest_path.exists():
        # Rebuild the manifest when requested or missing to keep data aligned.
        build_manifest.run(
            str(images_root),
            str(sr_root),
            str(manifest_path),
            image_modality=config.image_modality,
            series_description_keyword=config.series_description_keyword,
            sr_modality=config.sr_modality,
        )

    train.run_training(config)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run full experiment")
    parser.add_argument("--config", required=True, help="Path to experiment config JSON")
    parser.add_argument(
        "--inputdir",
        required=True,
        help="Root directory containing raw image and SR data (expects 'images' and 'sr' subdirs)",
    )
    parser.add_argument(
        "--preprocresultdir",
        required=True,
        help="Directory to store intermediate preprocessing results and manifest",
    )
    parser.add_argument(
        "--outputdir",
        required=True,
        help="Directory to store training outputs (checkpoints, logs, metrics)",
    )
    parser.add_argument("--build-manifest", action="store_true", help="Force rebuilding manifest")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    run_experiment(
        args.config,
        input_dir=args.inputdir,
        preproc_result_dir=args.preprocresultdir,
        output_dir=args.outputdir,
        build_manifest_flag=args.build_manifest,
    )


if __name__ == "__main__":
    main()
