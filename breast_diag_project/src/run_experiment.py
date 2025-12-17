"""Master entrypoint to build manifest and launch training."""
from __future__ import annotations

import argparse
from pathlib import Path

from breast_diag_project.src import build_manifest, train
from breast_diag_project.src.config import load_config


def run_experiment(config_path: str, build_manifest_flag: bool = False) -> None:
    config = load_config(config_path)

    images_root = config.raw_images_dir
    sr_root = config.raw_sr_dir
    processed_dir = config.processed_dir
    interim_dir = config.interim_dir
    manifest_path = config.manifest_path

    processed_dir.mkdir(parents=True, exist_ok=True)
    interim_dir.mkdir(parents=True, exist_ok=True)

    if build_manifest_flag or not manifest_path.exists():
        # Rebuild the manifest when requested or missing to keep data aligned.
        build_manifest.run(str(images_root), str(sr_root), str(manifest_path))

    train.run_training(config)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run full experiment")
    parser.add_argument("--config", required=True, help="Path to experiment config JSON")
    parser.add_argument("--build-manifest", action="store_true", help="Force rebuilding manifest")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    run_experiment(args.config, build_manifest_flag=args.build_manifest)


if __name__ == "__main__":
    main()
