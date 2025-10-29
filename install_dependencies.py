#!/usr/bin/env python3
"""Download and install project dependencies in a cross-platform manner."""

from __future__ import annotations

import argparse
import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import List

from script_utils import detect_python_command


def download_dependencies(python_cmd: List[str], requirements_file: Path, download_dir: Path) -> None:
    download_dir.mkdir(parents=True, exist_ok=True)

    if any(download_dir.iterdir()):
        print(f"Using existing downloaded packages in {download_dir}")
        return

    print(f"Downloading dependencies listed in {requirements_file} to {download_dir}")
    subprocess.run(
        python_cmd
        + [
            "-m",
            "pip",
            "download",
            "--dest",
            str(download_dir),
            "-r",
            str(requirements_file),
        ],
        check=True,
    )


def install_dependencies(python_cmd: List[str], requirements_file: Path, download_dir: Path) -> None:
    print(f"Installing dependencies from {download_dir}")
    subprocess.run(
        python_cmd
        + [
            "-m",
            "pip",
            "install",
            "--no-index",
            "--find-links",
            str(download_dir),
            "-r",
            str(requirements_file),
        ],
        check=True,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--requirements-file",
        default=os.environ.get("REQUIREMENTS_FILE", "requirements.txt"),
        type=Path,
        help="Path to the requirements file",
    )
    parser.add_argument(
        "--download-dir",
        default=Path(os.environ.get("DOWNLOAD_DIR", "local_packages")),
        type=Path,
        help="Directory where wheels/source packages will be stored",
    )

    args = parser.parse_args(argv)

    requirements_file = args.requirements_file.resolve()
    download_dir = args.download_dir.resolve()

    if not requirements_file.is_file():
        parser.error(f"Requirements file not found: {requirements_file}")

    python_cmd = detect_python_command()
    print(f"Detected operating system: {platform.system().lower()}")
    print(f"Using Python command: {' '.join(python_cmd)}")

    download_dependencies(python_cmd, requirements_file, download_dir)
    install_dependencies(python_cmd, requirements_file, download_dir)

    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("Operation cancelled by user", file=sys.stderr)
        sys.exit(1)
