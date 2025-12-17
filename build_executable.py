#!/usr/bin/env python3
"""Cross-platform executable packaging script using PyInstaller."""

from __future__ import annotations

import argparse
import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import List

from script_utils import detect_python_command


def run_pyinstaller(
    python_cmd: List[str],
    script_path: Path,
    app_name: str,
    spec_dir: Path,
    dist_dir: Path,
    build_dir: Path,
) -> None:
    # Assemble the PyInstaller invocation in a list to avoid shell quoting issues.
    command = (
        python_cmd
        + [
            "-m",
            "PyInstaller",
            "--name",
            app_name,
            "--onedir",
            str(script_path),
            "--distpath",
            str(dist_dir),
            "--workpath",
            str(build_dir),
            "--specpath",
            str(spec_dir),
            "--noconfirm",
        ]
        )

    # ``check=True`` raises immediately on packaging failures.
    subprocess.run(command, check=True)



def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "script_path",
        nargs="?",
        default="dicom_metadata_extractor.py",
        help="Python entry script to package",
    )
    parser.add_argument(
        "--app-name",
        default=os.environ.get("APP_NAME", "dicom_metadata_extractor"),
        help="Output application name (default: %(default)s)",
    )
    parser.add_argument(
        "--spec-dir",
        type=Path,
        default=Path(os.environ.get("SPEC_DIR", "pyinstaller")),
        help="Directory to store PyInstaller spec files",
    )
    parser.add_argument(
        "--dist-dir",
        type=Path,
        default=Path(os.environ.get("DIST_DIR", "dist")),
        help="Directory where executables will be placed",
    )
    parser.add_argument(
        "--build-dir",
        type=Path,
        default=Path(os.environ.get("BUILD_DIR", "build")),
        help="PyInstaller build working directory",
    )

    args = parser.parse_args(argv)

    script_path = Path(args.script_path).resolve()
    if not script_path.is_file():
        parser.error(f"Target script not found: {script_path}")

    spec_dir = args.spec_dir.resolve()
    dist_dir = args.dist_dir.resolve()
    build_dir = args.build_dir.resolve()

    spec_dir.mkdir(parents=True, exist_ok=True)
    dist_dir.mkdir(parents=True, exist_ok=True)
    build_dir.mkdir(parents=True, exist_ok=True)

    python_cmd = detect_python_command()

    print(f"Detected operating system: {platform.system().lower()}")
    print(f"Using Python command: {' '.join(python_cmd)}")
    print(
        f"Packaging {script_path.name} into an executable named "
        f"{args.app_name}"
    )

    run_pyinstaller(python_cmd, script_path, args.app_name, spec_dir, dist_dir, build_dir)

    executable_path = dist_dir / args.app_name
    if platform.system() == "Windows":
        executable_path = executable_path.with_suffix(".exe")

    print(f"Executable created at {executable_path}")
    print(f"Spec file generated at {spec_dir / (args.app_name + '.spec')}")

    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("Operation cancelled by user", file=sys.stderr)
        sys.exit(1)
