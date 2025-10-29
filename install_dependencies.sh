#!/usr/bin/env bash
set -euo pipefail

REQUIREMENTS_FILE="${REQUIREMENTS_FILE:-requirements.txt}"
DOWNLOAD_DIR="${DOWNLOAD_DIR:-local_packages}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

if [[ ! -f "$REQUIREMENTS_FILE" ]]; then
  echo "Requirements file not found: $REQUIREMENTS_FILE" >&2
  exit 1
fi

mkdir -p "$DOWNLOAD_DIR"

if [[ -z "$(find "$DOWNLOAD_DIR" -mindepth 1 -maxdepth 1 -type f 2>/dev/null)" ]]; then
  echo "Downloading dependencies listed in $REQUIREMENTS_FILE to $DOWNLOAD_DIR"
  "$PYTHON_BIN" -m pip download --dest "$DOWNLOAD_DIR" -r "$REQUIREMENTS_FILE"
else
  echo "Using existing downloaded packages in $DOWNLOAD_DIR"
fi

echo "Installing dependencies from $DOWNLOAD_DIR"
"$PYTHON_BIN" -m pip install --no-index --find-links "$DOWNLOAD_DIR" -r "$REQUIREMENTS_FILE"
