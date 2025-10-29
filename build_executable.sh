#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python3}"
SCRIPT_PATH="${1:-dicom_metadata_extractor.py}"
APP_NAME="${APP_NAME:-dicom_metadata_extractor}"
SPEC_DIR="${SPEC_DIR:-pyinstaller}"
DIST_DIR="${DIST_DIR:-dist}"
BUILD_DIR="${BUILD_DIR:-build}"

if [[ ! -f "$SCRIPT_PATH" ]]; then
  echo "Target script not found: $SCRIPT_PATH" >&2
  exit 1
fi

mkdir -p "$SPEC_DIR" "$DIST_DIR" "$BUILD_DIR"

"$PYTHON_BIN" -m PyInstaller \
  --name "$APP_NAME" \
  --onefile "$SCRIPT_PATH" \
  --distpath "$DIST_DIR" \
  --workpath "$BUILD_DIR" \
  --specpath "$SPEC_DIR" \
  --noconfirm

echo "Executable created at $DIST_DIR/$APP_NAME"
echo "Spec file generated at $SPEC_DIR/${APP_NAME}.spec"
