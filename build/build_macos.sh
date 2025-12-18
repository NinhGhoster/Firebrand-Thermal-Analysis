#!/usr/bin/env bash
set -euo pipefail

# Build on macOS only. Requires FLIR SDK and PyInstaller installed.
# Optional env vars:
# - PYTHON_BIN: python executable (default: python3)
# - FLIR_SDK_LIB_DIR: directory containing FLIR SDK dylibs
# - FLIR_SDK_BIN_DIR: directory containing FLIR SDK binaries

APP_NAME="FirebrandThermalAnalysis"
ENTRY="SDK_dashboard.py"
PYTHON_BIN="${PYTHON_BIN:-python3}"

opts=(--windowed --onedir --name "$APP_NAME" "$ENTRY")

if [[ -n "${FLIR_SDK_LIB_DIR:-}" ]]; then
  opts+=(--add-binary "${FLIR_SDK_LIB_DIR}/*:./")
fi
if [[ -n "${FLIR_SDK_BIN_DIR:-}" ]]; then
  opts+=(--add-binary "${FLIR_SDK_BIN_DIR}/*:./")
fi

"$PYTHON_BIN" -m PyInstaller "${opts[@]}"
echo "Build output: dist/${APP_NAME}"
