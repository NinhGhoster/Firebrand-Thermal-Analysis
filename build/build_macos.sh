#!/usr/bin/env bash
set -euo pipefail

# Build on macOS only. Requires FLIR SDK and PyInstaller installed.
# Optional env vars:
# - PYTHON_BIN: python executable (default: python3)
# - FLIR_SDK_LIB_DIR: directory containing FLIR SDK dylibs
# - FLIR_SDK_BIN_DIR: directory containing FLIR SDK binaries
# - FLIR_SDK_WHEEL: path to a prebuilt FileSDK wheel (preferred)
# - FLIR_SDK_PYTHON_DIR: FLIR SDK python folder containing setup.py
# - FLIR_SDK_SHADOW_DIR: shadow dir for wheel build (default: /tmp/flir_sdk_build)

APP_NAME="FirebrandThermalAnalysis"
ENTRY="FirebrandThermalAnalysis.py"
PYTHON_BIN="${PYTHON_BIN:-python3}"

if [[ -n "${FLIR_SDK_WHEEL:-}" ]]; then
  "$PYTHON_BIN" -m pip install "$FLIR_SDK_WHEEL"
elif [[ -n "${FLIR_SDK_PYTHON_DIR:-}" ]]; then
  SHADOW_DIR="${FLIR_SDK_SHADOW_DIR:-/tmp/flir_sdk_build}"
  mkdir -p "$SHADOW_DIR"
  "$PYTHON_BIN" "${FLIR_SDK_PYTHON_DIR}/setup.py" bdist_wheel --shadow-dir "$SHADOW_DIR"
  WHEEL_PATH="$(ls "$SHADOW_DIR"/dist/*.whl | head -n 1)"
  if [[ -n "$WHEEL_PATH" ]]; then
    "$PYTHON_BIN" -m pip install "$WHEEL_PATH"
  else
    echo "No wheel found in ${SHADOW_DIR}/dist" >&2
    exit 1
  fi
fi

opts=(--windowed --onedir --noconfirm --strip --name "$APP_NAME" --collect-all fnv "$ENTRY")

if [[ -n "${FLIR_SDK_LIB_DIR:-}" ]]; then
  opts+=(--add-binary "${FLIR_SDK_LIB_DIR}/*:./")
fi
if [[ -n "${FLIR_SDK_BIN_DIR:-}" ]]; then
  opts+=(--add-binary "${FLIR_SDK_BIN_DIR}/*:./")
fi

"$PYTHON_BIN" -m PyInstaller "${opts[@]}"
echo "Build output: dist/${APP_NAME}"
