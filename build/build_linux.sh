#!/usr/bin/env bash
set -euo pipefail

# Build on Linux only. Uses uv to provision the build env.
# The project's pyproject.toml pins the per-OS FLIR SDK wheel via [tool.uv.sources],
# so `uv sync` handles all dependencies (including the dev group: pyinstaller).
# Optional env vars (custom SDK overrides):
# - FLIR_SDK_WHEEL: path to a prebuilt FileSDK wheel (preferred)
# - FLIR_SDK_PYTHON_DIR: FLIR SDK python folder containing setup.py
# - FLIR_SDK_SHADOW_DIR: shadow dir for wheel build (default: /tmp/flir_sdk_build)
# - FLIR_SDK_LIB_DIR: directory containing FLIR SDK shared libs
# - FLIR_SDK_BIN_DIR: directory containing FLIR SDK binaries

APP_NAME="FirebrandThermalAnalysis"
ENTRY="FirebrandThermalAnalysis.py"
ICON_PATH="docs/branding/logo-square.png"
LOGO_DATA="docs/branding/logo-square.png:docs/branding"

# Write VERSION file from git tag (CI sets APP_VERSION env var, else use git)
VERSION="${APP_VERSION:-$(git describe --tags --abbrev=0 2>/dev/null || echo dev)}"
printf '%s' "$VERSION" > VERSION
echo "Building version: $VERSION"

uv sync --group dev

if [[ -n "${FLIR_SDK_WHEEL:-}" ]]; then
  uv pip install --reinstall-package filesdk "$FLIR_SDK_WHEEL"
elif [[ -n "${FLIR_SDK_PYTHON_DIR:-}" ]]; then
  SHADOW_DIR="${FLIR_SDK_SHADOW_DIR:-/tmp/flir_sdk_build}"
  mkdir -p "$SHADOW_DIR"
  uv run python "${FLIR_SDK_PYTHON_DIR}/setup.py" bdist_wheel --shadow-dir "$SHADOW_DIR"
  WHEEL_PATH="$(ls "$SHADOW_DIR"/dist/*.whl | head -n 1)"
  if [[ -n "$WHEEL_PATH" ]]; then
    uv pip install --reinstall-package filesdk "$WHEEL_PATH"
  else
    echo "No wheel found in ${SHADOW_DIR}/dist" >&2
    exit 1
  fi
fi

opts=(
  --windowed
  --onedir
  --noconfirm
  --strip
  --name "$APP_NAME"
  --icon "$ICON_PATH"
  --add-data "$LOGO_DATA"
  --add-data "VERSION:."
  --paths libs
  --collect-all fnv
  --collect-all tkinterdnd2
  "$ENTRY"
)

if [[ -n "${FLIR_SDK_LIB_DIR:-}" ]]; then
  opts+=(--add-binary "${FLIR_SDK_LIB_DIR}/*:./")
fi
if [[ -n "${FLIR_SDK_BIN_DIR:-}" ]]; then
  opts+=(--add-binary "${FLIR_SDK_BIN_DIR}/*:./")
fi

uv run --group dev python -m PyInstaller "${opts[@]}"
echo "Build output: dist/${APP_NAME}"