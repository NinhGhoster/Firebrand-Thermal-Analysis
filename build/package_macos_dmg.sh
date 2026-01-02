#!/usr/bin/env bash
set -euo pipefail

# Package a macOS DMG from the PyInstaller output.
# Requires hdiutil (built-in on macOS).

APP_NAME="FirebrandThermalAnalysis"
DIST_DIR="dist/${APP_NAME}"
APP_BUNDLE="dist/${APP_NAME}.app"
OUT_DMG="dist/${APP_NAME}.dmg"
STAGING="build/_dmg"

rm -rf "$STAGING"
mkdir -p "$STAGING"

if [[ -d "$APP_BUNDLE" ]]; then
  cp -R "$APP_BUNDLE" "$STAGING/${APP_NAME}.app"
elif [[ -d "$DIST_DIR" ]]; then
  cp -R "$DIST_DIR" "$STAGING/${APP_NAME}"
else
  echo "Expected ${APP_BUNDLE} or ${DIST_DIR}. Run build/build_macos.sh first." >&2
  exit 1
fi

ln -s /Applications "$STAGING/Applications"
hdiutil create -volname "$APP_NAME" -srcfolder "$STAGING" -ov -format UDZO -imagekey zlib-level=9 "$OUT_DMG"
echo "DMG created at: $OUT_DMG"
