#!/usr/bin/env bash
set -euo pipefail

# Package a macOS DMG from the PyInstaller output.
# Requires hdiutil (built-in on macOS).

APP_NAME_INTERNAL="FirebrandThermalAnalysis"
APP_NAME_DISPLAY="Firebrand Thermal Analysis"
DIST_DIR="dist/${APP_NAME_INTERNAL}"
APP_BUNDLE="dist/${APP_NAME_INTERNAL}.app"
OUT_DMG="dist/${APP_NAME_INTERNAL}.dmg"
STAGING="build/_dmg"

rm -rf "$STAGING"
mkdir -p "$STAGING"

if [[ -d "$APP_BUNDLE" ]]; then
  cp -R "$APP_BUNDLE" "$STAGING/${APP_NAME_DISPLAY}.app"
elif [[ -d "$DIST_DIR" ]]; then
  cp -R "$DIST_DIR" "$STAGING/${APP_NAME_DISPLAY}"
else
  echo "Expected ${APP_BUNDLE} or ${DIST_DIR}. Run build/build_macos.sh first." >&2
  exit 1
fi

ln -s /Applications "$STAGING/Applications"

# GitHub Actions macOS runners frequently throw "Resource busy" due to Spotlight 
# indexing the temporary volume. A retry loop is the standard workaround.
MAX_RETRIES=5
for i in $(seq 1 $MAX_RETRIES); do
  if hdiutil create -volname "$APP_NAME_DISPLAY" -srcfolder "$STAGING" -ov -format UDZO -imagekey zlib-level=9 "$OUT_DMG"; then
    break
  else
    echo "hdiutil create failed (attempt $i/$MAX_RETRIES). Retrying in 3 seconds..."
    sleep 3
    if [ "$i" -eq "$MAX_RETRIES" ]; then
      echo "Failed to create DMG after $MAX_RETRIES attempts." >&2
      exit 1
    fi
  fi
done

echo "DMG created at: $OUT_DMG"
