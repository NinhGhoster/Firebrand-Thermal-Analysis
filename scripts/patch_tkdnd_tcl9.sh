#!/usr/bin/env bash
# Patch the bundled tkdnd macOS arm64 payload to a Tcl 9.0 build.
#
# Why: Homebrew python-tk and uv's python-build-standalone CPython both ship
# Tcl/Tk 9.0.x, but the tkdnd 2.9.3 dylib bundled with tkinterdnd2-universal is
# built for Tcl 8.6 and fails to load with:
#   _tkinter.TclError: cannot find symbol "tkdnd_Init"
# This script replaces it with the official tkdnd 2.9.5 Tcl 9.0 arm64 build.
# Idempotent: re-run after every `uv sync`. Requires uv on PATH.
#
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TARGET_DIR="$(cd "$REPO_DIR" && uv run python -c 'import os, tkinterdnd2; print(os.path.join(os.path.dirname(tkinterdnd2.__file__), "tkdnd", "osx-arm64"))')"
[ -d "$TARGET_DIR" ] || { echo "error: osx-arm64 tkdnd payload dir not found" >&2; exit 1; }

TKDND_VERSION="2.9.5"
TARBALL="tkdnd-${TKDND_VERSION}-macOS-tcl9.0-arm64-x64-14.2.1.tgz"
RELEASE="tkdnd-release-test-v${TKDND_VERSION}"
URL="https://github.com/petasis/tkdnd/releases/download/${RELEASE}/${TARBALL}"
NEW_DYLIB="libtcl9tkdnd${TKDND_VERSION}.dylib"

if [ -f "$TARGET_DIR/$NEW_DYLIB" ]; then
    echo "tkdnd ${TKDND_VERSION} (Tcl 9.0) already installed in $TARGET_DIR"
    exit 0
fi

TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

echo "Downloading tkdnd ${TKDND_VERSION} (Tcl 9.0) build..."
curl -sSL -o "$TMP/$TARBALL" "$URL"
tar -xzf "$TMP/$TARBALL" -C "$TMP"

PAYLOAD_DIR="$TMP/tkdnd${TKDND_VERSION}"
[ -f "$PAYLOAD_DIR/$NEW_DYLIB" ] || { echo "error: $NEW_DYLIB not found in tarball" >&2; exit 1; }

echo "Patching $TARGET_DIR"
cp -p "$PAYLOAD_DIR"/* "$TARGET_DIR/"
rm -f "$TARGET_DIR"/libtkdnd*.dylib
echo "Done. tkdnd ${TKDND_VERSION} (Tcl 9.0) installed."