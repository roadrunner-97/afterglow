#!/bin/bash
# Build a portable AppImage of Afterglow.
#
# Output: ./Afterglow-x86_64.AppImage
#
# Strategy:
#   - Build Release with cmake/ninja into build-appimage/
#   - Stage into AppDir/ following AppDir conventions
#   - Use linuxdeploy + linuxdeploy-plugin-qt to bundle Qt6 + transitive deps
#   - Bundle POCL (CPU OpenCL) so the binary always has *some* working device
#   - DO NOT bundle libOpenCL.so.1 — we want the host's loader to honour the
#     user's installed vendor ICD. The loader is tiny and ABI-stable.
#   - DO NOT bundle libGL/libEGL/libGLX — graphics drivers must come from host.
#
# Requires: linuxdeploy, linuxdeploy-plugin-qt, appimagetool on PATH (or in
# scripts/appimage/.tools/, which this script will populate on first run).

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BUILD="$ROOT/build-appimage"
APPDIR="$ROOT/AppDir"
TOOLS="$ROOT/scripts/appimage/.tools"
ASSETS="$ROOT/scripts/appimage"

mkdir -p "$TOOLS"
fetch_tool() {
    local name="$1" url="$2"
    if [ ! -x "$TOOLS/$name" ]; then
        echo ">>> fetching $name"
        curl -fL "$url" -o "$TOOLS/$name"
        chmod +x "$TOOLS/$name"
    fi
}
fetch_tool linuxdeploy           "https://github.com/linuxdeploy/linuxdeploy/releases/download/continuous/linuxdeploy-x86_64.AppImage"
fetch_tool linuxdeploy-plugin-qt "https://github.com/linuxdeploy/linuxdeploy-plugin-qt/releases/download/continuous/linuxdeploy-plugin-qt-x86_64.AppImage"
export PATH="$TOOLS:$PATH"

echo ">>> configuring"
cmake -B "$BUILD" -G Ninja -DCMAKE_BUILD_TYPE=Release "$ROOT"
echo ">>> building"
cmake --build "$BUILD"

echo ">>> staging AppDir"
rm -rf "$APPDIR"
install -Dm755 "$BUILD/bin/afterglow"        "$APPDIR/usr/bin/afterglow"
install -Dm644 "$ASSETS/afterglow.desktop"   "$APPDIR/usr/share/applications/afterglow.desktop"
install -Dm644 "$ASSETS/afterglow.svg"       "$APPDIR/usr/share/icons/hicolor/scalable/apps/afterglow.svg"

# Bundle POCL as a fallback CPU ICD if available on the build host.
# (Arch: pacman -S pocl;  Debian/Ubuntu: apt install pocl-opencl-icd)
POCL_LIB="$(ldconfig -p | awk '/libpocl\.so/ {print $NF; exit}')"
if [ -n "${POCL_LIB:-}" ] && [ -f "$POCL_LIB" ]; then
    echo ">>> bundling POCL: $POCL_LIB"
    install -Dm644 "$POCL_LIB" "$APPDIR/usr/lib/libpocl.so"
else
    echo "!!! POCL not found on build host — AppImage will rely entirely on host vendor ICDs"
fi

# linuxdeploy-plugin-qt finds Qt via qmake6
export QMAKE="${QMAKE:-/usr/bin/qmake6}"

# Tell linuxdeploy not to bundle these — they must come from the host's driver:
export LINUXDEPLOY_OUTPUT_APP_NAME="Afterglow"
export LINUXDEPLOY_OUTPUT_VERSION="${AFTERGLOW_VERSION:-dev-$(git -C "$ROOT" rev-parse --short HEAD 2>/dev/null || echo unknown)}"

# Excludes: graphics + OpenCL loader come from host.
# (linuxdeploy already has a default excludelist for libc/libstdc++/libGL etc;
# we add libOpenCL explicitly to be sure.)
EXCLUDES=(
    --exclude-library=libOpenCL.so.1
    --exclude-library=libGL.so.1
    --exclude-library=libEGL.so.1
    --exclude-library=libGLX.so.0
)

echo ">>> running linuxdeploy"
linuxdeploy \
    --appdir "$APPDIR" \
    --plugin qt \
    --desktop-file "$APPDIR/usr/share/applications/afterglow.desktop" \
    --icon-file    "$APPDIR/usr/share/icons/hicolor/scalable/apps/afterglow.svg" \
    "${EXCLUDES[@]}"

# Replace AppRun with our wrapper that wires up ICD discovery.
install -m755 "$ASSETS/AppRun" "$APPDIR/AppRun"

echo ">>> packaging"
fetch_tool appimagetool "https://github.com/AppImage/appimagetool/releases/download/continuous/appimagetool-x86_64.AppImage"
ARCH=x86_64 appimagetool "$APPDIR" "$ROOT/Afterglow-x86_64.AppImage"

echo ">>> done: $ROOT/Afterglow-x86_64.AppImage"
