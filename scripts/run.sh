#!/usr/bin/env bash
# run.sh — build and launch the app. Extra args are forwarded to the binary.
# Usage: scripts/run.sh [image-path]
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

"$SCRIPT_DIR/build.sh" afterglow
exec ./build/bin/afterglow "$@"
