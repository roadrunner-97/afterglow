#!/usr/bin/env bash
# test.sh — incremental build + ctest. Optional regex selects a subset.
# Usage: scripts/test.sh [pattern]
#   scripts/test.sh                 # run all tests
#   scripts/test.sh CropRotate      # run tests matching the regex
#   scripts/test.sh Effects_        # run every effect test
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

"$SCRIPT_DIR/build.sh"

if [[ $# -ge 1 ]]; then
  ctest --test-dir build --output-on-failure -j8 -R "$1"
else
  ctest --test-dir build --output-on-failure -j8
fi
