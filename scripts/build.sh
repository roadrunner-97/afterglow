#!/usr/bin/env bash
# build.sh — configure (if needed) and build.
# Usage: scripts/build.sh [target...]
#   scripts/build.sh                    # build everything
#   scripts/build.sh afterglow          # build just the binary
#   scripts/build.sh test_croprotate    # build a single test target
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

if [[ ! -f build/build.ninja ]]; then
  cmake -B build -G Ninja
fi

if [[ $# -gt 0 ]]; then
  cmake --build build --target "$@"
else
  cmake --build build
fi
