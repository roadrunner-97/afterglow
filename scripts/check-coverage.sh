#!/usr/bin/env bash
# check-coverage.sh
# Build (ccache + ninja), run tests, and verify line coverage meets threshold.
#
# Skip entirely by creating:    .git/check-coverage-skip
# Threshold override:           COVERAGE_MIN_LINES env var (default 99.0).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_DIR"

if [[ -f "$PROJECT_DIR/.git/check-coverage-skip" ]]; then
  echo "check-coverage: skip flag present ($PROJECT_DIR/.git/check-coverage-skip) — exiting 0"
  exit 0
fi

BUILD_DIR="build-coverage"
THRESHOLD="${COVERAGE_MIN_LINES:-99.0}"

echo "=== check-coverage: configure ==="
cmake -B "$BUILD_DIR" \
  -G Ninja \
  -DCMAKE_C_COMPILER_LAUNCHER=ccache \
  -DCMAKE_CXX_COMPILER_LAUNCHER=ccache \
  -DCOVERAGE=ON \
  -DCMAKE_BUILD_TYPE=Debug \
  2>&1 | tail -5

echo "=== check-coverage: build ==="
cmake --build "$BUILD_DIR" -j"$(nproc)"

echo "=== check-coverage: test ==="
# Tests run serially: they share a single OpenCL GPU and deadlock in the
# ROCm driver under parallel gcov-instrumented load.
ctest --test-dir "$BUILD_DIR" --output-on-failure

echo "=== check-coverage: coverage (threshold ${THRESHOLD}%) ==="
GCOVR_OUT=$(gcovr \
  --root . \
  --object-directory "$BUILD_DIR" \
  --exclude "build.*/" --exclude "tests/" --exclude "src/ui/" \
  --exclude "src/core/ImageProcessor\.cpp" \
  --exclude "src/core/RawLoader\.cpp" \
  --exclude "src/main\.cpp" \
  --exclude-throw-branches \
  --exclude-unreachable-branches \
  --merge-mode-functions=merge-use-line-min \
  --gcov-ignore-errors=no_working_dir_found \
  --print-summary 2>&1)

echo "$GCOVR_OUT"

LINE_PCT=$(echo "$GCOVR_OUT" | grep -oP 'lines: \K[0-9]+\.[0-9]+' || true)
if [[ -z "$LINE_PCT" ]]; then
  echo "ERROR: could not parse line coverage from gcovr output"
  exit 2
fi

echo ""
echo "Line coverage: ${LINE_PCT}% (threshold: ${THRESHOLD}%)"

if python3 -c "import sys; sys.exit(0 if float('${LINE_PCT}') >= float('${THRESHOLD}') else 1)" 2>/dev/null; then
  echo "Coverage: PASS (${LINE_PCT}% >= ${THRESHOLD}%)"
  exit 0
else
  echo ""
  echo "Coverage FAIL: ${LINE_PCT}% < ${THRESHOLD}%"
  echo "Either add tests for the uncovered lines, mark infeasible paths with"
  echo "// GCOVR_EXCL_START / _STOP / _LINE markers, or lower COVERAGE_MIN_LINES."
  exit 2
fi
