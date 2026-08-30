#!/usr/bin/env bash
# check-format.sh
# Verify every C++ source file matches the project .clang-format style.
#
# Skip entirely by creating:    .git/check-coverage-skip  (same flag as coverage)
# Run manually:                 bash scripts/check-format.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_DIR"

CLANG_FORMAT_BIN="${CLANG_FORMAT:-clang-format}"

if [[ -f "$PROJECT_DIR/.git/check-coverage-skip" ]]; then
    echo "check-format: skip flag present — exiting 0"
    exit 0
fi

echo "=== check-format: clang-format ==="
echo "Using: $($CLANG_FORMAT_BIN --version)"

mapfile -t FILES < <(find src plugins tests -name '*.cpp' -o -name '*.h' | sort)

BAD=()
for f in "${FILES[@]}"; do
    if ! "$CLANG_FORMAT_BIN" --dry-run --Werror "$f" 2>/dev/null; then
        BAD+=("$f")
    fi
done

if [[ ${#BAD[@]} -eq 0 ]]; then
    echo "Format: PASS (${#FILES[@]} files checked)"
    exit 0
fi

echo ""
echo "Format FAIL: ${#BAD[@]} file(s) not formatted correctly:"
for f in "${BAD[@]}"; do
    echo "  $f"
done
echo ""
echo "Run: clang-format -i \$(git diff --name-only HEAD | grep -E '\\.(cpp|h)$')"
echo "  or: find src plugins tests -name '*.cpp' -o -name '*.h' | xargs clang-format -i"
exit 1
