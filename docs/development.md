# Development

## Coverage

One-shot via the `coverage` workflow preset:

```bash
cmake --workflow --preset coverage
```

Report lands in `build/coverage/index.html`. Requires `gcovr` on PATH.

A `pre-push` hook (installed automatically by `cmake -B build`) runs
`scripts/check-coverage.sh`, which builds into `build-coverage/`, runs the
suite, and enforces a 99% line-coverage threshold. Skip a single push with
`touch .git/check-coverage-skip`; lower the bar with `COVERAGE_MIN_LINES=<pct>
git push`.

## Development scripts

Thin wrappers in `scripts/` for the common loops:

| Script | What it does |
|---|---|
| `scripts/build.sh [target...]` | Configure if needed, then build. Extra args become Ninja targets. |
| `scripts/test.sh [pattern]` | Build and run `ctest`. With a pattern, runs only matching tests (`scripts/test.sh CropRotate`). |
| `scripts/run.sh [image-path]` | Build and launch the app, forwarding any args to the binary. |
| `scripts/check-coverage.sh` | Coverage build + threshold check (also wired up as the `pre-push` hook). |
