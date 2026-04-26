# Afterglow

A Qt6 / OpenCL photo editor. Non-destructive, GPU-accelerated, effects-stacked pipeline.

## Requirements

- C++17 compiler, CMake ≥ 3.16 (≥ 3.25 for workflow presets), Ninja
- Qt6: `Core`, `Gui`, `Widgets`, `Concurrent`, `OpenGLWidgets`
- OpenCL — **required** (`opencl-clhpp` on Arch). There is no CPU fallback.
- OpenGL (libGL)
- LibRaw — *optional*, detected via `pkg-config`; enables RAW file support when present

## Build

```bash
cmake -B build -G Ninja
cmake --build build
./build/bin/afterglow
```

Or via the wrapper scripts in `scripts/` (see [Development scripts](#development-scripts)).

Outputs:

- `build/bin/afterglow` — single self-contained ELF; the project's own libs
  (`photoeditor_core`, `photoeditor_widgets`, `photoeditor_ui`) and every
  `plugins/*_effect.a` are statically linked in. Only Qt6, OpenCL, OpenGL,
  and (optionally) LibRaw are shared system deps at runtime.

### Tests

```bash
ctest --test-dir build --output-on-failure -j$(nproc)
```

### Coverage

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

### Development scripts

Thin wrappers in `scripts/` for the common loops:

| Script | What it does |
|---|---|
| `scripts/build.sh [target...]` | Configure if needed, then build. Extra args become Ninja targets. |
| `scripts/test.sh [pattern]` | Build and run `ctest`. With a pattern, runs only matching tests (`scripts/test.sh CropRotate`). |
| `scripts/run.sh [image-path]` | Build and launch the app, forwarding any args to the binary. |
| `scripts/check-coverage.sh` | Coverage build + threshold check (also wired up as the `pre-push` hook). |

## Install

`cmake --install build` drops the `afterglow` binary into
`${CMAKE_INSTALL_PREFIX}/bin/`. The default prefix is `/usr/local`, so it
needs `sudo`; for a user-local install pass `--prefix ~/.local` and make
sure `~/.local/bin` is on your `PATH`.

```sh
cmake --install build --prefix ~/.local      # user-local, no sudo
sudo cmake --install build                   # system-wide, default prefix
```

To undo it: `cmake --build build --target uninstall` reads
`build/install_manifest.txt` and removes exactly what was installed (run
with `sudo` if the install was `sudo`).

## Releases

### Versioning

Versions follow [Semantic Versioning](https://semver.org) with a **pre-1.0
ramp** — i.e. while the major version is `0`, anything goes and there are no
backwards-compatibility promises. Tags are prefixed with `v`.

```
v MAJOR . MINOR . PATCH
   │       │       │
   │       │       └── bug fixes only       e.g. v0.1.0 → v0.1.1
   │       └────────── features / changes   e.g. v0.1.0 → v0.2.0
   └────────────────── reserved for "I'd be happy if a stranger downloaded this"
```

Rules of thumb while we're still on `0.x`:

- New effect, new UI feature, refactor that changes user-visible behaviour → bump **MINOR**
- Crash fix, rendering bug, build fix → bump **PATCH**
- Don't worry about MAJOR until the app feels stable enough to commit to APIs

### Cutting a release

```sh
git tag v0.2.0
git push origin v0.2.0
```

Pushing a `v*` tag fires `.github/workflows/release.yml`: it rebuilds on a
clean Ubuntu runner, runs the full test suite, strips the binary, packages
`afterglow-<version>-linux-x86_64.tar.gz` + a SHA-256 sum, and publishes a
GitHub Release with both attached. Auto-generated changelog notes are
appended below the install instructions.

Releases land at <https://github.com/roadrunner-97/afterglow/releases>.

### Re-doing a release

If the workflow fails (e.g. an apt package got renamed) and you need to
retry the same version, delete both the tag and the half-created GitHub
Release before re-tagging:

```sh
git push --delete origin v0.2.0     # remove from the remote
git tag -d v0.2.0                   # remove locally
# delete the draft Release in the GitHub web UI
git tag v0.2.0 && git push origin v0.2.0
```

If the broken release already shipped to people, **don't reuse the
version** — bump the patch (`v0.2.1`) and ship a fix.

## Architecture

Four Qt targets plus a set of statically-linked effect libs:

| Target | Role |
|---|---|
| `libphotoeditor_core` | `EffectManager`, `ImageProcessor`, `GpuDeviceRegistry`, `GpuPipeline`, `RawLoader` |
| `libphotoeditor_widgets` | Reusable custom widgets (`ParamSlider`) |
| `libphotoeditor_ui` | `PhotoEditorApp` main window, `ViewportWidget` (QOpenGLWidget) |
| `afterglow` | Composition root — instantiates effects, passes them to the app |
| `plugins/*_effect.a` | Individual effects, statically linked (no dlopen) |

### Pipeline

1. Load image → `ImageProcessor::processImageAsync` (QtConcurrent + `QFutureWatcher`)
2. Parameters snapshotted on the main thread; worker is dispatched with a generation counter so stale results are discarded
3. `GpuPipeline` uploads the source buffer once per image, then chains every enabled effect's `enqueueGpu` kernel against a persistent work buffer with a single `finish()` and a single readback
4. On pan/zoom only the downsample step re-runs — effects are skipped
5. Result delivered via signal → `ViewportWidget::setImage` → uploaded to a GL texture

## Effects

Instantiated in `src/main.cpp`, in pipeline order:

`CropRotate`, `HotPixel`, `Exposure`, `WhiteBalance`, `Brightness`, `Saturation`, `Blur`, `Grayscale`, `Unsharp`, `Denoise`, `Vignette`, `FilmGrain`, `SplitToning`, `Clarity`, `ColorBalance`.

Each effect inherits `PhotoEditorEffect` (the Qt-facing interface) and `IGpuEffect` (the pipeline mixin).

### Adding a new effect

1. Create `plugins/myeffect/MyEffect.{h,cpp}` inheriting `PhotoEditorEffect` and `IGpuEffect`
2. Add `plugins/myeffect/CMakeLists.txt` — a one-liner: `add_effect_plugin(My)`
3. Register in `plugins/CMakeLists.txt`: `add_subdirectory(myeffect)` + append `myeffect_effect` to the `all_effects` INTERFACE list
4. In `src/main.cpp`, `#include "MyEffect.h"` and `effects->addEffect(new MyEffect())`

See `CLAUDE.md` for conventions (pixel access via `scanLine()`, `ParamSlider` usage, mandatory `namespace { }` wrapper around OpenCL code, revision-based GPU context re-init).

## GPU device selection

`GpuDeviceRegistry` enumerates OpenCL devices at startup. The top-right combo box in the main window switches devices; `setDevice(idx)` bumps a revision counter so every per-effect `GpuContext` reinitialises on its next call.

## Known limitations

- OpenCL is a hard build requirement — no CPU fallback exists; if the GPU fails, `processImage` returns an empty image.
- No CL/GL interop — plain readback is used (AMD RDNA4 on Wayland doesn't support `cl_khr_gl_sharing`).
