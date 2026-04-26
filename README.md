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

- `build/bin/afterglow` — the app
- `build/lib/libphotoeditor_core.so`, `libphotoeditor_widgets.so`, `libphotoeditor_ui.so`
- `build/plugins/lib*_effect.a` — effect libraries, statically linked into the binary

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

## TODO

- [ ] **CMake `install` target** — no `install()` rules exist today, so `cmake --install build` is a no-op and there's no way to package or system-install the binary. Needs rules for `afterglow` (→ `bin/`), the three shared libs (→ `lib/`), and an RPATH / `CMAKE_INSTALL_RPATH` setup so the installed binary finds `libphotoeditor_*.so`. Test with `cmake --install build --prefix /tmp/lc && /tmp/lc/bin/afterglow`.

## Known limitations

- OpenCL is a hard build requirement — no CPU fallback exists; if the GPU fails, `processImage` returns an empty image.
- No CL/GL interop — plain readback is used (AMD RDNA4 on Wayland doesn't support `cl_khr_gl_sharing`).
