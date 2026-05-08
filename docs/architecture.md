# Architecture

## Targets

Four Qt targets plus a set of statically-linked effect libs:

| Target | Role |
|---|---|
| `libphotoeditor_core` | `EffectManager`, `ImageProcessor`, `GpuDeviceRegistry`, `GpuPipeline`, `RawLoader` |
| `libphotoeditor_widgets` | Reusable custom widgets (`ParamSlider`) |
| `libphotoeditor_ui` | `PhotoEditorApp` main window, `ViewportWidget` (QOpenGLWidget) |
| `afterglow` | Composition root — instantiates effects, passes them to the app |
| `plugins/*_effect.a` | Individual effects, statically linked (no dlopen) |

## Pipeline

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
