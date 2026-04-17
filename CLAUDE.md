# Lightroom Clone — Developer Notes

## Build

```bash
cmake -B build -G Ninja    # OpenCL REQUIRED — configure fails if not found
cmake --build build
./build/bin/lightroom_clone
```

Binary: `build/bin/lightroom_clone`
Libraries: `build/lib/libphotoeditor_core.so`, `build/lib/libphotoeditor_widgets.so`, `build/lib/libphotoeditor_ui.so`
Effect libs: `build/plugins/lib*_effect.a` (statically linked into the binary)

To reconfigure: `cmake -B build -G Ninja` from the repo root. Ninja parallelises automatically, so `-j$(nproc)` is unnecessary on `cmake --build`.

### Running tests

```bash
cmake --build build
ctest --test-dir build --output-on-failure -j$(nproc)
```

### Coverage build

One-shot via the `coverage` workflow preset (configure → build → test → coverage report):

```bash
cmake --workflow --preset coverage
```

Or manually:

```bash
cmake -B build -G Ninja -DCOVERAGE=ON   # enable --coverage flags (gcovr required)
cmake --build build
ctest --test-dir build -j$(nproc)                 # populate .gcda files
cmake --build build --target coverage             # generate report from existing .gcda files
```

Report lands in `build/coverage/index.html` (HTML detail) and `build/coverage/coverage.xml`.

To check a specific file's line coverage without regenerating the full HTML report:

```bash
gcovr \
  --root . \
  --object-directory build \
  --exclude "build/" --exclude "tests/" --exclude "src/ui/" \
  --exclude "src/core/ImageProcessor\.cpp" \
  --exclude "src/core/RawLoader\.cpp" \
  --exclude "src/main\.cpp" \
  --exclude-throw-branches \
  --exclude-unreachable-branches \
  --merge-mode-functions=merge-use-line-min \
  --gcov-ignore-errors=no_working_dir_found \
  --print-summary
```

`gcovr` must be on PATH (`sudo pacman -S python-gcovr`). The `--gcov-ignore-errors=no_working_dir_found` flag suppresses a harmless warning from system headers that lack debug info.

### Excluding infeasible code from coverage

Some paths (OpenCL `cl::Error` catch blocks, enumeration fallbacks) cannot be driven from a unit test. Two mechanisms keep them out of the headline number:

- **Source markers** — wrap the block with `// GCOVR_EXCL_START` / `// GCOVR_EXCL_STOP` (or `// GCOVR_EXCL_LINE` for a single line). The marker lines themselves are also excluded.
- **gcovr flags** — `--exclude-throw-branches` drops compiler-synthesised exception branches (every potentially-throwing call emits one); `--exclude-unreachable-branches` drops branches the compiler proved unreachable. Both are on by default in the `coverage` target.

---

## Architecture

Four shared libraries + static effect libs + a thin `main.cpp`:

| Target | Sources | Role |
|---|---|---|
| `libphotoeditor_core` | `src/core/` | `EffectManager`, `ImageProcessor`, `GpuDeviceRegistry`, `GpuPipeline`, `RawLoader` |
| `libphotoeditor_widgets` | `src/widgets/` | Reusable custom Qt widgets (`ParamSlider`) |
| `libphotoeditor_ui` | `src/ui/` | `PhotoEditorApp` (main window), `ViewportWidget` |
| `lightroom_clone` | `src/main.cpp` | Composition root: creates effects, passes to `PhotoEditorApp` |
| `*_effect.a` | `plugins/*/` | Effect libs, statically linked (no dlopen) |

Effects are **static libraries** instantiated in `main.cpp` — no runtime loading.

---

## Key files

- `src/core/PhotoEditorEffect.h` — abstract effect interface (`parametersChanged`, `liveParametersChanged` signals)
- `src/core/EffectManager.h/.cpp` — owns effects, tracks enabled state (`EffectEntry { effect, bool enabled }`)
- `src/core/ImageProcessor.h/.cpp` — async processing via `QtConcurrent::run` + `QFutureWatcher<QImage>`; generation counter discards stale results; parameters snapshotted on the main thread before dispatch
- `src/core/GpuDeviceRegistry.h/.cpp` — singleton registry of OpenCL devices; `enumerate()` at startup; `setDevice(idx)` bumps `revision()` so all GpuContexts reinitialise on next GPU call
- `src/core/GpuPipeline.h/.cpp` — persistent GPU buffer (uploaded once per image); chains all effect kernels, single `finish()`, single readback; owns `srcBuf`/`workBuf`/`auxBuf`
- `src/core/IGpuEffect.h` — mixin interface for effects in the shared GPU pipeline
- `src/core/RawLoader.h/.cpp` — loads RAW files into 16-bit RGBX64 (requires HAVE_LIBRAW)
- `src/ui/PhotoEditorApp.h/.cpp` — main window; View→Effects menu; GPU selector combo
- `src/ui/ViewportWidget.h/.cpp` — QOpenGLWidget; pan/zoom; uploads processed images to a GL texture
- `src/main.cpp` — composition root: enumerate GPU, create all effects, pass to PhotoEditorApp
- `plugins/CMakeLists.txt` — all effect subdirectories + `all_effects` INTERFACE aggregator

---

## Image processing pipeline

1. **On load** — `openImage()` → `triggerReprocess()` → `processImageAsync()` (full run)
2. **On parameter change** — `onParametersChanged()` → `triggerReprocess()` → `processImageAsync()` (full run)
3. **On pan/zoom** — `triggerViewportUpdate()` → `processImageAsync(viewportOnly=true)` (skip effects, re-downsample only)
4. **GPU fast path** — all effects implement `IGpuEffect`: single upload, chain kernels, single readback
5. Result delivered via `processingComplete` signal → `onProcessingComplete()` → `ViewportWidget::setImage()`

---

## Coding conventions

### Pixel access
Always use `scanLine()` pointer access — never `pixel()`/`setPixel()`:
```cpp
QRgb* row = reinterpret_cast<QRgb*>(result.scanLine(y));
int r = qRed(row[x]);
row[x] = qRgb(r, g, b);
```

### Numeric parameter controls — use `ParamSlider`
All numeric effect parameters use `ParamSlider` (`src/widgets/ParamSlider.h`).
It encapsulates the label + QSlider + QDoubleSpinBox trio.
Two signals: `valueChanged(double)` (every drag), `editingFinished()` (slider release / spinbox commit).

```cpp
// Integer steps:
auto* p = new ParamSlider("Brightness", -100, 100);
// Sub-integer steps (0.1 resolution, 1 decimal place):
auto* p = new ParamSlider("Saturation", -20.0, 20.0, 0.1, 1);

connect(p, &ParamSlider::editingFinished, this, [this]() { emit parametersChanged(); });
connect(p, &ParamSlider::valueChanged,    this, [this](double) { emit liveParametersChanged(); });
layout->addWidget(p);
```

### OpenCL GPU path
Every effect with a GPU path wraps all GPU code in an anonymous `namespace {}`:

```cpp
#define CL_HPP_TARGET_OPENCL_VERSION 200
#define CL_HPP_MINIMUM_OPENCL_VERSION 110
#define CL_HPP_ENABLE_EXCEPTIONS
#include <CL/opencl.hpp>
#include "GpuDeviceRegistryOCL.h"

namespace {
    static const char* GPU_KERNEL_SOURCE = R"CL(...)CL";
    struct GpuContext { ... };   // singleton with revision-based re-init
    static std::mutex gpuMutex;
    static QImage processImageGPU(...) { ... }
} // namespace
```

**The `namespace { }` wrapper is mandatory** — prevents ODR violations between static libs linked into the same binary.

Each effect also implements `IGpuEffect` for the shared pipeline (no per-effect upload/readback overhead):
```cpp
bool initGpuKernels(cl::Context& ctx, cl::Device& dev) override;
bool enqueueGpu(cl::CommandQueue&, cl::Buffer& buf, cl::Buffer& aux,
                int w, int h, int stride, bool is16bit,
                const QMap<QString,QVariant>&) override;
// Do NOT call queue.finish() in enqueueGpu — GpuPipeline does it once
```

- Pixel format: `QImage::Format_RGB32` (`0xFFRRGGBB`); `stride = bytesPerLine() / 4`
- Pixel format: `QImage::Format_RGBX64` (`ushort4`); `stride = bytesPerLine() / 8`
- Upload: `CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR`
- Download: `enqueueReadBuffer(..., CL_TRUE, ...)` (blocking)
- `GpuContext::instance()` must be called with `gpuMutex` held; checks `GpuDeviceRegistry::instance().revision()` for device changes

---

## Adding a new effect

1. Create `plugins/myeffect/` with `MyEffect.h`, `MyEffect.cpp`, `CMakeLists.txt`
2. Inherit `PhotoEditorEffect` and `IGpuEffect`; implement all pure virtuals; no `extern "C"` exports needed
3. Add `add_subdirectory(myeffect)` + `myeffect_effect` to `plugins/CMakeLists.txt`
4. `#include "MyEffect.h"` + `effectManager->addEffect(new MyEffect())` in `src/main.cpp`

Effect CMake boilerplate:
```cmake
add_library(myeffect_effect STATIC MyEffect.cpp MyEffect.h)
find_package(OpenCL REQUIRED)
target_link_libraries(myeffect_effect PRIVATE OpenCL::OpenCL)
target_compile_definitions(myeffect_effect PRIVATE HAVE_OPENCL)
target_link_libraries(myeffect_effect PRIVATE photoeditor_core photoeditor_widgets Qt6::Core Qt6::Gui)
target_include_directories(myeffect_effect PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
set_target_properties(myeffect_effect PROPERTIES AUTOMOC ON)
```

---

## Dependencies

- Qt6: Core, Gui, Widgets, Concurrent, OpenGLWidgets
- OpenCL REQUIRED (`opencl-clhpp` — `sudo pacman -S opencl-clhpp`)
- OpenGL REQUIRED (libGL)
- LibRaw optional (detected via pkg-config)
- C++17, `-O3`

---

## Known gotchas

- **clangd false positives** inside `HAVE_OPENCL` blocks: clangd doesn't see the compile definition and marks OpenCL code as unreachable. The actual compiler is fine — ignore these IDE errors.
- **`QFutureWatcher` lambda**: the worker lambda must not capture `this` — capture only the image by value and the snapshotted effect calls by move.
- **Effect collapse state**: uses `std::shared_ptr<bool>` captured by value in the lambda — no heap-allocated state structs needed.
- **No CPU fallback** — all effects use GPU paths only; if GPU fails, `processImage` returns `{}`.
