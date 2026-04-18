# Install

How to get the build toolchain + dependencies in place. Once the deps are satisfied, the build is the three-liner from [README.md](README.md).

## Dependencies

Required:

- C++17 compiler (GCC or Clang)
- CMake ≥ 3.16 (≥ 3.25 if you want the `coverage` workflow preset)
- Ninja
- Qt 6 — `Core`, `Gui`, `Widgets`, `Concurrent`, `OpenGLWidgets`
- OpenCL — ICD loader **and** the C++ headers (`cl2.hpp` / `opencl.hpp`)
- OpenGL (libGL)

Optional:

- **LibRaw** — RAW decode support (`.cr2 .cr3 .nef .arw .dng .raf .orf .rw2` …). Detected via `pkg-config`; absent → RAW files silently unavailable.
- **gcovr** — coverage reports
- **ccache** — detected automatically and used as the compiler launcher when present

### Arch Linux

```bash
sudo pacman -S base-devel cmake ninja \
               qt6-base qt6-tools \
               opencl-headers opencl-clhpp ocl-icd \
               libraw              \
               python-gcovr ccache
```

Plus the OpenCL runtime for your GPU:

- AMD: `opencl-rusticl-mesa` or `rocm-opencl-runtime`
- NVIDIA: `opencl-nvidia`
- Intel: `intel-compute-runtime`

### Ubuntu / Debian

```bash
sudo apt install build-essential cmake ninja-build \
                 qt6-base-dev qt6-tools-dev \
                 libopengl-dev \
                 ocl-icd-opencl-dev opencl-clhpp-headers \
                 libraw-dev \
                 gcovr ccache
```

Plus a vendor ICD (`mesa-opencl-icd`, `nvidia-opencl-icd`, `intel-opencl-icd`, …).

### Verifying OpenCL

```bash
clinfo -l    # should list at least one platform + device
```

If `clinfo` finds no devices, `cmake -B build` will still succeed (headers are present) but the app will fail to process images at runtime.

## Build

From the repo root:

```bash
cmake -B build -G Ninja
cmake --build build
```

The configure step will fail fast if OpenCL or any required Qt module is missing. Output binary is `build/bin/lightroom_clone`.

### Options

| Option | Default | Effect |
|---|---|---|
| `-DCOVERAGE=ON` | `OFF` | Strips `-O3`, adds `--coverage`, enables the `coverage` target |

### Presets

`CMakePresets.json` ships two configure presets (`default`, `coverage`) and a `coverage` workflow that configures → builds → tests → generates the HTML report in one shot:

```bash
cmake --workflow --preset coverage
# report: build/coverage/index.html
```

## Tests

```bash
cmake --build build
ctest --test-dir build --output-on-failure -j$(nproc)
```

Test targets live under `tests/` (core, effects, widgets) and are discovered once `enable_testing()` runs during configure.

## Troubleshooting

- **`OpenCL NOT FOUND`** at configure time — install the ICD loader + C++ headers (`ocl-icd-opencl-dev` or `opencl-headers` + `opencl-clhpp`).
- **`clinfo` lists no devices** — install your GPU vendor's OpenCL runtime (see above).
- **clangd marks OpenCL code as unreachable** — clangd doesn't see `HAVE_OPENCL`; the actual compile is fine, ignore the IDE squiggles.
- **Qt6 not found** — some distros ship Qt6 in a non-default prefix; pass `-DCMAKE_PREFIX_PATH=/path/to/qt6` to `cmake -B build`.
