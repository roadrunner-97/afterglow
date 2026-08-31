# Effect performance benchmarks

The benchmark suite is optional and is not part of `ctest` or the coverage
gate. It requires Google Benchmark and a working OpenCL device.

```bash
cmake -B build -G Ninja -DBUILD_BENCHMARKS=ON
cmake --build build --target afterglow_effect_benchmarks
./build/bin/afterglow_effect_benchmarks --benchmark_min_time=0.2s
```

Use `--benchmark_out=results.json --benchmark_out_format=json` for a
machine-readable baseline. The suite selects the same default OpenCL device as
the application and prints it at startup.

Benchmark groups:

- `KernelOnly`: effect kernels with input and scratch buffers already resident.
- `Transfer`: forced source upload, preview processing, packing, and readback.
- `Repeated`: twelve rapid asynchronous parameter changes through
  `ImageProcessor`; counters show submitted and delivered requests.
- `AllEffects`: the complete stack in live-preview and full-resolution commit
  modes, plus the cached pan/zoom path.

Every group runs at 1920x1080, 3840x2160, and 6000x4000. Live-preview cases
use a half-width, half-height viewport for each source resolution. Pass a
Google Benchmark filter to select a smaller subset when investigating one
effect, for example `--benchmark_filter='Blur|Denoise'`.
