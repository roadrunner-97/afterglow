# GPU optimization follow-ups

The GPU pipeline review identified the following opportunities. Redundant
`cl::CommandQueue::finish()` calls were removed separately; the final blocking
readback remains the synchronization point for each run.

## Priorities

### 1. Optimize shared blur kernels

The Gaussian blur kernels recalculate `native_exp()` for every pixel and tap,
and adjacent work-items repeatedly fetch overlapping neighborhoods from global
memory. The vertical pass also has poorly coalesced reads.

Potential improvements:

- Precompute normalized one-dimensional weights once per radius and upload them
  in a small constant buffer.
- Implement box blur with a sliding window, reducing each pass from O(radius)
  work per pixel to O(1).
- Tile rows and columns in local memory, including halo pixels.
- Approximate large Gaussian radii with repeated box blurs or a
  downsample/blur/upsample path.

Profile these approaches on both GPU and CPU OpenCL devices before selecting a
default.

### 2. Reuse the Denoise scratch buffer

`DenoiseEffect::enqueueGpu()` allocates a full-frame `cl_float4` buffer on every
run. Store it on the effect and resize it only when the processing dimensions
change, following the existing Clarity and Unsharp pattern.

### 3. Eliminate full-frame buffer copies

Denoise, Hot Pixel, Clarity, and Unsharp write results to the auxiliary buffer
and then copy the entire frame back to the pipeline's primary buffer. Extend the
`IGpuEffect` contract so an effect can report or swap which buffer owns its
result. Clarity and Unsharp may also be able to write their final combine pass
directly to the primary buffer because each work-item reads only the
corresponding original pixel; verify OpenCL aliasing behavior across supported
devices first.

### 4. Add a linear-light mip pyramid

Preview downsampling currently walks essentially the complete visible source
region and, for sRGB inputs, repeats transfer-function evaluation. Build and
retain a linear-light mip pyramid after upload/decode, then sample the closest
level for interactive pan, zoom, and live previews.

### 5. Reuse more of the full-resolution effect chain

Every commit decodes the source and reruns every enabled effect. Keep an
immutable full-resolution linear decode cache separate from the processed
cache. Investigate selective checkpoints, such as the prefix before the effect
currently being edited or the point before the first spatial effect, while
balancing GPU memory use.

### 6. Consolidate OpenCL program compilation

Effects independently compile programs containing repeated color and blur
helpers. Cache programs by source, build options, device, and driver, or build a
consolidated pipeline program. Persistent compiled binaries could additionally
reduce cold-start and device-switch latency.

### 7. Tune work-group sizes using profiling

All launches currently let the driver select the local work size. Add optional
profiling around decode, individual effects, downsampling, packing, and
readback, then benchmark work-group shapes such as 16x16 and 32x8. Round global
dimensions upward and retain kernel boundary checks. Avoid a single hard-coded
choice without coverage across supported GPU and CPU OpenCL implementations.
