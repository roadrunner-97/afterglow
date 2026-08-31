#include "GpuPipeline.h"
#include "IGpuEffect.h"
#include "GpuDeviceRegistry.h"
#include "GpuDeviceRegistryOCL.h"
#include "color_kernels.h"
#include <QDebug>
#include <algorithm>

// ── Pipeline kernels ─────────────────────────────────────────────────────────
//
// Work / aux buffers are cl_float4 in scene-linear sRGB primaries.  Values are
// nominally [0, 1] but may exceed 1.0 (scene-linear HDR) during the effect
// chain — the final pack kernel clamps immediately before readback.
//
// Downsample kernels box-average the source crop region and emit float4 linear
// pixels.  Three variants cover the two input depths and two gamma encodings:
//   * 8-bit uint RGB32, sRGB-gamma encoded   (JPEG/PNG loaded via QImage)
//   * 16-bit ushort4 RGBX64, sRGB-gamma      (legacy 16-bit sRGB inputs)
//   * 16-bit ushort4 RGBX64, scene-linear    (RAW via LibRaw with gamm=1.0)
//
// cropX0/Y0/X1/Y1 are the visible region in source image pixels (may extend
// outside [0..srcW/H) — those pixels are output as black for letterboxing).
static const char *PIPELINE_KERNEL_SOURCE = COLOR_KERNELS_SRC R"CL(

// Shared crop-to-source-range helper.  Returns [sx0, sx1) × [sy0, sy1) for the
// dest pixel (dx, dy); callers check sx0<sx1 && sy0<sy1 before sampling.
static void crop_range(int dx, int dy, int srcW, int srcH, int dstW, int dstH,
                       float cropX0, float cropY0, float cropX1, float cropY1,
                       int* sx0, int* sy0, int* sx1, int* sy1)
{
    float rgnW = cropX1 - cropX0, rgnH = cropY1 - cropY0;
    float sx0f = cropX0 + (float)dx       * rgnW / dstW;
    float sy0f = cropY0 + (float)dy       * rgnH / dstH;
    float sx1f = cropX0 + (float)(dx + 1) * rgnW / dstW;
    float sy1f = cropY0 + (float)(dy + 1) * rgnH / dstH;
    *sx0 = max(0, (int)sx0f);
    *sy0 = max(0, (int)sy0f);
    *sx1 = min(srcW, (int)ceil(sx1f));
    *sy1 = min(srcH, (int)ceil(sy1f));
}

__kernel void preview_downsample_8bit_srgb_to_linear(
    __global const uint* src, __global float4* dst,
    int srcW, int srcH, int srcStride, int dstW, int dstH,
    float cropX0, float cropY0, float cropX1, float cropY1)
{
    int dx = get_global_id(0), dy = get_global_id(1);
    if (dx >= dstW || dy >= dstH) return;

    int sx0, sy0, sx1, sy1;
    crop_range(dx, dy, srcW, srcH, dstW, dstH, cropX0, cropY0, cropX1, cropY1,
               &sx0, &sy0, &sx1, &sy1);
    if (sx0 >= sx1 || sy0 >= sy1) { dst[dy*dstW + dx] = (float4)(0, 0, 0, 1); return; }

    float r = 0, g = 0, b = 0; int n = 0;
    for (int sy = sy0; sy < sy1; ++sy)
    for (int sx = sx0; sx < sx1; ++sx) {
        uint p = src[sy*srcStride + sx];
        r += srgb_to_linear(((p >> 16) & 0xFFu) * (1.0f/255.0f));
        g += srgb_to_linear(((p >>  8) & 0xFFu) * (1.0f/255.0f));
        b += srgb_to_linear(( p        & 0xFFu) * (1.0f/255.0f));
        ++n;
    }
    float inv = 1.0f / (float)n;
    dst[dy*dstW + dx] = (float4)(r * inv, g * inv, b * inv, 1.0f);
}

__kernel void preview_downsample_16bit_srgb_to_linear(
    __global const ushort* src, __global float4* dst,
    int srcW, int srcH, int srcStride, int dstW, int dstH,
    float cropX0, float cropY0, float cropX1, float cropY1)
{
    int dx = get_global_id(0), dy = get_global_id(1);
    if (dx >= dstW || dy >= dstH) return;

    int sx0, sy0, sx1, sy1;
    crop_range(dx, dy, srcW, srcH, dstW, dstH, cropX0, cropY0, cropX1, cropY1,
               &sx0, &sy0, &sx1, &sy1);
    if (sx0 >= sx1 || sy0 >= sy1) { dst[dy*dstW + dx] = (float4)(0, 0, 0, 1); return; }

    float r = 0, g = 0, b = 0; int n = 0;
    for (int sy = sy0; sy < sy1; ++sy)
    for (int sx = sx0; sx < sx1; ++sx) {
        int i = (sy*srcStride + sx) * 4;
        r += srgb_to_linear(src[i  ] * (1.0f/65535.0f));
        g += srgb_to_linear(src[i+1] * (1.0f/65535.0f));
        b += srgb_to_linear(src[i+2] * (1.0f/65535.0f));
        ++n;
    }
    float inv = 1.0f / (float)n;
    dst[dy*dstW + dx] = (float4)(r * inv, g * inv, b * inv, 1.0f);
}

__kernel void preview_downsample_16bit_linear(
    __global const ushort* src, __global float4* dst,
    int srcW, int srcH, int srcStride, int dstW, int dstH,
    float cropX0, float cropY0, float cropX1, float cropY1)
{
    int dx = get_global_id(0), dy = get_global_id(1);
    if (dx >= dstW || dy >= dstH) return;

    int sx0, sy0, sx1, sy1;
    crop_range(dx, dy, srcW, srcH, dstW, dstH, cropX0, cropY0, cropX1, cropY1,
               &sx0, &sy0, &sx1, &sy1);
    if (sx0 >= sx1 || sy0 >= sy1) { dst[dy*dstW + dx] = (float4)(0, 0, 0, 1); return; }

    float r = 0, g = 0, b = 0; int n = 0;
    for (int sy = sy0; sy < sy1; ++sy)
    for (int sx = sx0; sx < sx1; ++sx) {
        int i = (sy*srcStride + sx) * 4;
        r += src[i  ] * (1.0f/65535.0f);
        g += src[i+1] * (1.0f/65535.0f);
        b += src[i+2] * (1.0f/65535.0f);
        ++n;
    }
    float inv = 1.0f / (float)n;
    dst[dy*dstW + dx] = (float4)(r * inv, g * inv, b * inv, 1.0f);
}

// 1:1 decode kernels: convert the full-res srcBuf into a full-res float4 linear
// buffer (m_processedBuf) without changing resolution.  Used at the start of a
// Commit run, before effects chain on top of m_processedBuf.
__kernel void fullres_decode_8bit_srgb_to_linear(
    __global const uint* src, __global float4* dst,
    int w, int h, int srcStride)
{
    int x = get_global_id(0), y = get_global_id(1);
    if (x >= w || y >= h) return;
    uint p = src[y*srcStride + x];
    float r = srgb_to_linear(((p >> 16) & 0xFFu) * (1.0f/255.0f));
    float g = srgb_to_linear(((p >>  8) & 0xFFu) * (1.0f/255.0f));
    float b = srgb_to_linear(( p        & 0xFFu) * (1.0f/255.0f));
    dst[y*w + x] = (float4)(r, g, b, 1.0f);
}

__kernel void fullres_decode_16bit_srgb_to_linear(
    __global const ushort* src, __global float4* dst,
    int w, int h, int srcStride)
{
    int x = get_global_id(0), y = get_global_id(1);
    if (x >= w || y >= h) return;
    int i = (y*srcStride + x) * 4;
    float r = srgb_to_linear(src[i  ] * (1.0f/65535.0f));
    float g = srgb_to_linear(src[i+1] * (1.0f/65535.0f));
    float b = srgb_to_linear(src[i+2] * (1.0f/65535.0f));
    dst[y*w + x] = (float4)(r, g, b, 1.0f);
}

__kernel void fullres_decode_16bit_linear(
    __global const ushort* src, __global float4* dst,
    int w, int h, int srcStride)
{
    int x = get_global_id(0), y = get_global_id(1);
    if (x >= w || y >= h) return;
    int i = (y*srcStride + x) * 4;
    float r = src[i  ] * (1.0f/65535.0f);
    float g = src[i+1] * (1.0f/65535.0f);
    float b = src[i+2] * (1.0f/65535.0f);
    dst[y*w + x] = (float4)(r, g, b, 1.0f);
}

// Downsample a float4-linear full-res buffer (m_processedBuf) to a smaller
// float4 preview buffer, using the same crop-region box average as the other
// downsample kernels.  No colour-space conversion — source and destination are
// both linear float4.
__kernel void preview_downsample_float4_linear(
    __global const float4* src, __global float4* dst,
    int srcW, int srcH, int srcStride, int dstW, int dstH,
    float cropX0, float cropY0, float cropX1, float cropY1)
{
    int dx = get_global_id(0), dy = get_global_id(1);
    if (dx >= dstW || dy >= dstH) return;

    int sx0, sy0, sx1, sy1;
    crop_range(dx, dy, srcW, srcH, dstW, dstH, cropX0, cropY0, cropX1, cropY1,
               &sx0, &sy0, &sx1, &sy1);
    if (sx0 >= sx1 || sy0 >= sy1) { dst[dy*dstW + dx] = (float4)(0, 0, 0, 1); return; }

    float r = 0, g = 0, b = 0; int n = 0;
    for (int sy = sy0; sy < sy1; ++sy)
    for (int sx = sx0; sx < sx1; ++sx) {
        float4 p = src[sy*srcStride + sx];
        r += p.x; g += p.y; b += p.z;
        ++n;
    }
    float inv = 1.0f / (float)n;
    dst[dy*dstW + dx] = (float4)(r * inv, g * inv, b * inv, 1.0f);
}

// Final stage: clamp each channel to [0, 1], apply the sRGB OETF, round to
// 8-bit, pack into 0xFFRRGGBB (QImage::Format_RGB32 byte order).
__kernel void pack_linear_to_srgb_rgb32(
    __global const float4* src, __global uint* dst, int w, int h)
{
    int x = get_global_id(0), y = get_global_id(1);
    if (x >= w || y >= h) return;
    float4 c = src[y*w + x];
    float r = linear_to_srgb(c.x);
    float g = linear_to_srgb(c.y);
    float b = linear_to_srgb(c.z);
    uint ri = (uint)(clamp(r, 0.0f, 1.0f) * 255.0f + 0.5f);
    uint gi = (uint)(clamp(g, 0.0f, 1.0f) * 255.0f + 0.5f);
    uint bi = (uint)(clamp(b, 0.0f, 1.0f) * 255.0f + 0.5f);
    dst[y*w + x] = 0xFF000000u | (ri << 16) | (gi << 8) | bi;
}

__kernel void local_linear_gradient_exposure(
    __global float4* pixels, int w, int h, int srcW, int srcH,
    float cropX0, float cropY0, float cropX1, float cropY1,
    float centerX, float centerY, float directionX, float directionY,
    float featherHalfWidth, float exposureEv, int inverted)
{
    int x = get_global_id(0), y = get_global_id(1);
    if (x >= w || y >= h) return;
    float sourceX = cropX0 + ((float)x + 0.5f) * (cropX1 - cropX0) / (float)w;
    float sourceY = cropY0 + ((float)y + 0.5f) * (cropY1 - cropY0) / (float)h;
    float nx = sourceX / (float)srcW;
    float ny = sourceY / (float)srcH;
    float projection = (nx - centerX) * directionX + (ny - centerY) * directionY;
    float weight = featherHalfWidth == 0.0f
        ? (projection >= 0.0f ? 1.0f : 0.0f)
        : clamp(0.5f + projection / (2.0f * featherHalfWidth), 0.0f, 1.0f);
    if (inverted) weight = 1.0f - weight;
    float multiplier = exp2(exposureEv * weight);
    int i = y*w + x;
    float4 p = pixels[i];
    pixels[i] = (float4)(p.x * multiplier, p.y * multiplier, p.z * multiplier, p.w);
}

__kernel void blend_linear_gradient(
    __global float4* base, __global const float4* effected,
    int w, int h, int srcW, int srcH,
    float cropX0, float cropY0, float cropX1, float cropY1,
    float centerX, float centerY, float directionX, float directionY,
    float featherHalfWidth, int inverted)
{
    int x = get_global_id(0), y = get_global_id(1);
    if (x >= w || y >= h) return;
    float sourceX = cropX0 + ((float)x + 0.5f) * (cropX1 - cropX0) / (float)w;
    float sourceY = cropY0 + ((float)y + 0.5f) * (cropY1 - cropY0) / (float)h;
    float projection = (sourceX / (float)srcW - centerX) * directionX
                     + (sourceY / (float)srcH - centerY) * directionY;
    float weight = featherHalfWidth == 0.0f
        ? (projection >= 0.0f ? 1.0f : 0.0f)
        : clamp(0.5f + projection / (2.0f * featherHalfWidth), 0.0f, 1.0f);
    if (inverted) weight = 1.0f - weight;
    int i = y*w + x;
    base[i] = mix(base[i], effected[i], weight);
}
)CL";

// ── run ──────────────────────────────────────────────────────────────────────

GpuPipelineResult GpuPipeline::run(const QImage &image, const QVector<GpuPipelineCall> &calls,
                                   const ViewportRequest &viewport, RunMode mode,
                                   const QVector<LocalAdjustment> &localAdjustments) {
    std::lock_guard<std::mutex> lock(m_mutex);

    const int rev = GpuDeviceRegistry::instance().revision();
    if (!m_available || m_revision != rev) {
        m_available    = false;
        m_lastImageKey = 0;
        m_initializedEffects.clear();
        m_previewW       = 0;
        m_previewH       = 0;
        m_processedValid = false;
        m_processedBytes = 0;
        m_processedCalls.clear();
        m_processedLocalAdjustments.clear();
        if (!initContext()) return {}; // GCOVR_EXCL_LINE
        m_revision = rev;
    }

    // Lazily compile kernels for any effect not yet seen in this context.
    for (const auto &call : calls) {
        IGpuEffect *g = call.gpu;
        if (m_initializedEffects.find(g) == m_initializedEffects.end()) {
            if (!g->initGpuKernels(m_context, m_device)) {
                qWarning() << "[GpuPipeline] initGpuKernels failed for" << call.effect->getName();
                return {};
            }
            m_initializedEffects.insert(g);
        }
    }

    // cacheKey() identifies the QImage data and changes when its pixels detach
    // or are modified. A raw bits address can be recycled for an unrelated
    // same-sized photo, which would otherwise leave stale pixels in m_srcBuf.
    if (image.cacheKey() != m_lastImageKey || image.width() != m_width || image.height() != m_height)
        uploadImageLocked(image);

    if (!m_available) return {};

    try {
        // Compute preview dimensions from viewport.
        const int previewW = viewport.displaySize.isValid() ? viewport.displaySize.width() : m_width;
        const int previewH = viewport.displaySize.isValid() ? viewport.displaySize.height() : m_height;

        // Compute the visible crop region in source image pixels.
        // Mirrors the pan/zoom math in ViewportWidget exactly.  Letterbox
        // padding shows up here as cropX0<0 / cropX1>srcW (the visible region
        // extends past the image into empty viewport space).
        const float W = static_cast<float>(m_width), H = static_cast<float>(m_height);
        const float Vw = static_cast<float>(previewW), Vh = static_cast<float>(previewH);
        const float fitScale     = std::min(Vw / W, Vh / H);
        const float displayScale = fitScale * viewport.zoom;
        const float regionW      = Vw / displayScale;
        const float regionH      = Vh / displayScale;
        const float cropX0       = static_cast<float>(viewport.center.x()) * W - regionW * 0.5f;
        const float cropY0       = static_cast<float>(viewport.center.y()) * H - regionH * 0.5f;

        // Clip the crop region to the actual image bounds.  Effects only run
        // on real image pixels; the viewport widget renders the result inside
        // a sub-rect of the viewport, leaving the surrounding letterbox to
        // the GL clear colour.  This is the root-cause fix for the original
        // bug where additive effects (brightness offset, contrast midpoint,
        // colour balance) painted onto the black letterbox pixels.
        const float clipX0 = std::max(0.0f, cropX0);
        const float clipY0 = std::max(0.0f, cropY0);
        const float clipX1 = std::min(W, cropX0 + regionW);
        const float clipY1 = std::min(H, cropY0 + regionH);
        if (clipX0 >= clipX1 || clipY0 >= clipY1) return {};

        // Preview-pixel range that covers the clipped source region.  Round
        // to nearest pixel so the boundary lands consistently across runs.
        const int imgX0 = static_cast<int>(std::lround((clipX0 - cropX0) / regionW * Vw));
        const int imgY0 = static_cast<int>(std::lround((clipY0 - cropY0) / regionH * Vh));
        const int imgX1 = static_cast<int>(std::lround((clipX1 - cropX0) / regionW * Vw));
        const int imgY1 = static_cast<int>(std::lround((clipY1 - cropY0) / regionH * Vh));
        const int imgW  = imgX1 - imgX0;
        const int imgH  = imgY1 - imgY0;
        if (imgW <= 0 || imgH <= 0) return {}; // GCOVR_EXCL_LINE

        // Reallocate work/aux/packed buffers when the visible region size changes.
        if (m_previewW != imgW || m_previewH != imgH) {
            const size_t f4Bytes     = static_cast<size_t>(imgW) * static_cast<size_t>(imgH) * sizeof(cl_float4);
            const size_t packedBytes = static_cast<size_t>(imgW) * static_cast<size_t>(imgH) * sizeof(cl_uint);
            m_workBuf                = cl::Buffer(m_context, CL_MEM_READ_WRITE, f4Bytes);
            m_auxBuf                 = cl::Buffer(m_context, CL_MEM_READ_WRITE, f4Bytes);
            m_packedBuf              = cl::Buffer(m_context, CL_MEM_READ_WRITE, packedBytes);
            m_previewW               = imgW;
            m_previewH               = imgH;
        }

        const QPoint offset(imgX0, imgY0);

        // ── PanZoom fast path ─────────────────────────────────────────────────
        // If the cache is valid the visible preview can be produced with a
        // single float4→float4 downsample plus pack+readback.  No effect work.
        if (mode == RunMode::PanZoom && m_processedValid && processedCacheMatches(calls, localAdjustments)) {
            cl::Kernel &ds = m_downsampleKernelFloat4;
            ds.setArg(0, m_processedBuf);
            ds.setArg(1, m_workBuf);
            ds.setArg(2, m_width);
            ds.setArg(3, m_height);
            ds.setArg(4, m_width); // m_processedBuf is tightly packed
            ds.setArg(5, imgW);
            ds.setArg(6, imgH);
            ds.setArg(7, clipX0);
            ds.setArg(8, clipY0);
            ds.setArg(9, clipX1);
            ds.setArg(10, clipY1);
            m_queue.enqueueNDRangeKernel(ds, cl::NullRange,
                                         cl::NDRange(static_cast<size_t>(imgW), static_cast<size_t>(imgH)));
            return {packAndReadbackLocked(m_workBuf, imgW, imgH), offset};
        }

        // ── Commit path ───────────────────────────────────────────────────────
        // Rebuild the full-res cache: decode srcBuf → processedBuf, then run
        // effects in-place on processedBuf.  Finally downsample the cache and
        // pack+readback for display.
        if (mode == RunMode::Commit) {
            if (!decodeFullResLocked()) return {}; // GCOVR_EXCL_LINE — decodeFullResLocked can only fail via cl::Error

            // Effects at full resolution: pixel radii are in source pixels.
            for (const auto &call : calls) {
                if (!call.enabled) continue;
                IGpuEffect             *g      = call.gpu;
                QMap<QString, QVariant> params = call.params;
                params.insert("_srcPixelsPerPreviewPixel", 1.0);
                params.insert("_cropX0", 0.0);
                params.insert("_cropY0", 0.0);
                params.insert("_srcW", m_width);
                params.insert("_srcH", m_height);
                if (!g->enqueueGpu(m_queue, m_processedBuf, m_fullAuxBuf, m_width, m_height, params)) {
                    qWarning() << "[GpuPipeline]" << call.effect->getName()
                               << "enqueueGpu() failed — aborting pipeline";
                    return {};
                }
            }
            enqueueLocalAdjustmentsLocked(m_processedBuf, m_fullAuxBuf, m_width, m_height, 0.0f, 0.0f,
                                          static_cast<float>(m_width), static_cast<float>(m_height), calls,
                                          localAdjustments);
            m_processedValid            = true;
            m_processedCalls            = calls;
            m_processedLocalAdjustments = localAdjustments;

            // Downsample cache → workBuf at the visible-region dimensions.
            cl::Kernel &ds = m_downsampleKernelFloat4;
            ds.setArg(0, m_processedBuf);
            ds.setArg(1, m_workBuf);
            ds.setArg(2, m_width);
            ds.setArg(3, m_height);
            ds.setArg(4, m_width);
            ds.setArg(5, imgW);
            ds.setArg(6, imgH);
            ds.setArg(7, clipX0);
            ds.setArg(8, clipY0);
            ds.setArg(9, clipX1);
            ds.setArg(10, clipY1);
            m_queue.enqueueNDRangeKernel(ds, cl::NullRange,
                                         cl::NDRange(static_cast<size_t>(imgW), static_cast<size_t>(imgH)));

            return {packAndReadbackLocked(m_workBuf, imgW, imgH), offset};
        }

        // ── LiveDrag / PanZoom fallback ───────────────────────────────────────
        // Preview-sized pipeline: decode+downsample srcBuf → workBuf, run
        // effects on the preview buffer with radius scaling so perceptual
        // strength stays constant across zoom levels.  Invalidates the cache
        // since a new drag-state overrides the last committed frame.
        m_processedValid = false;
        m_processedCalls.clear();
        m_processedLocalAdjustments.clear();

        cl::Kernel *dsKernel = nullptr;
        if (m_is16bit) dsKernel = m_inputIsLinear ? &m_downsampleKernel16Linear : &m_downsampleKernel16Srgb;
        else dsKernel = &m_downsampleKernel8Srgb;

        dsKernel->setArg(0, m_srcBuf);
        dsKernel->setArg(1, m_workBuf);
        dsKernel->setArg(2, m_width);
        dsKernel->setArg(3, m_height);
        dsKernel->setArg(4, m_stride);
        dsKernel->setArg(5, imgW);
        dsKernel->setArg(6, imgH);
        dsKernel->setArg(7, clipX0);
        dsKernel->setArg(8, clipY0);
        dsKernel->setArg(9, clipX1);
        dsKernel->setArg(10, clipY1);
        m_queue.enqueueNDRangeKernel(*dsKernel, cl::NullRange,
                                     cl::NDRange(static_cast<size_t>(imgW), static_cast<size_t>(imgH)));
        const float srcPixelsPerPreviewPixel = (clipX1 - clipX0) / static_cast<float>(imgW);
        for (const auto &call : calls) {
            if (!call.enabled) continue;
            IGpuEffect             *g            = call.gpu;
            QMap<QString, QVariant> scaledParams = call.params;
            scaledParams.insert("_srcPixelsPerPreviewPixel", static_cast<double>(srcPixelsPerPreviewPixel));
            scaledParams.insert("_cropX0", static_cast<double>(clipX0));
            scaledParams.insert("_cropY0", static_cast<double>(clipY0));
            scaledParams.insert("_srcW", m_width);
            scaledParams.insert("_srcH", m_height);
            if (!g->enqueueGpu(m_queue, m_workBuf, m_auxBuf, imgW, imgH, scaledParams)) {
                // GCOVR_EXCL_START — every shipped IGpuEffect returns true
                // unless it threw, in which case the surrounding catch fires.
                qWarning() << "[GpuPipeline]" << call.effect->getName() << "enqueueGpu() failed — aborting pipeline";
                return {};
                // GCOVR_EXCL_STOP
            }
        }
        enqueueLocalAdjustmentsLocked(m_workBuf, m_auxBuf, imgW, imgH, clipX0, clipY0, clipX1, clipY1, calls,
                                      localAdjustments);
        return {packAndReadbackLocked(m_workBuf, imgW, imgH), offset};
    }
    // GCOVR_EXCL_START
    catch (const cl::Error &e) {
        qWarning() << "[GpuPipeline] run() failed:" << e.what() << "(err" << e.err() << ")";
        m_available = false;
        return {};
    }
    // GCOVR_EXCL_STOP
}

static bool sameLocalAdjustment(const LocalAdjustment &a, const LocalAdjustment &b) {
    return a.id == b.id && a.enabled == b.enabled && a.exposureEv == b.exposureEv && a.effects == b.effects &&
           a.mask.center() == b.mask.center() && a.mask.direction() == b.mask.direction() &&
           a.mask.featherHalfWidth() == b.mask.featherHalfWidth() && a.mask.isInverted() == b.mask.isInverted();
}

bool GpuPipeline::processedCacheMatches(const QVector<GpuPipelineCall> &calls,
                                        const QVector<LocalAdjustment> &localAdjustments) const {
    if (calls.size() != m_processedCalls.size()) return false;
    for (qsizetype i = 0; i < calls.size(); ++i) {
        if (calls[i].gpu != m_processedCalls[i].gpu || calls[i].params != m_processedCalls[i].params ||
            calls[i].enabled != m_processedCalls[i].enabled)
            return false;
    }
    if (localAdjustments.size() != m_processedLocalAdjustments.size()) return false;
    for (qsizetype i = 0; i < localAdjustments.size(); ++i)
        if (!sameLocalAdjustment(localAdjustments[i], m_processedLocalAdjustments[i])) return false;
    return true;
}

void GpuPipeline::enqueueLocalAdjustmentsLocked(cl::Buffer &buffer, cl::Buffer &aux, int width, int height,
                                                float cropX0, float cropY0, float cropX1, float cropY1,
                                                const QVector<GpuPipelineCall> &calls,
                                                const QVector<LocalAdjustment> &localAdjustments) {
    for (const LocalAdjustment &adjustment : localAdjustments) {
        if (!adjustment.enabled) continue;
        if (!adjustment.effects.isEmpty()) {
            const size_t bytes = static_cast<size_t>(width) * static_cast<size_t>(height) * sizeof(cl_float4);
            m_queue.enqueueCopyBuffer(buffer, aux, 0, 0, bytes);
            bool applied = false;
            for (const GpuPipelineCall &call : calls) {
                const auto local = adjustment.effects.constFind(call.effect->getId());
                if (local == adjustment.effects.constEnd() || !local.value().enabled) continue;
                if (call.effect->getId() != QStringLiteral("exposure") &&
                    call.effect->getId() != QStringLiteral("saturation_vibrancy") &&
                    call.effect->getId() != QStringLiteral("grayscale"))
                    continue;
                QMap<QString, QVariant> parameters = local.value().parameters;
                parameters.insert("_srcPixelsPerPreviewPixel", 1.0);
                if (!call.gpu->enqueueGpu(m_queue, aux, buffer, width, height, parameters)) continue;
                applied = true;
            }
            if (!applied) continue;
            m_localBlendKernel.setArg(0, buffer);
            m_localBlendKernel.setArg(1, aux);
            m_localBlendKernel.setArg(2, width);
            m_localBlendKernel.setArg(3, height);
            m_localBlendKernel.setArg(4, m_width);
            m_localBlendKernel.setArg(5, m_height);
            m_localBlendKernel.setArg(6, cropX0);
            m_localBlendKernel.setArg(7, cropY0);
            m_localBlendKernel.setArg(8, cropX1);
            m_localBlendKernel.setArg(9, cropY1);
            m_localBlendKernel.setArg(10, static_cast<float>(adjustment.mask.center().x()));
            m_localBlendKernel.setArg(11, static_cast<float>(adjustment.mask.center().y()));
            m_localBlendKernel.setArg(12, static_cast<float>(adjustment.mask.direction().x()));
            m_localBlendKernel.setArg(13, static_cast<float>(adjustment.mask.direction().y()));
            m_localBlendKernel.setArg(14, static_cast<float>(adjustment.mask.featherHalfWidth()));
            m_localBlendKernel.setArg(15, adjustment.mask.isInverted() ? 1 : 0);
            m_queue.enqueueNDRangeKernel(m_localBlendKernel, cl::NullRange,
                                         cl::NDRange(static_cast<size_t>(width), static_cast<size_t>(height)));
            continue;
        }
        if (adjustment.exposureEv == 0.0) continue;
        m_localExposureKernel.setArg(0, buffer);
        m_localExposureKernel.setArg(1, width);
        m_localExposureKernel.setArg(2, height);
        m_localExposureKernel.setArg(3, m_width);
        m_localExposureKernel.setArg(4, m_height);
        m_localExposureKernel.setArg(5, cropX0);
        m_localExposureKernel.setArg(6, cropY0);
        m_localExposureKernel.setArg(7, cropX1);
        m_localExposureKernel.setArg(8, cropY1);
        m_localExposureKernel.setArg(9, static_cast<float>(adjustment.mask.center().x()));
        m_localExposureKernel.setArg(10, static_cast<float>(adjustment.mask.center().y()));
        m_localExposureKernel.setArg(11, static_cast<float>(adjustment.mask.direction().x()));
        m_localExposureKernel.setArg(12, static_cast<float>(adjustment.mask.direction().y()));
        m_localExposureKernel.setArg(13, static_cast<float>(adjustment.mask.featherHalfWidth()));
        m_localExposureKernel.setArg(14, static_cast<float>(adjustment.exposureEv));
        m_localExposureKernel.setArg(15, adjustment.mask.isInverted() ? 1 : 0);
        m_queue.enqueueNDRangeKernel(m_localExposureKernel, cl::NullRange,
                                     cl::NDRange(static_cast<size_t>(width), static_cast<size_t>(height)));
    }
}

bool GpuPipeline::decodeFullResLocked() {
    const size_t bytes = static_cast<size_t>(m_width) * static_cast<size_t>(m_height) * sizeof(cl_float4);
    if (m_processedBytes != bytes) {
        m_processedBuf   = cl::Buffer(m_context, CL_MEM_READ_WRITE, bytes);
        m_fullAuxBuf     = cl::Buffer(m_context, CL_MEM_READ_WRITE, bytes);
        m_processedBytes = bytes;
        m_processedValid = false;
    }

    cl::Kernel *k = nullptr;
    if (m_is16bit) k = m_inputIsLinear ? &m_decodeKernel16Linear : &m_decodeKernel16Srgb;
    else k = &m_decodeKernel8Srgb;

    k->setArg(0, m_srcBuf);
    k->setArg(1, m_processedBuf);
    k->setArg(2, m_width);
    k->setArg(3, m_height);
    k->setArg(4, m_stride);
    m_queue.enqueueNDRangeKernel(*k, cl::NullRange,
                                 cl::NDRange(static_cast<size_t>(m_width), static_cast<size_t>(m_height)));
    return true;
}

QImage GpuPipeline::packAndReadbackLocked(cl::Buffer &src, int w, int h) {
    m_packKernel.setArg(0, src);
    m_packKernel.setArg(1, m_packedBuf);
    m_packKernel.setArg(2, w);
    m_packKernel.setArg(3, h);
    m_queue.enqueueNDRangeKernel(m_packKernel, cl::NullRange,
                                 cl::NDRange(static_cast<size_t>(w), static_cast<size_t>(h)));

    QImage result(w, h, QImage::Format_RGB32);
    m_queue.enqueueReadBuffer(m_packedBuf, CL_TRUE, 0,
                              static_cast<size_t>(w) * static_cast<size_t>(h) * sizeof(cl_uint), result.bits());
    return result;
}

// ── initContext ───────────────────────────────────────────────────────────────

bool GpuPipeline::initContext() {
    cl::Device   device;
    cl::Platform platform;
    // GCOVR_EXCL_START
    if (!GpuDeviceRegistryOCL::getSelectedDevice(device, platform)) {
        qWarning() << "[GpuPipeline] no OpenCL device available";
        return false;
    }
    // GCOVR_EXCL_STOP

    try {
        m_context   = cl::Context(device);
        m_queue     = cl::CommandQueue(m_context, device);
        m_device    = device;
        m_available = true;
        qDebug() << "[GpuPipeline] context ready on:" << QString::fromStdString(device.getInfo<CL_DEVICE_NAME>());

        // GCOVR_EXCL_START
        if (!initDownsampleKernels()) {
            m_available = false;
            return false;
        }
        // GCOVR_EXCL_STOP

        return true;
    }
    // GCOVR_EXCL_START
    catch (const cl::Error &e) {
        qWarning() << "[GpuPipeline] initContext failed:" << e.what() << "(err" << e.err() << ")";
        return false;
    }
    // GCOVR_EXCL_STOP
}

bool GpuPipeline::initDownsampleKernels() {
    cl::Program prog(m_context, PIPELINE_KERNEL_SOURCE);
    try {
        prog.build({m_device});
        m_downsampleKernel8Srgb    = cl::Kernel(prog, "preview_downsample_8bit_srgb_to_linear");
        m_downsampleKernel16Srgb   = cl::Kernel(prog, "preview_downsample_16bit_srgb_to_linear");
        m_downsampleKernel16Linear = cl::Kernel(prog, "preview_downsample_16bit_linear");
        m_downsampleKernelFloat4   = cl::Kernel(prog, "preview_downsample_float4_linear");
        m_decodeKernel8Srgb        = cl::Kernel(prog, "fullres_decode_8bit_srgb_to_linear");
        m_decodeKernel16Srgb       = cl::Kernel(prog, "fullres_decode_16bit_srgb_to_linear");
        m_decodeKernel16Linear     = cl::Kernel(prog, "fullres_decode_16bit_linear");
        m_packKernel               = cl::Kernel(prog, "pack_linear_to_srgb_rgb32");
        m_localExposureKernel      = cl::Kernel(prog, "local_linear_gradient_exposure");
        m_localBlendKernel         = cl::Kernel(prog, "blend_linear_gradient");
        return true;
    }
    // GCOVR_EXCL_START
    catch (const cl::Error &e) {
        qWarning() << "[GpuPipeline] initDownsampleKernels failed:" << e.what() << "(err" << e.err() << ")";
        try {
            std::string log = prog.getBuildInfo<CL_PROGRAM_BUILD_LOG>(m_device);
            qWarning() << "Build log:" << QString::fromStdString(log);
        } catch (...) {}
        return false;
    }
    // GCOVR_EXCL_STOP
}

void GpuPipeline::uploadImageLocked(const QImage &image) {
    const bool is16bit = (image.format() == QImage::Format_RGBX64);
    const int  bpp     = is16bit ? 8 : 4;

    QImage src = is16bit ? image : image.convertToFormat(QImage::Format_RGB32);

    m_width    = src.width();
    m_height   = src.height();
    m_stride   = static_cast<int>(src.bytesPerLine() / bpp);
    m_bufBytes = static_cast<size_t>(src.bytesPerLine()) * static_cast<size_t>(m_height);
    m_is16bit  = is16bit;
    // RawLoader tags linear 16-bit inputs; any other QImage (JPEG/PNG/convertTo)
    // is sRGB-gamma encoded.  Read tag from the original image, not the converted
    // copy, to survive the convertToFormat round-trip.
    m_inputIsLinear = (image.text("color_space") == QStringLiteral("linear"));

    m_previewW       = 0;
    m_previewH       = 0;
    m_processedValid = false; // new image content invalidates the cache
    m_processedCalls.clear();

    try {
        m_srcBuf       = cl::Buffer(m_context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, m_bufBytes, src.bits());
        m_lastImageKey = image.cacheKey();
    }
    // GCOVR_EXCL_START
    catch (const cl::Error &e) {
        qWarning() << "[GpuPipeline] upload failed:" << e.what() << "(err" << e.err() << ")";
        m_available = false;
    }
    // GCOVR_EXCL_STOP
}
