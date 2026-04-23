#include "GpuPipeline.h"
#include "IGpuEffect.h"
#include "GpuDeviceRegistry.h"
#include "GpuDeviceRegistryOCL.h"
#include "color_kernels.h"
#include <QDebug>
#include <QElapsedTimer>
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
static const char* PIPELINE_KERNEL_SOURCE = COLOR_KERNELS_SRC R"CL(

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
    *sx1 = min(srcW, (int)sx1f + 1);
    *sy1 = min(srcH, (int)sy1f + 1);
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
)CL";

// ── run ──────────────────────────────────────────────────────────────────────

QImage GpuPipeline::run(const QImage& image, const QVector<GpuPipelineCall>& calls,
                        const ViewportRequest& viewport, RunMode mode) {
    std::lock_guard<std::mutex> lock(m_mutex);

    const int rev = GpuDeviceRegistry::instance().revision();
    if (!m_available || m_revision != rev) {
        m_available = false;
        m_lastBits  = nullptr;
        m_initializedEffects.clear();
        m_previewW = 0;
        m_previewH = 0;
        m_processedValid = false;
        m_processedBytes = 0;
        if (!initContext())
            return {}; // GCOVR_EXCL_LINE
        m_revision = rev;
    }

    // Lazily compile kernels for any effect not yet seen in this context.
    for (const auto& call : calls) {
        auto* g = dynamic_cast<IGpuEffect*>(call.effect);
        if (!g) {
            qWarning() << "[GpuPipeline]" << call.effect->getName()
                       << "does not implement IGpuEffect — aborting pipeline";
            return {};
        }
        if (m_initializedEffects.find(g) == m_initializedEffects.end()) {
            if (!g->initGpuKernels(m_context, m_device)) {
                qWarning() << "[GpuPipeline] initGpuKernels failed for"
                           << call.effect->getName();
                return {};
            }
            m_initializedEffects.insert(g);
        }
    }

    if (image.constBits() != m_lastBits)
        uploadImageLocked(image);

    if (!m_available) return {};

    QElapsedTimer t;
    t.start();

    try {
        // Compute preview dimensions from viewport.
        const int previewW = viewport.displaySize.isValid() ? viewport.displaySize.width()  : m_width;
        const int previewH = viewport.displaySize.isValid() ? viewport.displaySize.height() : m_height;

        // Compute the visible crop region in source image pixels.
        // Mirrors the pan/zoom math in ViewportWidget exactly.
        const float W  = m_width,  H  = m_height;
        const float Vw = previewW, Vh = previewH;
        const float fitScale     = std::min(Vw / W, Vh / H);
        const float displayScale = fitScale * viewport.zoom;
        const float regionW = Vw / displayScale;
        const float regionH = Vh / displayScale;
        const float cropX0 = (float)viewport.center.x() * W - regionW * 0.5f;
        const float cropY0 = (float)viewport.center.y() * H - regionH * 0.5f;
        const float cropX1 = cropX0 + regionW;
        const float cropY1 = cropY0 + regionH;

        // Reallocate preview-sized work/aux/packed buffers when dimensions change.
        if (m_previewW != previewW || m_previewH != previewH) {
            const size_t f4Bytes     = static_cast<size_t>(previewW) * previewH * sizeof(cl_float4);
            const size_t packedBytes = static_cast<size_t>(previewW) * previewH * sizeof(cl_uint);
            m_workBuf   = cl::Buffer(m_context, CL_MEM_READ_WRITE, f4Bytes);
            m_auxBuf    = cl::Buffer(m_context, CL_MEM_READ_WRITE, f4Bytes);
            m_packedBuf = cl::Buffer(m_context, CL_MEM_READ_WRITE, packedBytes);
            m_previewW  = previewW;
            m_previewH  = previewH;
        }

        // ── PanZoom fast path ─────────────────────────────────────────────────
        // If the cache is valid the visible preview can be produced with a
        // single float4→float4 downsample plus pack+readback.  No effect work.
        if (mode == RunMode::PanZoom && m_processedValid) {
            cl::Kernel& ds = m_downsampleKernelFloat4;
            ds.setArg(0, m_processedBuf);
            ds.setArg(1, m_workBuf);
            ds.setArg(2, m_width);
            ds.setArg(3, m_height);
            ds.setArg(4, m_width);   // m_processedBuf is tightly packed
            ds.setArg(5, previewW);
            ds.setArg(6, previewH);
            ds.setArg(7, cropX0);
            ds.setArg(8, cropY0);
            ds.setArg(9, cropX1);
            ds.setArg(10, cropY1);
            m_queue.enqueueNDRangeKernel(ds, cl::NullRange,
                                         cl::NDRange(previewW, previewH));
            const qint64 t1 = t.nsecsElapsed();
            QImage result = packAndReadbackLocked(m_workBuf, previewW, previewH);
            const qint64 t3 = t.nsecsElapsed();
            qDebug() << "[GpuPipeline] PanZoom (cache hit)"
                     << " downsample:" << (t1 + 500) / 1000 << "µs"
                     << " pack+read:"  << (t3 - t1 + 500) / 1000 << "µs"
                     << " total:"      << (t3 + 500) / 1000 << "µs"
                     << " preview:"    << previewW << "x" << previewH;
            return result;
        }

        // ── Commit path ───────────────────────────────────────────────────────
        // Rebuild the full-res cache: decode srcBuf → processedBuf, then run
        // effects in-place on processedBuf.  Finally downsample the cache and
        // pack+readback for display.
        if (mode == RunMode::Commit) {
            if (!decodeFullResLocked())
                return {}; // GCOVR_EXCL_LINE — decodeFullResLocked can only fail via cl::Error
            const qint64 tDecode = t.nsecsElapsed();

            // Effects at full resolution: pixel radii are in source pixels.
            for (const auto& call : calls) {
                auto* g = dynamic_cast<IGpuEffect*>(call.effect);
                QMap<QString, QVariant> params = call.params;
                params.insert("_srcPixelsPerPreviewPixel", 1.0);
                params.insert("_cropX0", 0.0);
                params.insert("_cropY0", 0.0);
                if (!g->enqueueGpu(m_queue, m_processedBuf, m_fullAuxBuf,
                                   m_width, m_height, params)) {
                    qWarning() << "[GpuPipeline]" << call.effect->getName()
                               << "enqueueGpu() failed — aborting pipeline";
                    return {};
                }
            }
            m_queue.finish();
            const qint64 tEffects = t.nsecsElapsed();
            m_processedValid = true;

            // Downsample cache → workBuf at the requested preview dimensions.
            cl::Kernel& ds = m_downsampleKernelFloat4;
            ds.setArg(0, m_processedBuf);
            ds.setArg(1, m_workBuf);
            ds.setArg(2, m_width);
            ds.setArg(3, m_height);
            ds.setArg(4, m_width);
            ds.setArg(5, previewW);
            ds.setArg(6, previewH);
            ds.setArg(7, cropX0);
            ds.setArg(8, cropY0);
            ds.setArg(9, cropX1);
            ds.setArg(10, cropY1);
            m_queue.enqueueNDRangeKernel(ds, cl::NullRange,
                                         cl::NDRange(previewW, previewH));
            const qint64 tDs = t.nsecsElapsed();

            QImage result = packAndReadbackLocked(m_workBuf, previewW, previewH);
            const qint64 tTotal = t.nsecsElapsed();
            qDebug() << "[GpuPipeline] Commit"
                     << " decode:"     << (tDecode + 500) / 1000 << "µs"
                     << " effects:"    << (tEffects - tDecode + 500) / 1000 << "µs"
                     << " downsample:" << (tDs - tEffects + 500) / 1000 << "µs"
                     << " pack+read:"  << (tTotal - tDs + 500) / 1000 << "µs"
                     << " total:"      << (tTotal + 500) / 1000 << "µs"
                     << " full:"       << m_width << "x" << m_height
                     << " preview:"    << previewW << "x" << previewH;
            return result;
        }

        // ── LiveDrag / PanZoom fallback ───────────────────────────────────────
        // Legacy preview-sized pipeline: decode+downsample srcBuf → workBuf,
        // run effects on the preview buffer with radius scaling so perceptual
        // strength stays constant across zoom levels.  Invalidates the cache
        // since a new drag-state overrides the last committed frame.
        m_processedValid = false;

        cl::Kernel* dsKernel = nullptr;
        if (m_is16bit)
            dsKernel = m_inputIsLinear ? &m_downsampleKernel16Linear
                                       : &m_downsampleKernel16Srgb;
        else
            dsKernel = &m_downsampleKernel8Srgb;

        dsKernel->setArg(0, m_srcBuf);
        dsKernel->setArg(1, m_workBuf);
        dsKernel->setArg(2, m_width);
        dsKernel->setArg(3, m_height);
        dsKernel->setArg(4, m_stride);
        dsKernel->setArg(5, previewW);
        dsKernel->setArg(6, previewH);
        dsKernel->setArg(7, cropX0);
        dsKernel->setArg(8, cropY0);
        dsKernel->setArg(9, cropX1);
        dsKernel->setArg(10, cropY1);
        m_queue.enqueueNDRangeKernel(*dsKernel, cl::NullRange,
                                     cl::NDRange(previewW, previewH));
        m_queue.finish();
        const qint64 t1 = t.nsecsElapsed();

        const float srcPixelsPerPreviewPixel = regionW / static_cast<float>(previewW);
        for (const auto& call : calls) {
            auto* g = dynamic_cast<IGpuEffect*>(call.effect);
            QMap<QString, QVariant> scaledParams = call.params;
            scaledParams.insert("_srcPixelsPerPreviewPixel",
                                static_cast<double>(srcPixelsPerPreviewPixel));
            scaledParams.insert("_cropX0", static_cast<double>(cropX0));
            scaledParams.insert("_cropY0", static_cast<double>(cropY0));
            if (!g->enqueueGpu(m_queue, m_workBuf, m_auxBuf,
                               previewW, previewH, scaledParams)) {
                qWarning() << "[GpuPipeline]" << call.effect->getName()
                           << "enqueueGpu() failed — aborting pipeline";
                return {};
            }
        }
        m_queue.finish();
        const qint64 t2 = t.nsecsElapsed();

        QImage result = packAndReadbackLocked(m_workBuf, previewW, previewH);
        const qint64 t3 = t.nsecsElapsed();

        qDebug() << "[GpuPipeline] LiveDrag"
                 << " downsample:" << (t1 + 500) / 1000 << "µs"
                 << " effects:"    << (t2 - t1 + 500) / 1000 << "µs"
                 << " pack+read:"  << (t3 - t2 + 500) / 1000 << "µs"
                 << " total:"      << (t3 + 500) / 1000 << "µs"
                 << " preview:"    << previewW << "x" << previewH
                 << (m_inputIsLinear ? "(linear src)" : "(sRGB src)");
        return result;

    }
    // GCOVR_EXCL_START
    catch (const cl::Error& e) {
        qWarning() << "[GpuPipeline] run() failed:" << e.what() << "(err" << e.err() << ")";
        m_available = false;
        return {};
    }
    // GCOVR_EXCL_STOP
}

bool GpuPipeline::decodeFullResLocked() {
    const size_t bytes = static_cast<size_t>(m_width) * m_height * sizeof(cl_float4);
    if (m_processedBytes != bytes) {
        m_processedBuf   = cl::Buffer(m_context, CL_MEM_READ_WRITE, bytes);
        m_fullAuxBuf     = cl::Buffer(m_context, CL_MEM_READ_WRITE, bytes);
        m_processedBytes = bytes;
        m_processedValid = false;
    }

    cl::Kernel* k = nullptr;
    if (m_is16bit)
        k = m_inputIsLinear ? &m_decodeKernel16Linear : &m_decodeKernel16Srgb;
    else
        k = &m_decodeKernel8Srgb;

    k->setArg(0, m_srcBuf);
    k->setArg(1, m_processedBuf);
    k->setArg(2, m_width);
    k->setArg(3, m_height);
    k->setArg(4, m_stride);
    m_queue.enqueueNDRangeKernel(*k, cl::NullRange,
                                 cl::NDRange(m_width, m_height));
    m_queue.finish();
    return true;
}

QImage GpuPipeline::packAndReadbackLocked(cl::Buffer& src, int w, int h) {
    m_packKernel.setArg(0, src);
    m_packKernel.setArg(1, m_packedBuf);
    m_packKernel.setArg(2, w);
    m_packKernel.setArg(3, h);
    m_queue.enqueueNDRangeKernel(m_packKernel, cl::NullRange, cl::NDRange(w, h));

    QImage result(w, h, QImage::Format_RGB32);
    m_queue.enqueueReadBuffer(m_packedBuf, CL_TRUE, 0,
                              static_cast<size_t>(w) * h * sizeof(cl_uint),
                              result.bits());
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
        m_context = cl::Context(device);
        m_queue   = cl::CommandQueue(m_context, device);
        m_device  = device;
        m_available = true;
        qDebug() << "[GpuPipeline] context ready on:"
                 << QString::fromStdString(device.getInfo<CL_DEVICE_NAME>());

        // GCOVR_EXCL_START
        if (!initDownsampleKernels()) {
            m_available = false;
            return false;
        }
        // GCOVR_EXCL_STOP

        return true;
    }
    // GCOVR_EXCL_START
    catch (const cl::Error& e) {
        qWarning() << "[GpuPipeline] initContext failed:" << e.what() << "(err" << e.err() << ")";
        return false;
    }
    // GCOVR_EXCL_STOP
}

bool GpuPipeline::initDownsampleKernels() {
    try {
        cl::Program prog(m_context, PIPELINE_KERNEL_SOURCE);
        prog.build({m_device});
        m_downsampleKernel8Srgb     = cl::Kernel(prog, "preview_downsample_8bit_srgb_to_linear");
        m_downsampleKernel16Srgb    = cl::Kernel(prog, "preview_downsample_16bit_srgb_to_linear");
        m_downsampleKernel16Linear  = cl::Kernel(prog, "preview_downsample_16bit_linear");
        m_downsampleKernelFloat4    = cl::Kernel(prog, "preview_downsample_float4_linear");
        m_decodeKernel8Srgb         = cl::Kernel(prog, "fullres_decode_8bit_srgb_to_linear");
        m_decodeKernel16Srgb        = cl::Kernel(prog, "fullres_decode_16bit_srgb_to_linear");
        m_decodeKernel16Linear      = cl::Kernel(prog, "fullres_decode_16bit_linear");
        m_packKernel                = cl::Kernel(prog, "pack_linear_to_srgb_rgb32");
        return true;
    }
    // GCOVR_EXCL_START
    catch (const cl::Error& e) {
        qWarning() << "[GpuPipeline] initDownsampleKernels failed:" << e.what()
                   << "(err" << e.err() << ")";
        try {
            cl::Program prog(m_context, PIPELINE_KERNEL_SOURCE);
            std::string log = prog.getBuildInfo<CL_PROGRAM_BUILD_LOG>(m_device);
            qWarning() << "Build log:" << QString::fromStdString(log);
        } catch (...) {}
        return false;
    }
    // GCOVR_EXCL_STOP
}

void GpuPipeline::uploadImageLocked(const QImage& image) {
    const bool is16bit = (image.format() == QImage::Format_RGBX64);
    const int  bpp     = is16bit ? 8 : 4;

    QImage src = is16bit ? image : image.convertToFormat(QImage::Format_RGB32);

    m_width    = src.width();
    m_height   = src.height();
    m_stride   = src.bytesPerLine() / bpp;
    m_bufBytes = static_cast<size_t>(src.bytesPerLine()) * m_height;
    m_is16bit  = is16bit;
    // RawLoader tags linear 16-bit inputs; any other QImage (JPEG/PNG/convertTo)
    // is sRGB-gamma encoded.  Read tag from the original image, not the converted
    // copy, to survive the convertToFormat round-trip.
    m_inputIsLinear = (image.text("color_space") == QStringLiteral("linear"));

    m_previewW = 0;
    m_previewH = 0;
    m_processedValid = false;  // new image content invalidates the cache

    QElapsedTimer t;
    t.start();

    try {
        m_srcBuf   = cl::Buffer(m_context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                                m_bufBytes, src.bits());
        m_lastBits = image.constBits();

        qDebug() << "[GpuPipeline] uploaded" << m_width << "x" << m_height
                 << (m_is16bit ? "16-bit" : "8-bit")
                 << (m_inputIsLinear ? "linear" : "sRGB")
                 << "bufBytes" << m_bufBytes
                 << "in" << (t.nsecsElapsed() + 500) / 1000 << "µs";

    }
    // GCOVR_EXCL_START
    catch (const cl::Error& e) {
        qWarning() << "[GpuPipeline] upload failed:" << e.what() << "(err" << e.err() << ")";
        m_available = false;
    }
    // GCOVR_EXCL_STOP
}
