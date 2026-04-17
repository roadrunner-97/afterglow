#include "GpuPipeline.h"
#include "IGpuEffect.h"
#include "GpuDeviceRegistry.h"
#include "GpuDeviceRegistryOCL.h"
#include <QDebug>
#include <QElapsedTimer>
#include <algorithm>

// Downsample kernels output RGB32 (0xFFRRGGBB), matching QImage::Format_RGB32.
// R and B are NOT swapped — effect kernels expect RGB32 byte order.
// cropX0/Y0/X1/Y1 are the visible region in source image pixels (may extend
// outside [0..srcW/H) — those pixels are output as black for letterboxing).
static const char* DOWNSAMPLE_KERNEL_SOURCE = R"CL(
kernel void preview_downsample_8bit_rgb32(
    global const uint* src, global uint* dst,
    int srcW, int srcH, int srcStride, int dstW, int dstH,
    float cropX0, float cropY0, float cropX1, float cropY1)
{
    int dx = get_global_id(0), dy = get_global_id(1);
    if (dx >= dstW || dy >= dstH) return;
    float rgnW = cropX1 - cropX0, rgnH = cropY1 - cropY0;
    float sx0f = cropX0 + (float)dx       * rgnW / dstW;
    float sy0f = cropY0 + (float)dy       * rgnH / dstH;
    float sx1f = cropX0 + (float)(dx + 1) * rgnW / dstW;
    float sy1f = cropY0 + (float)(dy + 1) * rgnH / dstH;
    int sx0 = max(0, (int)sx0f);
    int sy0 = max(0, (int)sy0f);
    int sx1 = min(srcW, (int)sx1f + 1);
    int sy1 = min(srcH, (int)sy1f + 1);
    if (sx0 >= sx1 || sy0 >= sy1) { dst[dy*dstW+dx] = 0xFF000000u; return; }
    float r=0,g=0,b=0; int n=0;
    for (int sy=sy0;sy<sy1;sy++) for (int sx=sx0;sx<sx1;sx++) {
        uint p = src[sy*srcStride+sx];
        r += (p>>16)&0xFF; g += (p>>8)&0xFF; b += p&0xFF; n++;
    }
    // Output RGB32: keep R/G/B order (not swapped)
    if (n) dst[dy*dstW+dx] = 0xFF000000u
        | ((uint)(r/n+.5f)<<16)
        | ((uint)(g/n+.5f)<<8)
        |  (uint)(b/n+.5f);
}

kernel void preview_downsample_16bit_rgb32(
    global const ushort* src, global uint* dst,
    int srcW, int srcH, int srcStride, int dstW, int dstH,
    float cropX0, float cropY0, float cropX1, float cropY1)
{
    int dx = get_global_id(0), dy = get_global_id(1);
    if (dx >= dstW || dy >= dstH) return;
    float rgnW = cropX1 - cropX0, rgnH = cropY1 - cropY0;
    float sx0f = cropX0 + (float)dx       * rgnW / dstW;
    float sy0f = cropY0 + (float)dy       * rgnH / dstH;
    float sx1f = cropX0 + (float)(dx + 1) * rgnW / dstW;
    float sy1f = cropY0 + (float)(dy + 1) * rgnH / dstH;
    int sx0 = max(0, (int)sx0f);
    int sy0 = max(0, (int)sy0f);
    int sx1 = min(srcW, (int)sx1f + 1);
    int sy1 = min(srcH, (int)sy1f + 1);
    if (sx0 >= sx1 || sy0 >= sy1) { dst[dy*dstW+dx] = 0xFF000000u; return; }
    float r=0,g=0,b=0; int n=0;
    for (int sy=sy0;sy<sy1;sy++) for (int sx=sx0;sx<sx1;sx++) {
        int i = (sy*srcStride+sx)*4;
        r += src[i]; g += src[i+1]; b += src[i+2]; n++;
    }
    // Output RGB32: keep R/G/B order (not swapped), scale 16-bit → 8-bit
    if (n) dst[dy*dstW+dx] = 0xFF000000u
        | ((uint)(r/n/257.f+.5f)<<16)
        | ((uint)(g/n/257.f+.5f)<<8)
        |  (uint)(b/n/257.f+.5f);
}
)CL";

// ── run ──────────────────────────────────────────────────────────────────────

QImage GpuPipeline::run(const QImage& image, const QVector<GpuPipelineCall>& calls,
                        const ViewportRequest& viewport, bool /*viewportOnly*/) {
    std::lock_guard<std::mutex> lock(m_mutex);

    const int rev = GpuDeviceRegistry::instance().revision();
    if (!m_available || m_revision != rev) {
        m_available = false;
        m_lastBits  = nullptr;
        m_initializedEffects.clear();
        m_previewW = 0;
        m_previewH = 0;
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

        // Reallocate preview-sized work/aux buffers when dimensions change.
        if (m_previewW != previewW || m_previewH != previewH) {
            const size_t previewBytes = static_cast<size_t>(previewW) * previewH * 4;
            m_workBuf  = cl::Buffer(m_context, CL_MEM_READ_WRITE, previewBytes);
            m_auxBuf   = cl::Buffer(m_context, CL_MEM_READ_WRITE, previewBytes);
            m_previewW = previewW;
            m_previewH = previewH;
        }

        // Step 1: Downsample srcBuf → workBuf (preview-sized RGB32).
        cl::Kernel& dsKernel = m_is16bit ? m_downsampleKernel16 : m_downsampleKernel8;
        dsKernel.setArg(0, m_srcBuf);
        dsKernel.setArg(1, m_workBuf);
        dsKernel.setArg(2, m_width);
        dsKernel.setArg(3, m_height);
        dsKernel.setArg(4, m_stride);
        dsKernel.setArg(5, previewW);
        dsKernel.setArg(6, previewH);
        dsKernel.setArg(7, cropX0);
        dsKernel.setArg(8, cropY0);
        dsKernel.setArg(9, cropX1);
        dsKernel.setArg(10, cropY1);
        m_queue.enqueueNDRangeKernel(dsKernel, cl::NullRange,
                                     cl::NDRange(previewW, previewH));
        m_queue.finish();
        const qint64 t1 = t.nsecsElapsed();

        // Scale factor: how many source pixels correspond to one preview pixel.
        // Effects with pixel-radius parameters (blur, unsharp, denoise) divide
        // their radii by this value so the perceptual strength stays constant
        // regardless of zoom level.
        const float srcPixelsPerPreviewPixel = regionW / static_cast<float>(previewW);

        // Step 2: Run all effects on the preview-sized RGB32 workBuf.
        for (const auto& call : calls) {
            auto* g = dynamic_cast<IGpuEffect*>(call.effect);
            QMap<QString, QVariant> scaledParams = call.params;
            scaledParams.insert("_srcPixelsPerPreviewPixel",
                                static_cast<double>(srcPixelsPerPreviewPixel));
            if (!g->enqueueGpu(m_queue, m_workBuf, m_auxBuf,
                               previewW, previewH, previewW,
                               /*is16bit=*/false, scaledParams)) {
                qWarning() << "[GpuPipeline]" << call.effect->getName()
                           << "enqueueGpu() failed — aborting pipeline";
                return {};
            }
        }
        m_queue.finish();
        const qint64 t2 = t.nsecsElapsed();

        // Step 3: Read workBuf back to CPU as RGB32.
        QImage result(previewW, previewH, QImage::Format_RGB32);
        m_queue.enqueueReadBuffer(m_workBuf, CL_TRUE, 0,
                                  static_cast<size_t>(previewW) * previewH * 4,
                                  result.bits());
        const qint64 t3 = t.nsecsElapsed();

        qDebug() << "[GpuPipeline]"
                 << "downsample:" << (t1 + 500) / 1000 << "µs"
                 << " effects:"   << (t2 - t1 + 500) / 1000 << "µs"
                 << " readback:"  << (t3 - t2 + 500) / 1000 << "µs"
                 << " total:"     << (t3 + 500) / 1000 << "µs"
                 << " preview:"   << previewW << "x" << previewH;
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
        cl::Program prog(m_context, DOWNSAMPLE_KERNEL_SOURCE);
        prog.build({m_device});
        m_downsampleKernel8  = cl::Kernel(prog, "preview_downsample_8bit_rgb32");
        m_downsampleKernel16 = cl::Kernel(prog, "preview_downsample_16bit_rgb32");
        return true;
    }
    // GCOVR_EXCL_START
    catch (const cl::Error& e) {
        qWarning() << "[GpuPipeline] initDownsampleKernels failed:" << e.what()
                   << "(err" << e.err() << ")";
        try {
            cl::Program prog(m_context, DOWNSAMPLE_KERNEL_SOURCE);
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

    m_previewW = 0;
    m_previewH = 0;

    QElapsedTimer t;
    t.start();

    try {
        m_srcBuf   = cl::Buffer(m_context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                                m_bufBytes, src.bits());
        m_lastBits = image.constBits();

        qDebug() << "[GpuPipeline] uploaded" << m_width << "x" << m_height
                 << (m_is16bit ? "16-bit" : "8-bit")
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
