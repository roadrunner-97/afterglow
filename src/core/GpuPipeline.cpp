#include "GpuPipeline.h"
#include "IGpuEffect.h"
#include "GpuDeviceRegistry.h"
#include "GpuDeviceRegistryOCL.h"
#include <QDebug>
#include <QElapsedTimer>
#include <algorithm>

// Preview kernels output RGBA byte order (R,G,B,A in memory on little-endian)
// so the result is compatible with both GL_RGBA8 textures and QImage::Format_RGBA8888.
//
// Byte layout trick: on a LE machine a uint32 value
//   (0xFF << 24) | (B << 16) | (G << 8) | R
// is stored in memory as bytes [R, G, B, 0xFF] — i.e. RGBA. ✓
// cropX0/Y0/X1/Y1 are the visible region in source image pixels (may extend
// outside [0..srcW/H) — those pixels are output as black for letterboxing).
static const char* PREVIEW_KERNEL_SOURCE = R"CL(
kernel void preview_downsample_8bit(
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
    // Output RGBA: bytes [R, G, B, 0xFF] on little-endian
    if (n) dst[dy*dstW+dx] = 0xFF000000u
        | ((uint)(b/n+.5f)<<16)
        | ((uint)(g/n+.5f)<<8)
        |  (uint)(r/n+.5f);
}

kernel void preview_downsample_16bit(
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
    // Output RGBA: bytes [R, G, B, 0xFF] on little-endian
    if (n) dst[dy*dstW+dx] = 0xFF000000u
        | ((uint)(b/n/257.f+.5f)<<16)
        | ((uint)(g/n/257.f+.5f)<<8)
        |  (uint)(r/n/257.f+.5f);
}
)CL";

// ── run ──────────────────────────────────────────────────────────────────────

QImage GpuPipeline::run(const QImage& image, const QVector<GpuPipelineCall>& calls,
                        const ViewportRequest& viewport, bool viewportOnly) {
    std::lock_guard<std::mutex> lock(m_mutex);

    const int rev = GpuDeviceRegistry::instance().revision();
    if (!m_available || m_revision != rev) {
        m_available           = false;
        m_processedFrameValid = false;
        m_lastBits            = nullptr;
        m_initializedEffects.clear();
        m_previewW = 0;
        m_previewH = 0;
        if (!initContext())
            return {};
        m_revision = rev;
    }

    // Viewport-only: skip the effects chain if we already have a processed frame.
    // Falls back to a full run on first frame, after device change, or after
    // a new image is loaded (m_processedFrameValid is cleared in those cases).
    const bool doEffects = !viewportOnly || !m_processedFrameValid;

    if (doEffects) {
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
    }

    QElapsedTimer t;
    t.start();

    try {
        qint64 kernelUs = 0;

        if (doEffects) {
            // GPU-to-GPU copy of original — no PCIe transfer
            m_queue.enqueueCopyBuffer(m_srcBuf, m_workBuf, 0, 0, m_bufBytes);

            for (const auto& call : calls) {
                auto* g = dynamic_cast<IGpuEffect*>(call.effect);
                if (!g->enqueueGpu(m_queue, m_workBuf, m_auxBuf,
                                   m_width, m_height, m_stride, m_is16bit, call.params)) {
                    qWarning() << "[GpuPipeline]" << call.effect->getName()
                               << "enqueueGpu() failed — aborting pipeline";
                    return {};
                }
            }

            // Single finish for the entire kernel chain
            m_queue.finish();
            m_processedFrameValid = true;
            kernelUs = (t.nsecsElapsed() + 500) / 1000;
        }

        // Output is always the full viewport size.
        const int previewW = viewport.displaySize.isValid() ? viewport.displaySize.width()  : m_width;
        const int previewH = viewport.displaySize.isValid() ? viewport.displaySize.height() : m_height;

        // Compute the visible crop region in source image pixels.
        // Mirrors the pan/zoom math in ViewportWidget exactly.
        const float W  = m_width,  H  = m_height;
        const float Vw = previewW, Vh = previewH;
        const float fitScale    = std::min(Vw / W, Vh / H);
        const float displayScale = fitScale * viewport.zoom;
        const float regionW = Vw / displayScale;
        const float regionH = Vh / displayScale;
        const float cropX0 = (float)viewport.center.x() * W - regionW * 0.5f;
        const float cropY0 = (float)viewport.center.y() * H - regionH * 0.5f;
        const float cropX1 = cropX0 + regionW;
        const float cropY1 = cropY0 + regionH;

        // Reallocate preview staging buffer if size changed
        if (m_previewW != previewW || m_previewH != previewH) {
            m_previewBuf = cl::Buffer(m_context, CL_MEM_READ_WRITE,
                                     static_cast<size_t>(previewW) * previewH * 4);
            m_previewW = previewW;
            m_previewH = previewH;
        }

        // Enqueue downsample/crop kernel (output is RGBA byte order)
        cl::Kernel& kernel = m_is16bit ? m_previewKernel16 : m_previewKernel8;
        kernel.setArg(0, m_workBuf);
        kernel.setArg(1, m_previewBuf);
        kernel.setArg(2, m_width);
        kernel.setArg(3, m_height);
        kernel.setArg(4, m_stride);
        kernel.setArg(5, previewW);
        kernel.setArg(6, previewH);
        kernel.setArg(7, cropX0);
        kernel.setArg(8, cropY0);
        kernel.setArg(9, cropX1);
        kernel.setArg(10, cropY1);
        m_queue.enqueueNDRangeKernel(kernel, cl::NullRange,
                                     cl::NDRange(previewW, previewH));

        // Read back to CPU
        m_queue.finish();

        QImage result(previewW, previewH, QImage::Format_RGBA8888);
        m_queue.enqueueReadBuffer(m_previewBuf, CL_TRUE, 0,
                                  static_cast<size_t>(previewW) * previewH * 4,
                                  result.bits());
        qint64 totalUs = (t.nsecsElapsed() + 500) / 1000;

        qDebug() << "[GpuPipeline]"
                 << (doEffects ? "copy+kernels:" : "[viewport-only] kernels skipped,")
                 << kernelUs << "µs"
                 << " downsample+readback:" << (totalUs - kernelUs) << "µs"
                 << " total:" << totalUs << "µs"
                 << " preview:" << previewW << "x" << previewH;
        return result;

    } catch (const cl::Error& e) {
        qWarning() << "[GpuPipeline] run() failed:" << e.what() << "(err" << e.err() << ")";
        m_available = false;
        return {};
    }
}

// ── initContext ───────────────────────────────────────────────────────────────

bool GpuPipeline::initContext() {
    cl::Device   device;
    cl::Platform platform;
    if (!GpuDeviceRegistryOCL::getSelectedDevice(device, platform)) {
        qWarning() << "[GpuPipeline] no OpenCL device available";
        return false;
    }

    try {
        m_context = cl::Context(device);
        m_queue   = cl::CommandQueue(m_context, device);
        m_device  = device;
        m_available = true;
        qDebug() << "[GpuPipeline] context ready on:"
                 << QString::fromStdString(device.getInfo<CL_DEVICE_NAME>());

        if (!initPreviewKernels()) {
            m_available = false;
            return false;
        }

        return true;
    } catch (const cl::Error& e) {
        qWarning() << "[GpuPipeline] initContext failed:" << e.what() << "(err" << e.err() << ")";
        return false;
    }
}

bool GpuPipeline::initPreviewKernels() {
    try {
        cl::Program prog(m_context, PREVIEW_KERNEL_SOURCE);
        prog.build({m_device});
        m_previewKernel8  = cl::Kernel(prog, "preview_downsample_8bit");
        m_previewKernel16 = cl::Kernel(prog, "preview_downsample_16bit");
        return true;
    } catch (const cl::Error& e) {
        qWarning() << "[GpuPipeline] initPreviewKernels failed:" << e.what()
                   << "(err" << e.err() << ")";
        try {
            cl::Program prog(m_context, PREVIEW_KERNEL_SOURCE);
            std::string log = prog.getBuildInfo<CL_PROGRAM_BUILD_LOG>(m_device);
            qWarning() << "Build log:" << QString::fromStdString(log);
        } catch (...) {}
        return false;
    }
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
        m_srcBuf  = cl::Buffer(m_context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                               m_bufBytes, src.bits());
        m_workBuf = cl::Buffer(m_context, CL_MEM_READ_WRITE, m_bufBytes);
        m_auxBuf  = cl::Buffer(m_context, CL_MEM_READ_WRITE, m_bufBytes);
        m_lastBits = image.constBits();

        qDebug() << "[GpuPipeline] uploaded" << m_width << "x" << m_height
                 << (m_is16bit ? "16-bit" : "8-bit")
                 << "bufBytes" << m_bufBytes
                 << "in" << (t.nsecsElapsed() + 500) / 1000 << "µs";

    } catch (const cl::Error& e) {
        qWarning() << "[GpuPipeline] upload failed:" << e.what() << "(err" << e.err() << ")";
        m_available = false;
    }
}
