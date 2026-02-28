#include "GpuPipeline.h"
#include "IGpuEffect.h"
#include "GpuDeviceRegistry.h"
#include "GpuDeviceRegistryOCL.h"
#include <QDebug>
#include <QElapsedTimer>

static const char* PREVIEW_KERNEL_SOURCE = R"CL(
kernel void preview_downsample_8bit(
    global const uint* src, global uint* dst,
    int srcW, int srcH, int srcStride, int dstW, int dstH)
{
    int dx = get_global_id(0), dy = get_global_id(1);
    if (dx >= dstW || dy >= dstH) return;
    int x0 = dx * srcW / dstW, y0 = dy * srcH / dstH;
    int x1 = min((dx+1) * srcW / dstW + 1, srcW);
    int y1 = min((dy+1) * srcH / dstH + 1, srcH);
    float r=0,g=0,b=0; int n=0;
    for (int sy=y0;sy<y1;sy++) for (int sx=x0;sx<x1;sx++) {
        uint p = src[sy*srcStride+sx];
        r += (p>>16)&0xFF; g += (p>>8)&0xFF; b += p&0xFF; n++;
    }
    if (n) dst[dy*dstW+dx] = 0xFF000000|((int)(r/n+.5f)<<16)|((int)(g/n+.5f)<<8)|(int)(b/n+.5f);
}

kernel void preview_downsample_16bit(
    global const ushort* src, global uint* dst,
    int srcW, int srcH, int srcStride, int dstW, int dstH)
{
    int dx = get_global_id(0), dy = get_global_id(1);
    if (dx >= dstW || dy >= dstH) return;
    int x0 = dx * srcW / dstW, y0 = dy * srcH / dstH;
    int x1 = min((dx+1) * srcW / dstW + 1, srcW);
    int y1 = min((dy+1) * srcH / dstH + 1, srcH);
    float r=0,g=0,b=0; int n=0;
    for (int sy=y0;sy<y1;sy++) for (int sx=x0;sx<x1;sx++) {
        int i = (sy*srcStride+sx)*4;
        r += src[i]; g += src[i+1]; b += src[i+2]; n++;
    }
    if (n) dst[dy*dstW+dx] = 0xFF000000
        | ((int)(r/n/257.f+.5f)<<16)
        | ((int)(g/n/257.f+.5f)<<8)
        |  (int)(b/n/257.f+.5f);
}
)CL";

QImage GpuPipeline::run(const QImage& image, const QVector<GpuPipelineCall>& calls,
                        const ViewportRequest& viewport) {
    std::lock_guard<std::mutex> lock(m_mutex);

    const int rev = GpuDeviceRegistry::instance().revision();
    if (!m_available || m_revision != rev) {
        m_available = false;
        m_lastBits  = nullptr;  // force re-upload after context rebuild
        m_initializedEffects.clear();
        m_previewW = 0;
        m_previewH = 0;
        if (!initContext())
            return {};
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
        // GPU-to-GPU copy of original — no PCIe transfer
        m_queue.enqueueCopyBuffer(m_srcBuf, m_workBuf, 0, 0, m_bufBytes);

        for (const auto& call : calls) {
            auto* g = static_cast<IGpuEffect*>(dynamic_cast<IGpuEffect*>(call.effect));
            if (!g->enqueueGpu(m_queue, m_workBuf, m_auxBuf,
                               m_width, m_height, m_stride, m_is16bit, call.params)) {
                qWarning() << "[GpuPipeline]" << call.effect->getName()
                           << "enqueueGpu() failed — aborting pipeline";
                return {};
            }
        }

        // Single finish for the entire kernel chain
        m_queue.finish();
        qint64 kernelUs = (t.nsecsElapsed() + 500) / 1000;

        // Compute preview size, maintaining aspect ratio
        QSize srcSize(m_width, m_height);
        QSize previewSize = viewport.displaySize.isValid()
                          ? srcSize.scaled(viewport.displaySize, Qt::KeepAspectRatio)
                          : srcSize;
        // Clamp to source size (no upscale)
        if (previewSize.width() > m_width || previewSize.height() > m_height)
            previewSize = srcSize;

        const int previewW = previewSize.width();
        const int previewH = previewSize.height();

        // Reallocate preview buffer if size changed
        if (m_previewW != previewW || m_previewH != previewH) {
            m_previewBuf = cl::Buffer(m_context, CL_MEM_WRITE_ONLY,
                                     static_cast<size_t>(previewW) * previewH * 4);
            m_previewW = previewW;
            m_previewH = previewH;
        }

        // Enqueue downsample kernel
        cl::Kernel& kernel = m_is16bit ? m_previewKernel16 : m_previewKernel8;
        kernel.setArg(0, m_workBuf);
        kernel.setArg(1, m_previewBuf);
        kernel.setArg(2, m_width);
        kernel.setArg(3, m_height);
        kernel.setArg(4, m_stride);
        kernel.setArg(5, previewW);
        kernel.setArg(6, previewH);
        m_queue.enqueueNDRangeKernel(kernel, cl::NullRange,
                                     cl::NDRange(previewW, previewH));
        m_queue.finish();

        // Read back only the small preview
        QImage result(previewW, previewH, QImage::Format_RGB32);
        m_queue.enqueueReadBuffer(m_previewBuf, CL_TRUE, 0,
                                  static_cast<size_t>(previewW) * previewH * 4,
                                  result.bits());
        qint64 totalUs = (t.nsecsElapsed() + 500) / 1000;

        qDebug() << "[GpuPipeline] copy+kernels:" << kernelUs << "µs"
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

    // Force preview buffer reallocation on next run
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
