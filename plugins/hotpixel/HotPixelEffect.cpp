#include "HotPixelEffect.h"
#include "ParamSlider.h"
#include <QDebug>
#include <QVBoxLayout>
#include <mutex>

// ============================================================================
// GPU path (OpenCL) — single-pass 3x3 neighbourhood average replacement
//
// Algorithm: for each pixel, compute the average of its 8 neighbours.
// If any channel deviates from that average by more than `threshold`, replace
// that channel with the neighbour average.  Isolated hot/dead pixels (which
// read wildly above or below their surroundings) are corrected; normal
// detail is untouched because both the pixel and its neighbours deviate
// similarly across edges.
// ============================================================================

#define CL_HPP_TARGET_OPENCL_VERSION 200
#define CL_HPP_MINIMUM_OPENCL_VERSION 110
#define CL_HPP_ENABLE_EXCEPTIONS
#include <CL/opencl.hpp>
#include "GpuDeviceRegistryOCL.h"
#include "GpuContextBase.h"

namespace {

// threshold is supplied in the native channel range:
//   8-bit  → 0–255
//   16-bit → 0–65535
static const char* GPU_KERNEL_SOURCE = R"CL(

// 8-bit path  (QImage::Format_RGB32, each uint = 0xFFRRGGBB)
// stride = bytesPerLine / 4
__kernel void hotPixelRemove(
    __global const uint* input,
    __global       uint* output,
    int stride, int width, int height,
    float threshold)
{
    int x = get_global_id(0);
    int y = get_global_id(1);
    if (x >= width || y >= height) return;

    uint center = input[y * stride + x];
    float cr = (float)((center >> 16) & 0xFFu);
    float cg = (float)((center >>  8) & 0xFFu);
    float cb = (float)( center        & 0xFFu);

    float sumR = 0.0f, sumG = 0.0f, sumB = 0.0f;
    for (int dy = -1; dy <= 1; dy++) {
        for (int dx = -1; dx <= 1; dx++) {
            if (dx == 0 && dy == 0) continue;
            int nx = clamp(x + dx, 0, width  - 1);
            int ny = clamp(y + dy, 0, height - 1);
            uint n  = input[ny * stride + nx];
            sumR += (float)((n >> 16) & 0xFFu);
            sumG += (float)((n >>  8) & 0xFFu);
            sumB += (float)( n        & 0xFFu);
        }
    }
    float avgR = sumR * 0.125f;   // / 8
    float avgG = sumG * 0.125f;
    float avgB = sumB * 0.125f;

    uint outR = (fabs(cr - avgR) > threshold) ? (uint)(avgR + 0.5f) : (uint)cr;
    uint outG = (fabs(cg - avgG) > threshold) ? (uint)(avgG + 0.5f) : (uint)cg;
    uint outB = (fabs(cb - avgB) > threshold) ? (uint)(avgB + 0.5f) : (uint)cb;

    output[y * stride + x] = 0xFF000000u | (outR << 16) | (outG << 8) | outB;
}

// 16-bit path  (QImage::Format_RGBX64, ushort4 per pixel)
// stride = bytesPerLine / 8
__kernel void hotPixelRemove16(
    __global const ushort4* input,
    __global       ushort4* output,
    int stride, int width, int height,
    float threshold)
{
    int x = get_global_id(0);
    int y = get_global_id(1);
    if (x >= width || y >= height) return;

    ushort4 center = input[y * stride + x];
    float cr = (float)center.s0;
    float cg = (float)center.s1;
    float cb = (float)center.s2;

    float sumR = 0.0f, sumG = 0.0f, sumB = 0.0f;
    for (int dy = -1; dy <= 1; dy++) {
        for (int dx = -1; dx <= 1; dx++) {
            if (dx == 0 && dy == 0) continue;
            int nx = clamp(x + dx, 0, width  - 1);
            int ny = clamp(y + dy, 0, height - 1);
            ushort4 n = input[ny * stride + nx];
            sumR += (float)n.s0;
            sumG += (float)n.s1;
            sumB += (float)n.s2;
        }
    }
    float avgR = sumR * 0.125f;
    float avgG = sumG * 0.125f;
    float avgB = sumB * 0.125f;

    ushort4 out;
    out.s0 = (fabs(cr - avgR) > threshold) ? (ushort)(avgR + 0.5f) : (ushort)cr;
    out.s1 = (fabs(cg - avgG) > threshold) ? (ushort)(avgG + 0.5f) : (ushort)cg;
    out.s2 = (fabs(cb - avgB) > threshold) ? (ushort)(avgB + 0.5f) : (ushort)cb;
    out.s3 = 65535;
    output[y * stride + x] = out;
}

)CL";

struct GpuContext : GpuContextBase<GpuContext> {
    cl::Kernel kernel8;
    cl::Kernel kernel16;

    void init() {
        cl::Device device;
        if (!acquireDevice(device, "HotPixel")) return;
        try {
            cl::Program prog(context, GPU_KERNEL_SOURCE);
            prog.build({device});
            kernel8  = cl::Kernel(prog, "hotPixelRemove");
            kernel16 = cl::Kernel(prog, "hotPixelRemove16");
            available = true;
            qDebug() << "[GPU] HotPixel ready on:"
                     << QString::fromStdString(device.getInfo<CL_DEVICE_NAME>());
        } catch (const cl::Error& e) {
            qWarning() << "[GPU] HotPixel init failed:" << e.what() << "(err" << e.err() << ")";
        }
    }
};

static std::mutex gpuMutex;

// Single place that owns the UI (0–100) → native channel scale conversion.
// 8-bit channels: 0–255; 16-bit channels: 0–65535.
static float scaledThreshold(int pct, bool is16bit) {
    return pct * (is16bit ? 655.35f : 2.55f);
}

// threshold: pre-scaled to native channel range (call scaledThreshold() first)
static QImage processImageGPU(const QImage& image, float threshold) {
    QImage result   = image.convertToFormat(QImage::Format_RGB32);
    const int  w    = result.width();
    const int  h    = result.height();
    const int  stride = result.bytesPerLine() / 4;
    const size_t bufBytes = static_cast<size_t>(result.bytesPerLine()) * h;

    try {
        std::lock_guard<std::mutex> lock(gpuMutex);
        GpuContext& gpu = GpuContext::instance();
        if (!gpu.available) return {};

        cl::Buffer bufA(gpu.context,
                        CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                        bufBytes, result.bits());
        cl::Buffer bufB(gpu.context, CL_MEM_READ_WRITE, bufBytes);

        gpu.kernel8.setArg(0, bufA);
        gpu.kernel8.setArg(1, bufB);
        gpu.kernel8.setArg(2, stride);
        gpu.kernel8.setArg(3, w);
        gpu.kernel8.setArg(4, h);
        gpu.kernel8.setArg(5, threshold);
        gpu.queue.enqueueNDRangeKernel(gpu.kernel8, cl::NullRange,
                                       cl::NDRange(w, h), cl::NullRange);
        gpu.queue.finish();
        gpu.queue.enqueueReadBuffer(bufB, CL_TRUE, 0, bufBytes, result.bits());
    } catch (const cl::Error& e) {
        qWarning() << "[GPU] HotPixel kernel failed:" << e.what() << "(err" << e.err() << ")";
        return {};
    }
    return result;
}

// threshold: pre-scaled to native channel range (call scaledThreshold() first)
static QImage processImageGPU16(const QImage& image, float threshold) {
    QImage result     = image; // already RGBX64
    const int  w      = result.width();
    const int  h      = result.height();
    const int  stride = result.bytesPerLine() / 8;
    const size_t bufBytes = static_cast<size_t>(result.bytesPerLine()) * h;

    try {
        std::lock_guard<std::mutex> lock(gpuMutex);
        GpuContext& gpu = GpuContext::instance();
        if (!gpu.available) return {};

        cl::Buffer bufA(gpu.context,
                        CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                        bufBytes, result.bits());
        cl::Buffer bufB(gpu.context, CL_MEM_READ_WRITE, bufBytes);

        gpu.kernel16.setArg(0, bufA);
        gpu.kernel16.setArg(1, bufB);
        gpu.kernel16.setArg(2, stride);
        gpu.kernel16.setArg(3, w);
        gpu.kernel16.setArg(4, h);
        gpu.kernel16.setArg(5, threshold);
        gpu.queue.enqueueNDRangeKernel(gpu.kernel16, cl::NullRange,
                                       cl::NDRange(w, h), cl::NullRange);
        gpu.queue.finish();
        gpu.queue.enqueueReadBuffer(bufB, CL_TRUE, 0, bufBytes, result.bits());
    } catch (const cl::Error& e) {
        qWarning() << "[GPU] HotPixel16 kernel failed:" << e.what() << "(err" << e.err() << ")";
        return {};
    }
    return result;
}

} // namespace

// ============================================================================
// IGpuEffect — shared pipeline interface
// ============================================================================

bool HotPixelEffect::initGpuKernels(cl::Context& ctx, cl::Device& dev) {
    try {
        cl::Program prog(ctx, GPU_KERNEL_SOURCE);
        prog.build({dev});
        m_kernel8  = cl::Kernel(prog, "hotPixelRemove");
        m_kernel16 = cl::Kernel(prog, "hotPixelRemove16");
        return true;
    } catch (const cl::Error& e) {
        qWarning() << "[GpuPipeline] HotPixel initGpuKernels failed:"
                   << e.what() << "(err" << e.err() << ")";
        return false;
    }
}

bool HotPixelEffect::enqueueGpu(cl::CommandQueue& queue,
                                 cl::Buffer& buf, cl::Buffer& aux,
                                 int w, int h, int stride, bool is16bit,
                                 const QMap<QString, QVariant>& params) {
    const int   thresholdPct = params.value("threshold", 30).toInt();
    const float threshold    = scaledThreshold(thresholdPct, is16bit);

    cl::Kernel& k = is16bit ? m_kernel16 : m_kernel8;
    k.setArg(0, buf);
    k.setArg(1, aux);
    k.setArg(2, stride);
    k.setArg(3, w);
    k.setArg(4, h);
    k.setArg(5, threshold);
    queue.enqueueNDRangeKernel(k, cl::NullRange, cl::NDRange(w, h), cl::NullRange);

    // Result is in aux; copy back to buf so downstream effects see it in buf.
    // (In-order queue: copy waits for the kernel above.)
    const size_t bufBytes = static_cast<size_t>(stride) * h * (is16bit ? 8 : 4);
    queue.enqueueCopyBuffer(aux, buf, 0, 0, bufBytes);

    return true;
}

// ============================================================================
// Effect implementation
// ============================================================================

HotPixelEffect::HotPixelEffect()
    : controlsWidget(nullptr), thresholdParam(nullptr) {}

HotPixelEffect::~HotPixelEffect() {}

QString HotPixelEffect::getName() const        { return "Hot Pixel Removal"; }
QString HotPixelEffect::getDescription() const { return "Removes hot/dead pixels by replacing outliers with the local neighbourhood average"; }
QString HotPixelEffect::getVersion() const     { return "1.0.0"; }

bool HotPixelEffect::initialize() {
    qDebug() << "HotPixel effect initialized";
    return true;
}

QWidget* HotPixelEffect::createControlsWidget() {
    if (controlsWidget) return controlsWidget;

    controlsWidget = new QWidget();
    QVBoxLayout* layout = new QVBoxLayout(controlsWidget);
    layout->setContentsMargins(0, 0, 0, 0);
    layout->setSpacing(8);

    // Threshold slider: 0–100 (maps internally to per-channel deviation 0–255 / 0–65535)
    thresholdParam = new ParamSlider("Threshold", 0, 100);
    thresholdParam->setValue(30);
    thresholdParam->setToolTip("How far a pixel must deviate from its 8 neighbours to be considered a hot/dead pixel and replaced with the local average.\nLower values = more aggressive correction. Typical range: 20–40.");
    connect(thresholdParam, &ParamSlider::editingFinished, this, [this]() {
        emit parametersChanged();
    });
    connect(thresholdParam, &ParamSlider::valueChanged, this, [this](double) {
        emit liveParametersChanged();
    });
    layout->addWidget(thresholdParam);

    layout->addStretch();
    return controlsWidget;
}

QMap<QString, QVariant> HotPixelEffect::getParameters() const {
    QMap<QString, QVariant> params;
    params["threshold"] = thresholdParam ? static_cast<int>(thresholdParam->value()) : 30;
    return params;
}

QImage HotPixelEffect::processImage(const QImage& image,
                                     const QMap<QString, QVariant>& parameters) {
    if (image.isNull()) return image;

    const int thresholdPct = parameters.value("threshold", 30).toInt();
    if (thresholdPct == 0) return image;

    const bool is16 = image.format() == QImage::Format_RGBX64;
    const float threshold = scaledThreshold(thresholdPct, is16);
    if (is16)
        return processImageGPU16(image, threshold);
    return processImageGPU(image, threshold);
}
