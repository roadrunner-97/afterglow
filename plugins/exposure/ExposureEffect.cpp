#include "ExposureEffect.h"
#include "ParamSlider.h"
#include <QDebug>
#include <QVBoxLayout>
#include <cmath>
#include <mutex>

// ============================================================================
// GPU path (OpenCL)
// ============================================================================

#define CL_HPP_TARGET_OPENCL_VERSION 200
#define CL_HPP_MINIMUM_OPENCL_VERSION 110
#define CL_HPP_ENABLE_EXCEPTIONS
#include <CL/opencl.hpp>
#include "GpuDeviceRegistryOCL.h"

namespace {

// libraw outputs sRGB gamma-encoded data (output_color=1), so exposure must be
// applied in linear light: decode sRGB → scale by 2^EV → re-encode sRGB.
// Without linearisation, channels clip at different rates and highlights become
// garish/saturated (hue shift from asymmetric clipping in gamma space).
//
// evFactor = pow(2.0, ev) — precomputed on host, passed as a single float.
// 8-bit path: QImage::Format_RGB32 (uint = 0xFFRRGGBB), stride = bytesPerLine / 4.
// 16-bit path: QImage::Format_RGBX64 (ushort4 per pixel), stride = bytesPerLine / 8.
static const char* GPU_KERNEL_SOURCE = R"CL(
float srgb_to_linear(float v) {
    return v <= 0.04045f ? v * (1.0f / 12.92f)
                         : native_powr((v + 0.055f) * (1.0f / 1.055f), 2.4f);
}

float linear_to_srgb(float v) {
    v = clamp(v, 0.0f, 1.0f);
    return v <= 0.0031308f ? v * 12.92f
                           : 1.055f * native_powr(v, 1.0f / 2.4f) - 0.055f;
}

__kernel void adjustExposure(__global uint* pixels,
                              int   stride,
                              int   width,
                              int   height,
                              float evFactor)
{
    int x = get_global_id(0);
    int y = get_global_id(1);
    if (x >= width || y >= height) return;

    uint pixel = pixels[y * stride + x];
    float r = srgb_to_linear(((pixel >> 16) & 0xFFu) / 255.0f);
    float g = srgb_to_linear(((pixel >>  8) & 0xFFu) / 255.0f);
    float b = srgb_to_linear(( pixel        & 0xFFu) / 255.0f);

    r = linear_to_srgb(r * evFactor);
    g = linear_to_srgb(g * evFactor);
    b = linear_to_srgb(b * evFactor);

    uint ri = (uint)(r * 255.0f + 0.5f);
    uint gi = (uint)(g * 255.0f + 0.5f);
    uint bi = (uint)(b * 255.0f + 0.5f);
    pixels[y * stride + x] = 0xFF000000u | (ri << 16) | (gi << 8) | bi;
}

__kernel void adjustExposure16(__global ushort4* pixels,
                                int   stride,
                                int   width,
                                int   height,
                                float evFactor)
{
    int x = get_global_id(0);
    int y = get_global_id(1);
    if (x >= width || y >= height) return;

    ushort4 px = pixels[y * stride + x];
    float r = srgb_to_linear(px.s0 / 65535.0f);
    float g = srgb_to_linear(px.s1 / 65535.0f);
    float b = srgb_to_linear(px.s2 / 65535.0f);

    r = linear_to_srgb(r * evFactor);
    g = linear_to_srgb(g * evFactor);
    b = linear_to_srgb(b * evFactor);

    px.s0 = (ushort)(r * 65535.0f + 0.5f);
    px.s1 = (ushort)(g * 65535.0f + 0.5f);
    px.s2 = (ushort)(b * 65535.0f + 0.5f);
    // px.s3 (alpha/X) unchanged
    pixels[y * stride + x] = px;
}
)CL";

struct GpuContext {
    cl::Context      context;
    cl::CommandQueue queue;
    cl::Kernel       kernel;
    cl::Kernel       kernel16;
    bool             available  = false;
    int              m_revision = 0;

    // Must be called with gpuMutex held.
    static GpuContext& instance() {
        static GpuContext ctx;
        int rev = GpuDeviceRegistry::instance().revision();
        if (ctx.m_revision != rev) {
            ctx            = GpuContext{};
            ctx.m_revision = rev;
            ctx.init();
        }
        return ctx;
    }

private:
    void init() {
        cl::Device   device;
        cl::Platform platform;
        if (!GpuDeviceRegistryOCL::getSelectedDevice(device, platform)) {
            qWarning() << "[GPU] Exposure: no OpenCL device available";
            return;
        }
        try {
            context = cl::Context(device);
            queue   = cl::CommandQueue(context, device);
            cl::Program prog(context, GPU_KERNEL_SOURCE);
            prog.build({device});
            kernel    = cl::Kernel(prog, "adjustExposure");
            kernel16  = cl::Kernel(prog, "adjustExposure16");
            available = true;
            qDebug() << "[GPU] Exposure ready on:"
                     << QString::fromStdString(device.getInfo<CL_DEVICE_NAME>());
        } catch (const cl::Error& e) {
            qWarning() << "[GPU] Exposure init failed:" << e.what() << "(err" << e.err() << ")";
        }
    }
};

static std::mutex gpuMutex;

static QImage processImageGPU(const QImage& image, float evFactor) {
    QImage result = image.convertToFormat(QImage::Format_RGB32);
    const int    width    = result.width();
    const int    height   = result.height();
    const int    stride   = result.bytesPerLine() / 4;
    const size_t bufBytes = static_cast<size_t>(result.bytesPerLine()) * height;

    try {
        std::lock_guard<std::mutex> lock(gpuMutex);
        GpuContext& gpu = GpuContext::instance();
        if (!gpu.available) return {};

        cl::Buffer buf(gpu.context,
                       CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                       bufBytes, result.bits());

        gpu.kernel.setArg(0, buf);
        gpu.kernel.setArg(1, stride);
        gpu.kernel.setArg(2, width);
        gpu.kernel.setArg(3, height);
        gpu.kernel.setArg(4, evFactor);

        gpu.queue.enqueueNDRangeKernel(gpu.kernel, cl::NullRange,
                                       cl::NDRange(width, height), cl::NullRange);
        gpu.queue.finish();
        gpu.queue.enqueueReadBuffer(buf, CL_TRUE, 0, bufBytes, result.bits());
    } catch (const cl::Error& e) {
        qWarning() << "[GPU] Exposure kernel failed:" << e.what() << "(err" << e.err() << ")";
        return {};
    }
    return result;
}

static QImage processImageGPU16(const QImage& image, float evFactor) {
    QImage result = image;
    const int    width    = result.width();
    const int    height   = result.height();
    const int    stride   = result.bytesPerLine() / 8;
    const size_t bufBytes = static_cast<size_t>(result.bytesPerLine()) * height;

    try {
        std::lock_guard<std::mutex> lock(gpuMutex);
        GpuContext& gpu = GpuContext::instance();
        if (!gpu.available) return {};

        cl::Buffer buf(gpu.context,
                       CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                       bufBytes, result.bits());

        gpu.kernel16.setArg(0, buf);
        gpu.kernel16.setArg(1, stride);
        gpu.kernel16.setArg(2, width);
        gpu.kernel16.setArg(3, height);
        gpu.kernel16.setArg(4, evFactor);

        gpu.queue.enqueueNDRangeKernel(gpu.kernel16, cl::NullRange,
                                       cl::NDRange(width, height), cl::NullRange);
        gpu.queue.finish();
        gpu.queue.enqueueReadBuffer(buf, CL_TRUE, 0, bufBytes, result.bits());
    } catch (const cl::Error& e) {
        qWarning() << "[GPU] Exposure16 kernel failed:" << e.what() << "(err" << e.err() << ")";
        return {};
    }
    return result;
}

} // namespace

ExposureEffect::ExposureEffect()
    : controlsWidget(nullptr), exposureParam(nullptr) {
}

ExposureEffect::~ExposureEffect() {
}

QString ExposureEffect::getName() const {
    return "Exposure";
}

QString ExposureEffect::getDescription() const {
    return "Adjusts overall image exposure in EV stops";
}

QString ExposureEffect::getVersion() const {
    return "1.0.0";
}

bool ExposureEffect::initialize() {
    qDebug() << "Exposure effect initialized";
    return true;
}

QWidget* ExposureEffect::createControlsWidget() {
    if (controlsWidget) return controlsWidget;

    controlsWidget = new QWidget();
    QVBoxLayout* layout = new QVBoxLayout(controlsWidget);
    layout->setContentsMargins(0, 0, 0, 0);
    layout->setSpacing(8);

    // 0.1 EV steps, 1 decimal place, range -5 to +5
    exposureParam = new ParamSlider("Exposure (EV)", -5.0, 5.0, 0.1, 1);
    connect(exposureParam, &ParamSlider::editingFinished, this, [this]() {
        emit parametersChanged();
    });
    connect(exposureParam, &ParamSlider::valueChanged, this, [this](double) {
        emit liveParametersChanged();
    });
    layout->addWidget(exposureParam);

    layout->addStretch();
    return controlsWidget;
}

QMap<QString, QVariant> ExposureEffect::getParameters() const {
    QMap<QString, QVariant> params;
    params["exposure"] = exposureParam ? exposureParam->value() : 0.0;
    return params;
}

// ============================================================================
// IGpuEffect — shared pipeline interface
// ============================================================================

bool ExposureEffect::initGpuKernels(cl::Context& ctx, cl::Device& dev) {
    try {
        cl::Program prog(ctx, GPU_KERNEL_SOURCE);
        prog.build({dev});
        m_kernel   = cl::Kernel(prog, "adjustExposure");
        m_kernel16 = cl::Kernel(prog, "adjustExposure16");
        return true;
    } catch (const cl::Error& e) {
        qWarning() << "[GpuPipeline] Exposure initGpuKernels failed:"
                   << e.what() << "(err" << e.err() << ")";
        return false;
    }
}

bool ExposureEffect::enqueueGpu(cl::CommandQueue& queue,
                                 cl::Buffer& buf, cl::Buffer& /*aux*/,
                                 int w, int h, int stride, bool is16bit,
                                 const QMap<QString, QVariant>& params) {
    const float ev       = static_cast<float>(params.value("exposure", 0.0).toDouble());
    const float evFactor = std::pow(2.0f, ev);

    cl::Kernel& k = is16bit ? m_kernel16 : m_kernel;
    k.setArg(0, buf);
    k.setArg(1, stride);
    k.setArg(2, w);
    k.setArg(3, h);
    k.setArg(4, evFactor);
    queue.enqueueNDRangeKernel(k, cl::NullRange, cl::NDRange(w, h), cl::NullRange);
    return true;
}

QImage ExposureEffect::processImage(const QImage& image, const QMap<QString, QVariant>& parameters) {
    if (image.isNull()) return image;

    const float ev       = static_cast<float>(parameters.value("exposure", 0.0).toDouble());
    const float evFactor = std::pow(2.0f, ev);

    if (image.format() == QImage::Format_RGBX64)
        return processImageGPU16(image, evFactor);
    return processImageGPU(image, evFactor);
}
