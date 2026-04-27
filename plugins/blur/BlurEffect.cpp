#include "BlurEffect.h"
#include "ParamSlider.h"
#include "blur_kernels.h"
#include "color_kernels.h"
#include <QDebug>
#include <QVBoxLayout>
#include <QLabel>
#include <QComboBox>
#include <algorithm>
#include <mutex>

// ============================================================================
// GPU path (OpenCL) — two-pass separable blur (horizontal then vertical)
// ============================================================================

#define CL_HPP_TARGET_OPENCL_VERSION 200
#define CL_HPP_MINIMUM_OPENCL_VERSION 110
#define CL_HPP_ENABLE_EXCEPTIONS
#include <CL/opencl.hpp>
#include "GpuDeviceRegistryOCL.h"
#include "GpuContextBase.h"

namespace {

// Pixels are QImage::Format_RGB32: each uint = 0xFFRRGGBB.
// stride = bytesPerLine/4.
// Gaussian sigma = radius/3 (edge weight ≈ exp(-4.5) ≈ 0.01 — negligible).
// Box blur: uniform weights, equivalent to sigma = ∞.
static const char* GPU_KERNEL_SOURCE = R"CL(

// 8-bit path ---------------------------------------------------------------
__kernel void blurHorizontal(__global const uint* input,
                              __global       uint* output,
                              int stride, int width, int height,
                              int radius, int isGaussian)
{
    int x = get_global_id(0);
    int y = get_global_id(1);
    if (x >= width || y >= height) return;

    float sigma = max((float)radius / 3.0f, 0.5f);
    float r = 0.0f, g = 0.0f, b = 0.0f, wsum = 0.0f;

    for (int dx = -radius; dx <= radius; dx++) {
        int sx = clamp(x + dx, 0, width - 1);
        uint pixel = input[y * stride + sx];
        float w = isGaussian ? exp(-0.5f * dx * dx / (sigma * sigma)) : 1.0f;
        r += w * ((pixel >> 16) & 0xFFu);
        g += w * ((pixel >>  8) & 0xFFu);
        b += w * ( pixel        & 0xFFu);
        wsum += w;
    }

    uint ri = (uint)(r / wsum + 0.5f);
    uint gi = (uint)(g / wsum + 0.5f);
    uint bi = (uint)(b / wsum + 0.5f);
    output[y * stride + x] = 0xFF000000u | (ri << 16) | (gi << 8) | bi;
}

__kernel void blurVertical(__global const uint* input,
                            __global       uint* output,
                            int stride, int width, int height,
                            int radius, int isGaussian)
{
    int x = get_global_id(0);
    int y = get_global_id(1);
    if (x >= width || y >= height) return;

    float sigma = max((float)radius / 3.0f, 0.5f);
    float r = 0.0f, g = 0.0f, b = 0.0f, wsum = 0.0f;

    for (int dy = -radius; dy <= radius; dy++) {
        int sy = clamp(y + dy, 0, height - 1);
        uint pixel = input[sy * stride + x];
        float w = isGaussian ? exp(-0.5f * dy * dy / (sigma * sigma)) : 1.0f;
        r += w * ((pixel >> 16) & 0xFFu);
        g += w * ((pixel >>  8) & 0xFFu);
        b += w * ( pixel        & 0xFFu);
        wsum += w;
    }

    uint ri = (uint)(r / wsum + 0.5f);
    uint gi = (uint)(g / wsum + 0.5f);
    uint bi = (uint)(b / wsum + 0.5f);
    output[y * stride + x] = 0xFF000000u | (ri << 16) | (gi << 8) | bi;
}

// 16-bit path --------------------------------------------------------------
// pixels are QImage::Format_RGBX64 (ushort4 per pixel).
// On little-endian: ushort4.s0=R, .s1=G, .s2=B, .s3=A
// stride = bytesPerLine / 8

__kernel void blurHorizontal16(__global const ushort4* input,
                                __global       ushort4* output,
                                int stride, int width, int height,
                                int radius, int isGaussian)
{
    int x = get_global_id(0);
    int y = get_global_id(1);
    if (x >= width || y >= height) return;

    float sigma = max((float)radius / 3.0f, 0.5f);
    float r = 0.0f, g = 0.0f, b = 0.0f, wsum = 0.0f;

    for (int dx = -radius; dx <= radius; dx++) {
        int sx = clamp(x + dx, 0, width - 1);
        ushort4 px = input[y * stride + sx];
        float w = isGaussian ? exp(-0.5f * dx * dx / (sigma * sigma)) : 1.0f;
        r += w * px.s0;
        g += w * px.s1;
        b += w * px.s2;
        wsum += w;
    }

    ushort4 out;
    out.s0 = (ushort)(r / wsum + 0.5f);
    out.s1 = (ushort)(g / wsum + 0.5f);
    out.s2 = (ushort)(b / wsum + 0.5f);
    out.s3 = 65535;
    output[y * stride + x] = out;
}

__kernel void blurVertical16(__global const ushort4* input,
                              __global       ushort4* output,
                              int stride, int width, int height,
                              int radius, int isGaussian)
{
    int x = get_global_id(0);
    int y = get_global_id(1);
    if (x >= width || y >= height) return;

    float sigma = max((float)radius / 3.0f, 0.5f);
    float r = 0.0f, g = 0.0f, b = 0.0f, wsum = 0.0f;

    for (int dy = -radius; dy <= radius; dy++) {
        int sy = clamp(y + dy, 0, height - 1);
        ushort4 px = input[sy * stride + x];
        float w = isGaussian ? exp(-0.5f * dy * dy / (sigma * sigma)) : 1.0f;
        r += w * px.s0;
        g += w * px.s1;
        b += w * px.s2;
        wsum += w;
    }

    ushort4 out;
    out.s0 = (ushort)(r / wsum + 0.5f);
    out.s1 = (ushort)(g / wsum + 0.5f);
    out.s2 = (ushort)(b / wsum + 0.5f);
    out.s3 = 65535;
    output[y * stride + x] = out;
}

)CL";

struct GpuContext : GpuContextBase<GpuContext> {
    cl::Kernel kernelH;    // horizontal pass (8-bit)
    cl::Kernel kernelV;    // vertical pass   (8-bit)
    cl::Kernel kernelH16;  // horizontal pass (16-bit)
    cl::Kernel kernelV16;  // vertical pass   (16-bit)

    void init() {
        cl::Device device;
        if (!acquireDevice(device, "Blur")) return;
        try {
            cl::Program prog(context, GPU_KERNEL_SOURCE);
            prog.build({device});
            kernelH   = cl::Kernel(prog, "blurHorizontal");
            kernelV   = cl::Kernel(prog, "blurVertical");
            kernelH16 = cl::Kernel(prog, "blurHorizontal16");
            kernelV16 = cl::Kernel(prog, "blurVertical16");
            available = true;
            qDebug() << "[GPU] Blur ready on:"
                     << QString::fromStdString(device.getInfo<CL_DEVICE_NAME>());
        }
        // GCOVR_EXCL_START
        catch (const cl::Error& e) {
            qWarning() << "[GPU] Blur init failed:" << e.what() << "(err" << e.err() << ")";
        }
        // GCOVR_EXCL_STOP
    }
};

static std::mutex gpuMutex;

static QImage processImageGPU(const QImage& image, int radius, bool gaussian) {
    QImage result = image.convertToFormat(QImage::Format_RGB32);
    const int    width    = result.width();
    const int    height   = result.height();
    const int    stride   = result.bytesPerLine() / 4;
    const size_t bufBytes = static_cast<size_t>(result.bytesPerLine()) * height;

    try {
        std::lock_guard<std::mutex> lock(gpuMutex);
        GpuContext& gpu = GpuContext::instance();
        if (!gpu.available) return {};

        // Two buffers — ping-pong between horizontal and vertical passes
        cl::Buffer bufA(gpu.context,
                        CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                        bufBytes, result.bits());
        cl::Buffer bufB(gpu.context, CL_MEM_READ_WRITE, bufBytes);

        const int isGaussian = gaussian ? 1 : 0;

        // Horizontal: A → B
        gpu.kernelH.setArg(0, bufA);
        gpu.kernelH.setArg(1, bufB);
        gpu.kernelH.setArg(2, stride);
        gpu.kernelH.setArg(3, width);
        gpu.kernelH.setArg(4, height);
        gpu.kernelH.setArg(5, radius);
        gpu.kernelH.setArg(6, isGaussian);
        gpu.queue.enqueueNDRangeKernel(gpu.kernelH, cl::NullRange,
                                       cl::NDRange(width, height), cl::NullRange);
        gpu.queue.finish();

        // Vertical: B → A
        gpu.kernelV.setArg(0, bufB);
        gpu.kernelV.setArg(1, bufA);
        gpu.kernelV.setArg(2, stride);
        gpu.kernelV.setArg(3, width);
        gpu.kernelV.setArg(4, height);
        gpu.kernelV.setArg(5, radius);
        gpu.kernelV.setArg(6, isGaussian);
        gpu.queue.enqueueNDRangeKernel(gpu.kernelV, cl::NullRange,
                                       cl::NDRange(width, height), cl::NullRange);
        gpu.queue.finish();

        // Read result back from A
        gpu.queue.enqueueReadBuffer(bufA, CL_TRUE, 0, bufBytes, result.bits());
    }
    // GCOVR_EXCL_START
    catch (const cl::Error& e) {
        qWarning() << "[GPU] Blur kernel failed:" << e.what() << "(err" << e.err() << ")";
        return {};
    }
    // GCOVR_EXCL_STOP
    return result;
}

static QImage processImageGPU16(const QImage& image, int radius, bool gaussian) {
    QImage result = image; // already RGBX64
    const int    width    = result.width();
    const int    height   = result.height();
    const int    stride   = result.bytesPerLine() / 8;
    const size_t bufBytes = static_cast<size_t>(result.bytesPerLine()) * height;
    const int    isGaussian = gaussian ? 1 : 0;

    try {
        std::lock_guard<std::mutex> lock(gpuMutex);
        GpuContext& gpu = GpuContext::instance();
        if (!gpu.available) return {};

        cl::Buffer bufA(gpu.context,
                        CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                        bufBytes, result.bits());
        cl::Buffer bufB(gpu.context, CL_MEM_READ_WRITE, bufBytes);

        // Horizontal: A → B
        gpu.kernelH16.setArg(0, bufA);
        gpu.kernelH16.setArg(1, bufB);
        gpu.kernelH16.setArg(2, stride);
        gpu.kernelH16.setArg(3, width);
        gpu.kernelH16.setArg(4, height);
        gpu.kernelH16.setArg(5, radius);
        gpu.kernelH16.setArg(6, isGaussian);
        gpu.queue.enqueueNDRangeKernel(gpu.kernelH16, cl::NullRange,
                                       cl::NDRange(width, height), cl::NullRange);
        gpu.queue.finish();

        // Vertical: B → A
        gpu.kernelV16.setArg(0, bufB);
        gpu.kernelV16.setArg(1, bufA);
        gpu.kernelV16.setArg(2, stride);
        gpu.kernelV16.setArg(3, width);
        gpu.kernelV16.setArg(4, height);
        gpu.kernelV16.setArg(5, radius);
        gpu.kernelV16.setArg(6, isGaussian);
        gpu.queue.enqueueNDRangeKernel(gpu.kernelV16, cl::NullRange,
                                       cl::NDRange(width, height), cl::NullRange);
        gpu.queue.finish();

        gpu.queue.enqueueReadBuffer(bufA, CL_TRUE, 0, bufBytes, result.bits());
    }
    // GCOVR_EXCL_START
    catch (const cl::Error& e) {
        qWarning() << "[GPU] Blur16 kernel failed:" << e.what() << "(err" << e.err() << ")";
        return {};
    }
    // GCOVR_EXCL_STOP
    return result;
}

} // namespace

// ============================================================================
// Pipeline kernel — float4 linear sRGB.  Blur is done directly in linear light,
// which gives specular highlights their natural bloom / bright-over-dark
// spread.  Two-pass separable: H(buf→aux), V(aux→buf); result in buf.
// ============================================================================
static const char* PIPELINE_KERNEL_SOURCE = SHARED_BLUR_KERNELS_F4;

// ============================================================================
// IGpuEffect — shared pipeline interface
// ============================================================================

bool BlurEffect::initGpuKernels(cl::Context& ctx, cl::Device& dev) {
    try {
        cl::Program prog(ctx, PIPELINE_KERNEL_SOURCE);
        prog.build({dev});
        m_kernelBlurHLinear = cl::Kernel(prog, "blurHLinear");
        m_kernelBlurVLinear = cl::Kernel(prog, "blurVLinear");
        return true;
    }
    // GCOVR_EXCL_START
    catch (const cl::Error& e) {
        qWarning() << "[GpuPipeline] Blur initGpuKernels failed:"
                   << e.what() << "(err" << e.err() << ")";
        return false;
    }
    // GCOVR_EXCL_STOP
}

bool BlurEffect::enqueueGpu(cl::CommandQueue& queue,
                             cl::Buffer& buf, cl::Buffer& aux,
                             int w, int h,
                             const QMap<QString, QVariant>& params) {
    const int radiusSrc = params.value("radius", 0).toInt();
    if (radiusSrc == 0) return true;  // no-op

    // Scale radius from source-pixel units to preview-pixel units so that the
    // perceived blur strength is independent of zoom level.
    const double scale = params.value("_srcPixelsPerPreviewPixel", 1.0).toDouble();
    int radius = std::max(1, static_cast<int>(radiusSrc / std::max(scale, 1e-6) + 0.5));

    const int isGaussian = (params.value("blurType", 0).toInt() == 0) ? 1 : 0;
    const cl::NDRange global(w, h);

    // Horizontal: buf → aux
    m_kernelBlurHLinear.setArg(0, buf);
    m_kernelBlurHLinear.setArg(1, aux);
    m_kernelBlurHLinear.setArg(2, w);
    m_kernelBlurHLinear.setArg(3, h);
    m_kernelBlurHLinear.setArg(4, radius);
    m_kernelBlurHLinear.setArg(5, isGaussian);
    queue.enqueueNDRangeKernel(m_kernelBlurHLinear, cl::NullRange, global, cl::NullRange);

    // Vertical: aux → buf (in-order queue; no finish needed)
    m_kernelBlurVLinear.setArg(0, aux);
    m_kernelBlurVLinear.setArg(1, buf);
    m_kernelBlurVLinear.setArg(2, w);
    m_kernelBlurVLinear.setArg(3, h);
    m_kernelBlurVLinear.setArg(4, radius);
    m_kernelBlurVLinear.setArg(5, isGaussian);
    queue.enqueueNDRangeKernel(m_kernelBlurVLinear, cl::NullRange, global, cl::NullRange);

    return true;
}

// ============================================================================
// Effect implementation
// ============================================================================

BlurEffect::BlurEffect()
    : controlsWidget(nullptr), blurTypeCombo(nullptr),
      radiusParam(nullptr), blurType(0) {
}

BlurEffect::~BlurEffect() {
}

QString BlurEffect::getName() const        { return "Blur"; }
QString BlurEffect::getDescription() const { return "Gaussian and box blur"; }
QString BlurEffect::getVersion() const     { return "1.0.0"; }

bool BlurEffect::initialize() {
    qDebug() << "Blur effect initialized";
    return true;
}

QWidget* BlurEffect::createControlsWidget() {
    if (controlsWidget) return controlsWidget;

    controlsWidget = new QWidget();
    QVBoxLayout* layout = new QVBoxLayout(controlsWidget);
    layout->setContentsMargins(0, 0, 0, 0);
    layout->setSpacing(8);

    // Blur type
    QLabel* typeLabel = new QLabel("Blur type:");
    typeLabel->setStyleSheet("color: #2C2018;");
    layout->addWidget(typeLabel);

    blurTypeCombo = new QComboBox();
    blurTypeCombo->addItem("Gaussian");
    blurTypeCombo->addItem("Box");
    blurTypeCombo->setToolTip("Gaussian: bell-curve weights for a soft, natural-looking blur.\nBox: uniform weights, slightly harder-edged but faster at large radii.");
    blurTypeCombo->setStyleSheet(
        "QComboBox { color: #2C2018; background-color: #F4F1EA;"
        "            border: 1px solid #CCC5B5; border-radius: 3px; padding: 3px; }"
        "QComboBox::drop-down { border: none; }"
        "QComboBox QAbstractItemView { color: #2C2018; background-color: #F4F1EA; }");
    layout->addWidget(blurTypeCombo);

    connect(blurTypeCombo, QOverload<int>::of(&QComboBox::activated), this, [this](int index) {
        blurType = index;
        emit parametersChanged();
    });

    // Radius
    radiusParam = new ParamSlider("Radius", 0, 50);
    radiusParam->setToolTip("Neighbourhood radius in pixels. Higher values produce stronger blurring. 0 = no effect.");
    connect(radiusParam, &ParamSlider::editingFinished, this, [this]() {
        emit parametersChanged();
    });
    connect(radiusParam, &ParamSlider::valueChanged, this, [this](double) {
        emit liveParametersChanged();
    });
    layout->addWidget(radiusParam);

    layout->addStretch();
    return controlsWidget;
}

QMap<QString, QVariant> BlurEffect::getParameters() const {
    QMap<QString, QVariant> params;
    params["blurType"] = blurType;
    params["radius"]   = radiusParam ? static_cast<int>(radiusParam->value()) : 0;
    return params;
}

void BlurEffect::applyParameters(const QMap<QString, QVariant>& parameters) {
    if (parameters.contains("blurType")) {
        blurType = parameters.value("blurType").toInt();
        if (blurTypeCombo) {
            QSignalBlocker block(blurTypeCombo);
            blurTypeCombo->setCurrentIndex(blurType);
        }
    }
    if (radiusParam && parameters.contains("radius"))
        radiusParam->setValue(parameters.value("radius").toDouble());
    emit parametersChanged();
}

QImage BlurEffect::processImage(const QImage& image, const QMap<QString, QVariant>& parameters) {
    if (image.isNull()) return image;

    const int radius    = parameters.value("radius",   0).toInt();
    const bool gaussian = (parameters.value("blurType", 0).toInt() == 0);

    if (radius == 0) return image;

    if (image.format() == QImage::Format_RGBX64)
        return processImageGPU16(image, radius, gaussian);
    return processImageGPU(image, radius, gaussian);
}
