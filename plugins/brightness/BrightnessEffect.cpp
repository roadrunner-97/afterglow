#include "BrightnessEffect.h"
#include "ParamSlider.h"
#include "color_kernels.h"
#include <QDebug>
#ifdef QT_DEBUG
#include <QElapsedTimer>
#endif
#include <QVBoxLayout>
#include <mutex>

// ============================================================================
// GPU path (OpenCL)
// ============================================================================

#define CL_HPP_TARGET_OPENCL_VERSION 200
#define CL_HPP_MINIMUM_OPENCL_VERSION 110
#define CL_HPP_ENABLE_EXCEPTIONS
#include <CL/opencl.hpp>
#include "GpuDeviceRegistryOCL.h"
#include "GpuContextBase.h"

namespace {

// brightnessFactor: integer offset added to raw [0,255] channels (same as CPU).
// contrastFactor:   precomputed (contrast + 100) / 100, applied as (v-0.5)*f+0.5
//                   which is identical to the CPU's (channel-128)*f+128 in float space.
static const char* GPU_KERNEL_SOURCE = R"CL(
// 8-bit path: pixels are QImage::Format_RGB32 (uint = 0xFFRRGGBB)
__kernel void adjustBrightness(__global uint* pixels,
                                int   stride,
                                int   width,
                                int   height,
                                int   brightnessFactor,
                                float contrastFactor)
{
    int x = get_global_id(0);
    int y = get_global_id(1);
    if (x >= width || y >= height) return;

    uint pixel = pixels[y * stride + x];
    float r = ((pixel >> 16) & 0xFFu) / 255.0f;
    float g = ((pixel >>  8) & 0xFFu) / 255.0f;
    float b = ( pixel        & 0xFFu) / 255.0f;

    float bd = brightnessFactor / 255.0f;
    r = clamp(r + bd, 0.0f, 1.0f);
    g = clamp(g + bd, 0.0f, 1.0f);
    b = clamp(b + bd, 0.0f, 1.0f);

    r = clamp((r - 0.5f) * contrastFactor + 0.5f, 0.0f, 1.0f);
    g = clamp((g - 0.5f) * contrastFactor + 0.5f, 0.0f, 1.0f);
    b = clamp((b - 0.5f) * contrastFactor + 0.5f, 0.0f, 1.0f);

    uint ri = (uint)(r * 255.0f + 0.5f);
    uint gi = (uint)(g * 255.0f + 0.5f);
    uint bi = (uint)(b * 255.0f + 0.5f);
    pixels[y * stride + x] = 0xFF000000u | (ri << 16) | (gi << 8) | bi;
}

// 16-bit path: pixels are QImage::Format_RGBX64 (ushort4 per pixel).
// On little-endian the QRgba64 in-memory layout is [R, G, B, A] as ushort,
// so ushort4.s0=R, .s1=G, .s2=B, .s3=A.
// stride = bytesPerLine / 8  (pixels per row including any row padding).
__kernel void adjustBrightness16(__global ushort4* pixels,
                                  int   stride,
                                  int   width,
                                  int   height,
                                  int   brightnessFactor,
                                  float contrastFactor)
{
    int x = get_global_id(0);
    int y = get_global_id(1);
    if (x >= width || y >= height) return;

    ushort4 px = pixels[y * stride + x];
    float r = px.s0 / 65535.0f;
    float g = px.s1 / 65535.0f;
    float b = px.s2 / 65535.0f;

    float bd = brightnessFactor / 255.0f;
    r = clamp(r + bd, 0.0f, 1.0f);
    g = clamp(g + bd, 0.0f, 1.0f);
    b = clamp(b + bd, 0.0f, 1.0f);

    r = clamp((r - 0.5f) * contrastFactor + 0.5f, 0.0f, 1.0f);
    g = clamp((g - 0.5f) * contrastFactor + 0.5f, 0.0f, 1.0f);
    b = clamp((b - 0.5f) * contrastFactor + 0.5f, 0.0f, 1.0f);

    px.s0 = (ushort)(r * 65535.0f + 0.5f);
    px.s1 = (ushort)(g * 65535.0f + 0.5f);
    px.s2 = (ushort)(b * 65535.0f + 0.5f);
    // px.s3 (alpha) unchanged
    pixels[y * stride + x] = px;
}
)CL";

struct GpuContext : GpuContextBase<GpuContext> {
    cl::Kernel kernel;
    cl::Kernel kernel16;

    void init() {
        cl::Device device;
        if (!acquireDevice(device, "Brightness")) return;
        try {
            cl::Program prog(context, GPU_KERNEL_SOURCE);
            prog.build({device});
            kernel   = cl::Kernel(prog, "adjustBrightness");
            kernel16 = cl::Kernel(prog, "adjustBrightness16");
            available = true;
            qDebug() << "[GPU] Brightness ready on:"
                     << QString::fromStdString(device.getInfo<CL_DEVICE_NAME>());
        }
        // GCOVR_EXCL_START
        catch (const cl::Error& e) {
            qWarning() << "[GPU] Brightness init failed:" << e.what() << "(err" << e.err() << ")";
        }
        // GCOVR_EXCL_STOP
    }
};

static std::mutex gpuMutex;

static QImage processImageGPU(const QImage& image, int brightnessFactor, float contrastFactor) {
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
        gpu.kernel.setArg(4, brightnessFactor);
        gpu.kernel.setArg(5, contrastFactor);

        gpu.queue.enqueueNDRangeKernel(gpu.kernel, cl::NullRange,
                                       cl::NDRange(width, height), cl::NullRange);
        gpu.queue.finish();
        gpu.queue.enqueueReadBuffer(buf, CL_TRUE, 0, bufBytes, result.bits());
    }
    // GCOVR_EXCL_START
    catch (const cl::Error& e) {
        qWarning() << "[GPU] Brightness kernel failed:" << e.what() << "(err" << e.err() << ")";
        return {};
    }
    // GCOVR_EXCL_STOP
    return result;
}

static QImage processImageGPU16(const QImage& image, int brightnessFactor, float contrastFactor) {
#ifdef QT_DEBUG
    QElapsedTimer t;
    t.start();
#endif

    QImage result = image; // already RGBX64; copy-on-write detaches
    const int    width    = result.width();
    const int    height   = result.height();
    const int    stride   = result.bytesPerLine() / 8; // pixels per row (8 bytes each)
    const size_t bufBytes = static_cast<size_t>(result.bytesPerLine()) * height;

#ifdef QT_DEBUG
    qDebug() << "[GPU16 Brightness] image" << width << "x" << height
             << "bufBytes" << bufBytes;
#endif

    try {
        std::lock_guard<std::mutex> lock(gpuMutex);
        GpuContext& gpu = GpuContext::instance();
        if (!gpu.available) return {};
#ifdef QT_DEBUG
        qint64 t0 = t.nsecsElapsed();
#endif

        // Phase 1: detach QImage (host copy) + upload to GPU
        cl::Buffer buf(gpu.context,
                       CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                       bufBytes, result.bits());
#ifdef QT_DEBUG
        qint64 t1 = t.nsecsElapsed();
        qDebug() << "[GPU16 Brightness] lock+upload:" << (t1 - t0) / 1000 << "µs";
#endif

        gpu.kernel16.setArg(0, buf);
        gpu.kernel16.setArg(1, stride);
        gpu.kernel16.setArg(2, width);
        gpu.kernel16.setArg(3, height);
        gpu.kernel16.setArg(4, brightnessFactor);
        gpu.kernel16.setArg(5, contrastFactor);

        // Phase 2: kernel execution
        gpu.queue.enqueueNDRangeKernel(gpu.kernel16, cl::NullRange,
                                       cl::NDRange(width, height), cl::NullRange);
        gpu.queue.finish();
#ifdef QT_DEBUG
        qint64 t2 = t.nsecsElapsed();
        qDebug() << "[GPU16 Brightness] kernel exec:" << (t2 - t1) / 1000 << "µs";
#endif

        // Phase 3: readback
        gpu.queue.enqueueReadBuffer(buf, CL_TRUE, 0, bufBytes, result.bits());
#ifdef QT_DEBUG
        qint64 t3 = t.nsecsElapsed();
        qDebug() << "[GPU16 Brightness] readback:" << (t3 - t2) / 1000 << "µs";
        qDebug() << "[GPU16 Brightness] TOTAL:" << t3 / 1000 << "µs";
#endif
    }
    // GCOVR_EXCL_START
    catch (const cl::Error& e) {
        qWarning() << "[GPU] Brightness16 kernel failed:" << e.what() << "(err" << e.err() << ")";
        return {};
    }
    // GCOVR_EXCL_STOP
    return result;
}

} // namespace

// ============================================================================
// Pipeline kernel — float4 linear sRGB, used by GpuPipeline via enqueueGpu.
// Brightness and contrast were defined as slider offsets in sRGB-gamma space
// (see processImageGPU above); preserving that UI behaviour here means
// gamma-encoding to apply the offsets, then decoding back to linear.  The
// surrounding pipeline (upstream downsample, downstream effects, final pack)
// stays entirely in linear light — the full RAW dynamic range is preserved.
// ============================================================================
static const char* PIPELINE_KERNEL_SOURCE = COLOR_KERNELS_SRC R"CL(
__kernel void adjustBrightnessLinear(__global float4* pixels,
                                     int   w,
                                     int   h,
                                     int   brightnessFactor,
                                     float contrastFactor)
{
    int x = get_global_id(0);
    int y = get_global_id(1);
    if (x >= w || y >= h) return;

    float4 px = pixels[y * w + x];

    // Encode to sRGB-gamma to apply UI-defined brightness/contrast (matches
    // the slider semantics of the 8-bit processImage kernel).
    float r = linear_to_srgb(px.x);
    float g = linear_to_srgb(px.y);
    float b = linear_to_srgb(px.z);

    float bd = brightnessFactor / 255.0f;
    r = r + bd;
    g = g + bd;
    b = b + bd;

    r = (r - 0.5f) * contrastFactor + 0.5f;
    g = (g - 0.5f) * contrastFactor + 0.5f;
    b = (b - 0.5f) * contrastFactor + 0.5f;

    // Back to linear — don't clamp here; the final pack kernel clamps once.
    pixels[y * w + x] = (float4)(srgb_to_linear(r),
                                 srgb_to_linear(g),
                                 srgb_to_linear(b),
                                 1.0f);
}
)CL";

BrightnessEffect::BrightnessEffect()
    : controlsWidget(nullptr), brightnessParam(nullptr), contrastParam(nullptr) {
}

BrightnessEffect::~BrightnessEffect() {
}

QString BrightnessEffect::getName() const {
    return "Brightness & Contrast";
}

QString BrightnessEffect::getDescription() const {
    return "Adjusts brightness and contrast of the image";
}

QString BrightnessEffect::getVersion() const {
    return "2.0.0";
}

bool BrightnessEffect::initialize() {
    qDebug() << "Brightness & Contrast effect initialized";
    return true;
}

QWidget* BrightnessEffect::createControlsWidget() {
    if (controlsWidget) return controlsWidget;

    controlsWidget = new QWidget();
    QVBoxLayout* layout = new QVBoxLayout(controlsWidget);
    layout->setContentsMargins(0, 0, 0, 0);
    layout->setSpacing(8);

    brightnessParam = new ParamSlider("Brightness", -100, 100);
    brightnessParam->setToolTip("Shifts all pixel values brighter (positive) or darker (negative).");
    connect(brightnessParam, &ParamSlider::editingFinished, this, [this]() {
        emit parametersChanged();
    });
    connect(brightnessParam, &ParamSlider::valueChanged, this, [this](double) {
        emit liveParametersChanged();
    });
    layout->addWidget(brightnessParam);

    contrastParam = new ParamSlider("Contrast", -50, 50);
    contrastParam->setToolTip("Expands (positive) or compresses (negative) the tonal range around the midpoint (128).");
    connect(contrastParam, &ParamSlider::editingFinished, this, [this]() {
        emit parametersChanged();
    });
    connect(contrastParam, &ParamSlider::valueChanged, this, [this](double) {
        emit liveParametersChanged();
    });
    layout->addWidget(contrastParam);

    layout->addStretch();
    return controlsWidget;
}

QMap<QString, QVariant> BrightnessEffect::getParameters() const {
    QMap<QString, QVariant> params;
    params["brightness"] = static_cast<int>(brightnessParam ? brightnessParam->value() : 0.0);
    params["contrast"]   = static_cast<int>(contrastParam   ? contrastParam->value()   : 0.0);
    return params;
}

// ============================================================================
// IGpuEffect — shared pipeline interface
// ============================================================================

bool BrightnessEffect::initGpuKernels(cl::Context& ctx, cl::Device& dev) {
    try {
        cl::Program prog(ctx, PIPELINE_KERNEL_SOURCE);
        prog.build({dev});
        m_kernelLinear = cl::Kernel(prog, "adjustBrightnessLinear");
        return true;
    }
    // GCOVR_EXCL_START
    catch (const cl::Error& e) {
        qWarning() << "[GpuPipeline] Brightness initGpuKernels failed:"
                   << e.what() << "(err" << e.err() << ")";
        return false;
    }
    // GCOVR_EXCL_STOP
}

bool BrightnessEffect::enqueueGpu(cl::CommandQueue& queue,
                                   cl::Buffer& buf, cl::Buffer& /*aux*/,
                                   int w, int h,
                                   const QMap<QString, QVariant>& params) {
    const int brightnessFactor = params.value("brightness", 0).toInt();
    const int contrastInt      = params.value("contrast", 0).toInt();
    if (brightnessFactor == 0 && contrastInt == 0) return true;  // no-op

    const float contrastFactor = (contrastInt + 100.0f) / 100.0f;

    m_kernelLinear.setArg(0, buf);
    m_kernelLinear.setArg(1, w);
    m_kernelLinear.setArg(2, h);
    m_kernelLinear.setArg(3, brightnessFactor);
    m_kernelLinear.setArg(4, contrastFactor);
    queue.enqueueNDRangeKernel(m_kernelLinear, cl::NullRange,
                               cl::NDRange(w, h), cl::NullRange);
    return true;
}

QImage BrightnessEffect::processImage(const QImage &image, const QMap<QString, QVariant> &parameters) {
    if (image.isNull()) return image;

    int brightnessFactor   = parameters.value("brightness", 0).toInt();
    int contrastFactor     = parameters.value("contrast", 0).toInt();
    if (brightnessFactor == 0 && contrastFactor == 0) return image;

    float contrastFactor_f = (contrastFactor + 100.0f) / 100.0f;

    if (image.format() == QImage::Format_RGBX64)
        return processImageGPU16(image, brightnessFactor, contrastFactor_f);
    return processImageGPU(image, brightnessFactor, contrastFactor_f);
}
