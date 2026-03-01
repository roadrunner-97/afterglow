#include "WhiteBalanceEffect.h"
#include "ParamSlider.h"
#include <QDebug>
#include <QVBoxLayout>
#include <cmath>
#include <algorithm>
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

// ---------------------------------------------------------------------------
// Kernel — receives pre-computed per-channel multipliers from the CPU.
// The Kang formula math stays on the host; the kernel just applies the
// three multipliers.  This keeps the kernel trivially simple and avoids
// reimplementing polynomial evaluation on the GPU.
//
// 8-bit format:  QImage::Format_RGB32  (uint = 0xFFRRGGBB)
// 16-bit format: QImage::Format_RGBX64 (ushort4, LE: s0=R s1=G s2=B s3=X)
// ---------------------------------------------------------------------------
static const char* GPU_KERNEL_SOURCE = R"CL(

__kernel void applyWB(__global uint* pixels,
                      int   stride,
                      int   width,
                      int   height,
                      float rMul,
                      float gMul,
                      float bMul)
{
    int x = get_global_id(0);
    int y = get_global_id(1);
    if (x >= width || y >= height) return;

    uint pixel = pixels[y * stride + x];
    float r = ((pixel >> 16) & 0xFFu) / 255.0f;
    float g = ((pixel >>  8) & 0xFFu) / 255.0f;
    float b = ( pixel        & 0xFFu) / 255.0f;

    r = clamp(r * rMul, 0.0f, 1.0f);
    g = clamp(g * gMul, 0.0f, 1.0f);
    b = clamp(b * bMul, 0.0f, 1.0f);

    uint ri = (uint)(r * 255.0f + 0.5f);
    uint gi = (uint)(g * 255.0f + 0.5f);
    uint bi = (uint)(b * 255.0f + 0.5f);
    pixels[y * stride + x] = 0xFF000000u | (ri << 16) | (gi << 8) | bi;
}

__kernel void applyWB16(__global ushort4* pixels,
                        int   stride,
                        int   width,
                        int   height,
                        float rMul,
                        float gMul,
                        float bMul)
{
    int x = get_global_id(0);
    int y = get_global_id(1);
    if (x >= width || y >= height) return;

    ushort4 px = pixels[y * stride + x];
    float r = px.s0 / 65535.0f;
    float g = px.s1 / 65535.0f;
    float b = px.s2 / 65535.0f;

    r = clamp(r * rMul, 0.0f, 1.0f);
    g = clamp(g * gMul, 0.0f, 1.0f);
    b = clamp(b * bMul, 0.0f, 1.0f);

    px.s0 = (ushort)(r * 65535.0f + 0.5f);
    px.s1 = (ushort)(g * 65535.0f + 0.5f);
    px.s2 = (ushort)(b * 65535.0f + 0.5f);
    // px.s3 (padding) unchanged
    pixels[y * stride + x] = px;
}

)CL";

// ---------------------------------------------------------------------------
// Kang et al. (2002) Planckian locus → linear sRGB (D65 reference white).
// Used on the CPU to compute correction multipliers from a pair of K values.
// ---------------------------------------------------------------------------
static void kangToRGB(float T, float& r, float& g, float& b) {
    T = std::max(1000.0f, std::min(15000.0f, T));

    float x, y;
    if (T <= 4000.0f) {
        x = -0.2661239e9f/(T*T*T) - 0.2343580e6f/(T*T) + 0.8776956e3f/T + 0.179910f;
        y = (T <= 2222.0f)
            ? (-1.1063814f*(x*x*x) - 1.34811020f*(x*x) + 2.18555832f*x - 0.20219683f)
            : (-0.9549476f*(x*x*x) - 1.37418593f*(x*x) + 2.09137015f*x - 0.16748867f);
    } else {
        x = -3.0258469e9f/(T*T*T) + 2.1070379e6f/(T*T) + 0.2226347e3f/T + 0.240390f;
        y = 3.0817580f*(x*x*x) - 5.8733867f*(x*x) + 3.75112997f*x - 0.37001483f;
    }

    float X = x / y;
    float Z = (1.0f - x - y) / y;
    r = std::max(1e-6f,  3.2406f*X - 1.5372f - 0.4986f*Z);
    g = std::max(1e-6f, -0.9689f*X + 1.8758f + 0.0415f*Z);
    b = std::max(1e-6f,  0.0557f*X - 0.2040f + 1.0570f*Z);
}

// ---------------------------------------------------------------------------
// Compute the three per-channel multipliers to shift the apparent white point
// from shotK to targetK, then layer in the tint correction.
//
// Convention: targetK > shotK → warmer image (more amber, less blue).
//             tint > 0 → magenta shift (less green).
//             When targetK == shotK and tint == 0: all multipliers are 1.
// ---------------------------------------------------------------------------
static void computeWBMuls(float shotK, float targetK, float tint,
                           float& rMul, float& gMul, float& bMul)
{
    float rs, gs, bs, rt, gt, bt;
    kangToRGB(shotK,   rs, gs, bs);
    kangToRGB(targetK, rt, gt, bt);

    // Normalise both to G = 1 so we only shift chromaticity, not exposure.
    rs /= gs;  bs /= gs;   // gs → 1
    rt /= gt;  bt /= gt;   // gt → 1

    // The pixel-space multiplier: when shotK == targetK, rs==rt → ratio = 1.
    rMul = rs / rt;
    gMul = 1.0f;
    bMul = bs / bt;

    // Tint axis (green ↔ magenta), orthogonal to temperature.
    // tint ∈ [−100, +100]:  +100 = 15% less green (magenta), −100 = 15% more green.
    gMul *= 1.0f - tint / 100.0f * 0.15f;

    // Safety clamp: avoid divide-by-near-zero or extreme blowouts.
    rMul = std::max(0.05f, std::min(20.0f, rMul));
    gMul = std::max(0.05f, std::min(20.0f, gMul));
    bMul = std::max(0.05f, std::min(20.0f, bMul));
}

// ---------------------------------------------------------------------------
// GpuContext (per-effect OpenCL objects)
// ---------------------------------------------------------------------------
struct GpuContext : GpuContextBase<GpuContext> {
    cl::Kernel kernel;
    cl::Kernel kernel16;

    void init() {
        cl::Device device;
        if (!acquireDevice(device, "WhiteBalance")) return;
        try {
            cl::Program prog(context, GPU_KERNEL_SOURCE);
            prog.build({device});
            kernel   = cl::Kernel(prog, "applyWB");
            kernel16 = cl::Kernel(prog, "applyWB16");
            available = true;
            qDebug() << "[GPU] WhiteBalance ready on:"
                     << QString::fromStdString(device.getInfo<CL_DEVICE_NAME>());
        } catch (const cl::Error& e) {
            qWarning() << "[GPU] WhiteBalance init failed:" << e.what() << "(err" << e.err() << ")";
        }
    }
};

static std::mutex gpuMutex;

static QImage processImageGPU(const QImage& image,
                               float shotK, float targetK, float tint)
{
    QImage result = image.convertToFormat(QImage::Format_RGB32);
    const int    width    = result.width();
    const int    height   = result.height();
    const int    stride   = result.bytesPerLine() / 4;
    const size_t bufBytes = static_cast<size_t>(result.bytesPerLine()) * height;

    float rMul, gMul, bMul;
    computeWBMuls(shotK, targetK, tint, rMul, gMul, bMul);

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
        gpu.kernel.setArg(4, rMul);
        gpu.kernel.setArg(5, gMul);
        gpu.kernel.setArg(6, bMul);

        gpu.queue.enqueueNDRangeKernel(gpu.kernel, cl::NullRange,
                                       cl::NDRange(width, height), cl::NullRange);
        gpu.queue.finish();
        gpu.queue.enqueueReadBuffer(buf, CL_TRUE, 0, bufBytes, result.bits());
    } catch (const cl::Error& e) {
        qWarning() << "[GPU] WB kernel failed:" << e.what() << "(err" << e.err() << ")";
        return {};
    }
    return result;
}

static QImage processImageGPU16(const QImage& image,
                                 float shotK, float targetK, float tint)
{
    QImage result = image; // already RGBX64
    const int    width    = result.width();
    const int    height   = result.height();
    const int    stride   = result.bytesPerLine() / 8;
    const size_t bufBytes = static_cast<size_t>(result.bytesPerLine()) * height;

    float rMul, gMul, bMul;
    computeWBMuls(shotK, targetK, tint, rMul, gMul, bMul);

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
        gpu.kernel16.setArg(4, rMul);
        gpu.kernel16.setArg(5, gMul);
        gpu.kernel16.setArg(6, bMul);

        gpu.queue.enqueueNDRangeKernel(gpu.kernel16, cl::NullRange,
                                       cl::NDRange(width, height), cl::NullRange);
        gpu.queue.finish();
        gpu.queue.enqueueReadBuffer(buf, CL_TRUE, 0, bufBytes, result.bits());
    } catch (const cl::Error& e) {
        qWarning() << "[GPU] WB16 kernel failed:" << e.what() << "(err" << e.err() << ")";
        return {};
    }
    return result;
}

} // namespace

// ============================================================================
// IGpuEffect — shared pipeline interface
// ============================================================================

bool WhiteBalanceEffect::initGpuKernels(cl::Context& ctx, cl::Device& dev) {
    try {
        cl::Program prog(ctx, GPU_KERNEL_SOURCE);
        prog.build({dev});
        m_kernel   = cl::Kernel(prog, "applyWB");
        m_kernel16 = cl::Kernel(prog, "applyWB16");
        return true;
    } catch (const cl::Error& e) {
        qWarning() << "[GpuPipeline] WhiteBalance initGpuKernels failed:"
                   << e.what() << "(err" << e.err() << ")";
        return false;
    }
}

bool WhiteBalanceEffect::enqueueGpu(cl::CommandQueue& queue,
                                    cl::Buffer& buf, cl::Buffer& /*aux*/,
                                    int w, int h, int stride, bool is16bit,
                                    const QMap<QString, QVariant>& params)
{
    const float shotK   = static_cast<float>(params.value("shot_temp",   5500.0).toDouble());
    const float targetK = static_cast<float>(params.value("temperature", 5500.0).toDouble());
    const float tint    = static_cast<float>(params.value("tint",        0.0).toDouble());

    float rMul, gMul, bMul;
    computeWBMuls(shotK, targetK, tint, rMul, gMul, bMul);

    cl::Kernel& k = is16bit ? m_kernel16 : m_kernel;
    k.setArg(0, buf);
    k.setArg(1, stride);
    k.setArg(2, w);
    k.setArg(3, h);
    k.setArg(4, rMul);
    k.setArg(5, gMul);
    k.setArg(6, bMul);
    queue.enqueueNDRangeKernel(k, cl::NullRange, cl::NDRange(w, h), cl::NullRange);
    return true;
}

// ============================================================================
// PhotoEditorEffect
// ============================================================================

WhiteBalanceEffect::WhiteBalanceEffect()
    : controlsWidget(nullptr), temperatureParam(nullptr), tintParam(nullptr)
    , m_shotK(5500.0f)
{
}

WhiteBalanceEffect::~WhiteBalanceEffect() {
}

QString WhiteBalanceEffect::getName() const    { return "White Balance"; }
QString WhiteBalanceEffect::getDescription() const { return "Adjusts color temperature and tint"; }
QString WhiteBalanceEffect::getVersion() const { return "1.0.0"; }
bool    WhiteBalanceEffect::initialize()       { qDebug() << "White Balance effect initialized"; return true; }

void WhiteBalanceEffect::onImageLoaded(const ImageMetadata& meta) {
    // Use the as-shot temperature from metadata if available; keep 5500 K otherwise.
    if (meta.colorTempK >= 1500.0f && meta.colorTempK <= 14000.0f)
        m_shotK = meta.colorTempK;
    else
        m_shotK = 5500.0f;

    // Update the slider to the new default without firing any signals
    // (ParamSlider::setValue blocks signals internally).
    if (temperatureParam)
        temperatureParam->setValue(m_shotK);
}

QWidget* WhiteBalanceEffect::createControlsWidget() {
    if (controlsWidget) return controlsWidget;

    controlsWidget = new QWidget();
    QVBoxLayout* layout = new QVBoxLayout(controlsWidget);
    layout->setContentsMargins(0, 0, 0, 0);
    layout->setSpacing(10);

    // Temperature: 2000 K – 12000 K, 50 K steps, default = as-shot (m_shotK).
    // Slider right = warmer (amber), left = cooler (blue).
    temperatureParam = new ParamSlider("Temperature", 2000.0, 12000.0, 50.0, 0);
    temperatureParam->setValue(m_shotK);
    temperatureParam->setToolTip("Colour temperature of the light source in Kelvin.\nLower = cooler/blue (e.g. overcast sky ~7000 K), higher = warmer/amber (e.g. tungsten ~3200 K).\nDefault is the as-shot value read from the RAW file metadata.");
    connect(temperatureParam, &ParamSlider::editingFinished, this, [this]() {
        emit parametersChanged();
    });
    connect(temperatureParam, &ParamSlider::valueChanged, this, [this](double) {
        emit liveParametersChanged();
    });
    layout->addWidget(temperatureParam);

    // Tint: −100 (green) … +100 (magenta), 1-step, default 0.
    tintParam = new ParamSlider("Tint", -100.0, 100.0, 1.0, 0);
    tintParam->setToolTip("Fine-tunes the green–magenta axis, orthogonal to colour temperature.\nPositive = magenta shift (less green), negative = green shift.\nUse to correct fluorescent or mixed-light casts after setting temperature.");
    connect(tintParam, &ParamSlider::editingFinished, this, [this]() {
        emit parametersChanged();
    });
    connect(tintParam, &ParamSlider::valueChanged, this, [this](double) {
        emit liveParametersChanged();
    });
    layout->addWidget(tintParam);

    layout->addStretch();
    return controlsWidget;
}

QMap<QString, QVariant> WhiteBalanceEffect::getParameters() const {
    QMap<QString, QVariant> params;
    params["temperature"] = temperatureParam ? temperatureParam->value() : static_cast<double>(m_shotK);
    params["tint"]        = tintParam        ? tintParam->value()        : 0.0;
    params["shot_temp"]   = static_cast<double>(m_shotK);
    return params;
}

QImage WhiteBalanceEffect::processImage(const QImage& image,
                                        const QMap<QString, QVariant>& parameters)
{
    if (image.isNull()) return image;

    const float shotK   = static_cast<float>(parameters.value("shot_temp",   5500.0).toDouble());
    const float targetK = static_cast<float>(parameters.value("temperature", 5500.0).toDouble());
    const float tint    = static_cast<float>(parameters.value("tint",        0.0).toDouble());

    if (image.format() == QImage::Format_RGBX64)
        return processImageGPU16(image, shotK, targetK, tint);
    return processImageGPU(image, shotK, targetK, tint);
}
