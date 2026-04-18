#include "ColorBalanceEffect.h"
#include "ParamSlider.h"
#include "color_kernels.h"
#include <QDebug>
#include <QVBoxLayout>
#include <QLabel>
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

// Color Balance: three independent RGB offsets applied to shadows, midtones,
// and highlights.  Each tonal band has a soft luminance-mask membership that
// sums to 1 at every luminance; sliders thus feel independent without gaps or
// double-counting.
//
// Masks (smooth triangles centred at L=0, 0.5, 1):
//   shadowW    = max(0, 1 - 2*L)
//   highlightW = max(0, 2*L - 1)
//   midtoneW   = 1 - shadowW - highlightW
//
// Each slider delivers up to ±0.25 on its channel at full weight.
static const char* GPU_KERNEL_SOURCE = R"CL(
inline float3 applyBalance(float3 rgb,
                            float3 shadowOff,
                            float3 midtoneOff,
                            float3 highlightOff)
{
    float L = 0.2126f * rgb.x + 0.7152f * rgb.y + 0.0722f * rgb.z;
    float shadowW    = fmax(0.0f, 1.0f - 2.0f * L);
    float highlightW = fmax(0.0f, 2.0f * L - 1.0f);
    float midtoneW   = 1.0f - shadowW - highlightW;

    float3 off = shadowOff    * shadowW
               + midtoneOff   * midtoneW
               + highlightOff * highlightW;

    return clamp(rgb + off, 0.0f, 1.0f);
}

__kernel void applyColorBalance(__global uint* pixels,
                                 int   stride,
                                 int   width,
                                 int   height,
                                 float sR, float sG, float sB,
                                 float mR, float mG, float mB,
                                 float hR, float hG, float hB)
{
    int x = get_global_id(0);
    int y = get_global_id(1);
    if (x >= width || y >= height) return;

    uint pixel = pixels[y * stride + x];
    float3 rgb = (float3)(((pixel >> 16) & 0xFFu) / 255.0f,
                           ((pixel >>  8) & 0xFFu) / 255.0f,
                           ( pixel        & 0xFFu) / 255.0f);

    rgb = applyBalance(rgb,
                        (float3)(sR, sG, sB),
                        (float3)(mR, mG, mB),
                        (float3)(hR, hG, hB));

    uint ri = (uint)(rgb.x * 255.0f + 0.5f);
    uint gi = (uint)(rgb.y * 255.0f + 0.5f);
    uint bi = (uint)(rgb.z * 255.0f + 0.5f);
    pixels[y * stride + x] = 0xFF000000u | (ri << 16) | (gi << 8) | bi;
}

__kernel void applyColorBalance16(__global ushort4* pixels,
                                   int   stride,
                                   int   width,
                                   int   height,
                                   float sR, float sG, float sB,
                                   float mR, float mG, float mB,
                                   float hR, float hG, float hB)
{
    int x = get_global_id(0);
    int y = get_global_id(1);
    if (x >= width || y >= height) return;

    ushort4 px = pixels[y * stride + x];
    float3 rgb = (float3)(px.s0 / 65535.0f, px.s1 / 65535.0f, px.s2 / 65535.0f);

    rgb = applyBalance(rgb,
                        (float3)(sR, sG, sB),
                        (float3)(mR, mG, mB),
                        (float3)(hR, hG, hB));

    px.s0 = (ushort)(clamp(rgb.x * 65535.0f + 0.5f, 0.0f, 65535.0f));
    px.s1 = (ushort)(clamp(rgb.y * 65535.0f + 0.5f, 0.0f, 65535.0f));
    px.s2 = (ushort)(clamp(rgb.z * 65535.0f + 0.5f, 0.0f, 65535.0f));
    pixels[y * stride + x] = px;
}
)CL";

struct GpuContext : GpuContextBase<GpuContext> {
    cl::Kernel kernel;
    cl::Kernel kernel16;

    void init() {
        cl::Device device;
        if (!acquireDevice(device, "ColorBalance")) return;
        try {
            cl::Program prog(context, GPU_KERNEL_SOURCE);
            prog.build({device});
            kernel   = cl::Kernel(prog, "applyColorBalance");
            kernel16 = cl::Kernel(prog, "applyColorBalance16");
            available = true;
            qDebug() << "[GPU] ColorBalance ready on:"
                     << QString::fromStdString(device.getInfo<CL_DEVICE_NAME>());
        }
        // GCOVR_EXCL_START
        catch (const cl::Error& e) {
            qWarning() << "[GPU] ColorBalance init failed:" << e.what() << "(err" << e.err() << ")";
        }
        // GCOVR_EXCL_STOP
    }
};

static std::mutex gpuMutex;

struct BalanceArgs {
    float sR, sG, sB;
    float mR, mG, mB;
    float hR, hG, hB;
};

// Each slider is an integer in [-100, 100] mapping to ±0.25 per channel.
static constexpr float SLIDER_TO_OFFSET = 0.25f / 100.0f;

static BalanceArgs makeArgs(int sR, int sG, int sB,
                             int mR, int mG, int mB,
                             int hR, int hG, int hB) {
    BalanceArgs a;
    a.sR = sR * SLIDER_TO_OFFSET; a.sG = sG * SLIDER_TO_OFFSET; a.sB = sB * SLIDER_TO_OFFSET;
    a.mR = mR * SLIDER_TO_OFFSET; a.mG = mG * SLIDER_TO_OFFSET; a.mB = mB * SLIDER_TO_OFFSET;
    a.hR = hR * SLIDER_TO_OFFSET; a.hG = hG * SLIDER_TO_OFFSET; a.hB = hB * SLIDER_TO_OFFSET;
    return a;
}

static void setKernelArgs(cl::Kernel& k, cl::Buffer& buf,
                           int stride, int w, int h, const BalanceArgs& a) {
    k.setArg(0,  buf);
    k.setArg(1,  stride);
    k.setArg(2,  w);
    k.setArg(3,  h);
    k.setArg(4,  a.sR); k.setArg(5,  a.sG); k.setArg(6,  a.sB);
    k.setArg(7,  a.mR); k.setArg(8,  a.mG); k.setArg(9,  a.mB);
    k.setArg(10, a.hR); k.setArg(11, a.hG); k.setArg(12, a.hB);
}

static QImage processImageGPU(const QImage& image, const BalanceArgs& a) {
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

        setKernelArgs(gpu.kernel, buf, stride, width, height, a);
        gpu.queue.enqueueNDRangeKernel(gpu.kernel, cl::NullRange,
                                       cl::NDRange(width, height), cl::NullRange);
        gpu.queue.finish();
        gpu.queue.enqueueReadBuffer(buf, CL_TRUE, 0, bufBytes, result.bits());
    }
    // GCOVR_EXCL_START
    catch (const cl::Error& e) {
        qWarning() << "[GPU] ColorBalance kernel failed:" << e.what() << "(err" << e.err() << ")";
        return {};
    }
    // GCOVR_EXCL_STOP
    return result;
}

static QImage processImageGPU16(const QImage& image, const BalanceArgs& a) {
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

        setKernelArgs(gpu.kernel16, buf, stride, width, height, a);
        gpu.queue.enqueueNDRangeKernel(gpu.kernel16, cl::NullRange,
                                       cl::NDRange(width, height), cl::NullRange);
        gpu.queue.finish();
        gpu.queue.enqueueReadBuffer(buf, CL_TRUE, 0, bufBytes, result.bits());
    }
    // GCOVR_EXCL_START
    catch (const cl::Error& e) {
        qWarning() << "[GPU] ColorBalance16 kernel failed:" << e.what() << "(err" << e.err() << ")";
        return {};
    }
    // GCOVR_EXCL_STOP
    return result;
}

} // namespace

// ============================================================================
// Pipeline kernel — float4 linear sRGB, used by GpuPipeline via enqueueGpu.
// Zone masks are computed on perceptually-placed luminance (sRGB-encoded L)
// so the shadow/midtone/highlight boundaries match the UI expectation of
// today's behaviour.  Slider deltas retain their ±0.25 per-channel semantics
// but are added in LINEAR RGB — no encode/decode around the addition, so the
// full dynamic range of the pipeline is preserved.
// ============================================================================
static const char* PIPELINE_KERNEL_SOURCE = COLOR_KERNELS_SRC R"CL(
__kernel void applyColorBalanceLinear(__global float4* pixels,
                                       int   w,
                                       int   h,
                                       float sR, float sG, float sB,
                                       float mR, float mG, float mB,
                                       float hR, float hG, float hB)
{
    int x = get_global_id(0);
    int y = get_global_id(1);
    if (x >= w || y >= h) return;

    float4 px = pixels[y * w + x];

    // Zone selection uses perceptual L (sRGB-encoded luma).
    float linLum = linear_luma(px);
    float L      = linear_to_srgb(linLum);

    float shadowW    = fmax(0.0f, 1.0f - 2.0f * L);
    float highlightW = fmax(0.0f, 2.0f * L - 1.0f);
    float midtoneW   = 1.0f - shadowW - highlightW;

    float offR = sR * shadowW + mR * midtoneW + hR * highlightW;
    float offG = sG * shadowW + mG * midtoneW + hG * highlightW;
    float offB = sB * shadowW + mB * midtoneW + hB * highlightW;

    // Add offsets in linear RGB — do not clamp (the final pack kernel clamps once).
    pixels[y * w + x] = (float4)(px.x + offR,
                                 px.y + offG,
                                 px.z + offB,
                                 1.0f);
}
)CL";

// ============================================================================
// ColorBalanceEffect
// ============================================================================

ColorBalanceEffect::ColorBalanceEffect()
    : controlsWidget(nullptr),
      shadowRParam(nullptr),    shadowGParam(nullptr),    shadowBParam(nullptr),
      midtoneRParam(nullptr),   midtoneGParam(nullptr),   midtoneBParam(nullptr),
      highlightRParam(nullptr), highlightGParam(nullptr), highlightBParam(nullptr) {}

ColorBalanceEffect::~ColorBalanceEffect() {}

QString ColorBalanceEffect::getName() const { return "Color Balance"; }
QString ColorBalanceEffect::getDescription() const {
    return "Shift colour in shadows, midtones, and highlights independently";
}
QString ColorBalanceEffect::getVersion() const { return "1.0.0"; }

bool ColorBalanceEffect::initialize() {
    qDebug() << "Color Balance effect initialized";
    return true;
}

QWidget* ColorBalanceEffect::createControlsWidget() {
    if (controlsWidget) return controlsWidget;

    controlsWidget = new QWidget();
    QVBoxLayout* layout = new QVBoxLayout(controlsWidget);
    layout->setContentsMargins(0, 0, 0, 0);
    layout->setSpacing(6);

    auto connectSlider = [&](ParamSlider* s) {
        connect(s, &ParamSlider::editingFinished, this, [this]() { emit parametersChanged(); });
        connect(s, &ParamSlider::valueChanged,    this, [this](double) { emit liveParametersChanged(); });
    };

    auto addBand = [&](const char* label,
                        ParamSlider*& r, ParamSlider*& g, ParamSlider*& b) {
        auto* header = new QLabel(label, controlsWidget);
        layout->addWidget(header);
        r = new ParamSlider("Red",   -100, 100);
        g = new ParamSlider("Green", -100, 100);
        b = new ParamSlider("Blue",  -100, 100);
        r->setToolTip("Shift the red channel (negative = cyan, positive = red).");
        g->setToolTip("Shift the green channel (negative = magenta, positive = green).");
        b->setToolTip("Shift the blue channel (negative = yellow, positive = blue).");
        connectSlider(r); connectSlider(g); connectSlider(b);
        layout->addWidget(r);
        layout->addWidget(g);
        layout->addWidget(b);
    };

    addBand("Shadows",    shadowRParam,    shadowGParam,    shadowBParam);
    addBand("Midtones",   midtoneRParam,   midtoneGParam,   midtoneBParam);
    addBand("Highlights", highlightRParam, highlightGParam, highlightBParam);

    layout->addStretch();
    return controlsWidget;
}

QMap<QString, QVariant> ColorBalanceEffect::getParameters() const {
    QMap<QString, QVariant> params;
    params["shadowR"]    = static_cast<int>(shadowRParam    ? shadowRParam->value()    : 0.0);
    params["shadowG"]    = static_cast<int>(shadowGParam    ? shadowGParam->value()    : 0.0);
    params["shadowB"]    = static_cast<int>(shadowBParam    ? shadowBParam->value()    : 0.0);
    params["midtoneR"]   = static_cast<int>(midtoneRParam   ? midtoneRParam->value()   : 0.0);
    params["midtoneG"]   = static_cast<int>(midtoneGParam   ? midtoneGParam->value()   : 0.0);
    params["midtoneB"]   = static_cast<int>(midtoneBParam   ? midtoneBParam->value()   : 0.0);
    params["highlightR"] = static_cast<int>(highlightRParam ? highlightRParam->value() : 0.0);
    params["highlightG"] = static_cast<int>(highlightGParam ? highlightGParam->value() : 0.0);
    params["highlightB"] = static_cast<int>(highlightBParam ? highlightBParam->value() : 0.0);
    return params;
}

// ============================================================================
// IGpuEffect — shared pipeline interface
// ============================================================================

bool ColorBalanceEffect::initGpuKernels(cl::Context& ctx, cl::Device& dev) {
    try {
        cl::Program prog(ctx, PIPELINE_KERNEL_SOURCE);
        prog.build({dev});
        m_kernelLinear = cl::Kernel(prog, "applyColorBalanceLinear");
        return true;
    }
    // GCOVR_EXCL_START
    catch (const cl::Error& e) {
        qWarning() << "[GpuPipeline] ColorBalance initGpuKernels failed:"
                   << e.what() << "(err" << e.err() << ")";
        return false;
    }
    // GCOVR_EXCL_STOP
}

static bool allZero(const QMap<QString, QVariant>& p) {
    static const char* keys[] = {
        "shadowR", "shadowG", "shadowB",
        "midtoneR", "midtoneG", "midtoneB",
        "highlightR", "highlightG", "highlightB",
    };
    for (const char* k : keys)
        if (p.value(k, 0).toInt() != 0) return false;
    return true;
}

bool ColorBalanceEffect::enqueueGpu(cl::CommandQueue& queue,
                                     cl::Buffer& buf, cl::Buffer& /*aux*/,
                                     int w, int h,
                                     const QMap<QString, QVariant>& params) {
    if (allZero(params)) return true;  // no-op

    const BalanceArgs a = makeArgs(
        params.value("shadowR",    0).toInt(),
        params.value("shadowG",    0).toInt(),
        params.value("shadowB",    0).toInt(),
        params.value("midtoneR",   0).toInt(),
        params.value("midtoneG",   0).toInt(),
        params.value("midtoneB",   0).toInt(),
        params.value("highlightR", 0).toInt(),
        params.value("highlightG", 0).toInt(),
        params.value("highlightB", 0).toInt());

    m_kernelLinear.setArg(0,  buf);
    m_kernelLinear.setArg(1,  w);
    m_kernelLinear.setArg(2,  h);
    m_kernelLinear.setArg(3,  a.sR); m_kernelLinear.setArg(4,  a.sG); m_kernelLinear.setArg(5,  a.sB);
    m_kernelLinear.setArg(6,  a.mR); m_kernelLinear.setArg(7,  a.mG); m_kernelLinear.setArg(8,  a.mB);
    m_kernelLinear.setArg(9,  a.hR); m_kernelLinear.setArg(10, a.hG); m_kernelLinear.setArg(11, a.hB);
    queue.enqueueNDRangeKernel(m_kernelLinear, cl::NullRange,
                               cl::NDRange(w, h), cl::NullRange);
    return true;
}

QImage ColorBalanceEffect::processImage(const QImage& image, const QMap<QString, QVariant>& parameters) {
    if (image.isNull()) return image;
    if (allZero(parameters)) return image;

    const BalanceArgs a = makeArgs(
        parameters.value("shadowR",    0).toInt(),
        parameters.value("shadowG",    0).toInt(),
        parameters.value("shadowB",    0).toInt(),
        parameters.value("midtoneR",   0).toInt(),
        parameters.value("midtoneG",   0).toInt(),
        parameters.value("midtoneB",   0).toInt(),
        parameters.value("highlightR", 0).toInt(),
        parameters.value("highlightG", 0).toInt(),
        parameters.value("highlightB", 0).toInt());

    if (image.format() == QImage::Format_RGBX64)
        return processImageGPU16(image, a);
    return processImageGPU(image, a);
}
