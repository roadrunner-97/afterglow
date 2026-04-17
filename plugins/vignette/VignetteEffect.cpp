#include "VignetteEffect.h"
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
#include "GpuContextBase.h"

namespace {

// Distance metric: L^p norm on the image's aspect-normalised coordinates,
// where p is derived from the roundness slider.  Higher p → the level curve
// at d=1 approaches a rectangle (corners no longer darker than sides);
// lower p → the level curve approaches a diamond (corners much farther, very
// heavy corner darkening).  p=2 is the plain ellipse inscribed in the frame.
//
// In the UI:
//   roundness = +100  →  p = 0.5  (very round/diamond, corners darkened heavily)
//   roundness =    0  →  p = 2    (aspect-matching ellipse)
//   roundness = -100  →  p = 8    (rectangular, hugs the frame)
//
// cornerD is passed in so we can normalise d to [0..1] across the roundness
// range — the host pre-computes cornerD = 2^(1/p), the d value at the corner
// (|nx|=|ny|=1).  smoothstep from (midpoint-feather/2) to (midpoint+feather/2)
// on that normalised distance gives the transition weight t; factor=1+amount*t
// multiplies RGB (amount<0 darkens corners, amount>0 lightens).
static const char* GPU_KERNEL_SOURCE = R"CL(
__kernel void applyVignette(__global uint* pixels,
                             int   stride,
                             int   width,
                             int   height,
                             float cx,
                             float cy,
                             float halfW,
                             float halfH,
                             float amount,
                             float midpoint,
                             float feather,
                             float p,
                             float cornerD)
{
    int x = get_global_id(0);
    int y = get_global_id(1);
    if (x >= width || y >= height) return;

    float nx = fabs(((float)x - cx) / halfW);
    float ny = fabs(((float)y - cy) / halfH);
    float d  = native_powr(native_powr(nx, p) + native_powr(ny, p), 1.0f / p);
    float dn = d / cornerD;

    float edge0 = midpoint - feather * 0.5f;
    float edge1 = midpoint + feather * 0.5f + 1e-5f;
    float t = smoothstep(edge0, edge1, dn);
    float factor = clamp(1.0f + amount * t, 0.0f, 2.0f);

    uint pixel = pixels[y * stride + x];
    float r = ((pixel >> 16) & 0xFFu) / 255.0f * factor;
    float g = ((pixel >>  8) & 0xFFu) / 255.0f * factor;
    float b = ( pixel        & 0xFFu) / 255.0f * factor;

    uint ri = (uint)(clamp(r, 0.0f, 1.0f) * 255.0f + 0.5f);
    uint gi = (uint)(clamp(g, 0.0f, 1.0f) * 255.0f + 0.5f);
    uint bi = (uint)(clamp(b, 0.0f, 1.0f) * 255.0f + 0.5f);
    pixels[y * stride + x] = 0xFF000000u | (ri << 16) | (gi << 8) | bi;
}

__kernel void applyVignette16(__global ushort4* pixels,
                               int   stride,
                               int   width,
                               int   height,
                               float cx,
                               float cy,
                               float halfW,
                               float halfH,
                               float amount,
                               float midpoint,
                               float feather,
                               float p,
                               float cornerD)
{
    int x = get_global_id(0);
    int y = get_global_id(1);
    if (x >= width || y >= height) return;

    float nx = fabs(((float)x - cx) / halfW);
    float ny = fabs(((float)y - cy) / halfH);
    float d  = native_powr(native_powr(nx, p) + native_powr(ny, p), 1.0f / p);
    float dn = d / cornerD;

    float edge0 = midpoint - feather * 0.5f;
    float edge1 = midpoint + feather * 0.5f + 1e-5f;
    float t = smoothstep(edge0, edge1, dn);
    float factor = clamp(1.0f + amount * t, 0.0f, 2.0f);

    ushort4 px = pixels[y * stride + x];
    float r = px.s0 / 65535.0f * factor;
    float g = px.s1 / 65535.0f * factor;
    float b = px.s2 / 65535.0f * factor;

    px.s0 = (ushort)(clamp(r, 0.0f, 1.0f) * 65535.0f + 0.5f);
    px.s1 = (ushort)(clamp(g, 0.0f, 1.0f) * 65535.0f + 0.5f);
    px.s2 = (ushort)(clamp(b, 0.0f, 1.0f) * 65535.0f + 0.5f);
    pixels[y * stride + x] = px;
}
)CL";

struct GpuContext : GpuContextBase<GpuContext> {
    cl::Kernel kernel;
    cl::Kernel kernel16;

    void init() {
        cl::Device device;
        if (!acquireDevice(device, "Vignette")) return;
        try {
            cl::Program prog(context, GPU_KERNEL_SOURCE);
            prog.build({device});
            kernel   = cl::Kernel(prog, "applyVignette");
            kernel16 = cl::Kernel(prog, "applyVignette16");
            available = true;
            qDebug() << "[GPU] Vignette ready on:"
                     << QString::fromStdString(device.getInfo<CL_DEVICE_NAME>());
        }
        // GCOVR_EXCL_START
        catch (const cl::Error& e) {
            qWarning() << "[GPU] Vignette init failed:" << e.what() << "(err" << e.err() << ")";
        }
        // GCOVR_EXCL_STOP
    }
};

static std::mutex gpuMutex;

struct VignetteArgs {
    float amount;    // -1..1
    float midpoint;  // 0..1
    float feather;   // 0..1
    float p;         // L^p exponent
    float cornerD;   // 2^(1/p) — distance at (|nx|=1,|ny|=1)
};

static VignetteArgs makeArgs(int amount, int midpoint, int feather, int roundness) {
    VignetteArgs a;
    a.amount   = amount   / 100.0f;
    a.midpoint = midpoint / 100.0f;
    a.feather  = feather  / 100.0f;
    float rn   = roundness / 100.0f;                 // [-1, 1]
    a.p        = std::exp2(1.0f - rn * 3.0f);        // roundness=0 → p=2
    a.cornerD  = std::exp2(1.0f / a.p);              // L^p of (1,1) = 2^(1/p)
    return a;
}

static void setKernelArgs(cl::Kernel& k, cl::Buffer& buf,
                           int stride, int w, int h, const VignetteArgs& a) {
    k.setArg(0, buf);
    k.setArg(1, stride);
    k.setArg(2, w);
    k.setArg(3, h);
    k.setArg(4, w * 0.5f);
    k.setArg(5, h * 0.5f);
    k.setArg(6, w * 0.5f);
    k.setArg(7, h * 0.5f);
    k.setArg(8, a.amount);
    k.setArg(9, a.midpoint);
    k.setArg(10, a.feather);
    k.setArg(11, a.p);
    k.setArg(12, a.cornerD);
}

static QImage processImageGPU(const QImage& image, const VignetteArgs& a) {
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
        qWarning() << "[GPU] Vignette kernel failed:" << e.what() << "(err" << e.err() << ")";
        return {};
    }
    // GCOVR_EXCL_STOP
    return result;
}

static QImage processImageGPU16(const QImage& image, const VignetteArgs& a) {
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
        qWarning() << "[GPU] Vignette16 kernel failed:" << e.what() << "(err" << e.err() << ")";
        return {};
    }
    // GCOVR_EXCL_STOP
    return result;
}

} // namespace

// ============================================================================
// VignetteEffect
// ============================================================================

VignetteEffect::VignetteEffect()
    : controlsWidget(nullptr), amountParam(nullptr), midpointParam(nullptr),
      featherParam(nullptr), roundnessParam(nullptr) {
}

VignetteEffect::~VignetteEffect() {
}

QString VignetteEffect::getName() const { return "Vignette"; }
QString VignetteEffect::getDescription() const {
    return "Radial darkening or lightening around the image center";
}
QString VignetteEffect::getVersion() const { return "1.0.0"; }

bool VignetteEffect::initialize() {
    qDebug() << "Vignette effect initialized";
    return true;
}

QWidget* VignetteEffect::createControlsWidget() {
    if (controlsWidget) return controlsWidget;

    controlsWidget = new QWidget();
    QVBoxLayout* layout = new QVBoxLayout(controlsWidget);
    layout->setContentsMargins(0, 0, 0, 0);
    layout->setSpacing(8);

    auto connectSlider = [&](ParamSlider* s) {
        connect(s, &ParamSlider::editingFinished, this, [this]() { emit parametersChanged(); });
        connect(s, &ParamSlider::valueChanged,    this, [this](double) { emit liveParametersChanged(); });
    };

    amountParam = new ParamSlider("Amount", -100, 100);
    amountParam->setToolTip("Strength and direction of the vignette.\nNegative darkens the corners; positive lightens them.");
    connectSlider(amountParam);
    layout->addWidget(amountParam);

    midpointParam = new ParamSlider("Midpoint", 0, 100);
    midpointParam->setValue(50);
    midpointParam->setToolTip("Centre of the transition, as a fraction of the way from image centre to corner.\nLower values move the vignette inward.");
    connectSlider(midpointParam);
    layout->addWidget(midpointParam);

    featherParam = new ParamSlider("Feather", 0, 100);
    featherParam->setValue(50);
    featherParam->setToolTip("Softness of the transition.\n0 is a hard edge; 100 fades across the full image.");
    connectSlider(featherParam);
    layout->addWidget(featherParam);

    roundnessParam = new ParamSlider("Roundness", -100, 100);
    roundnessParam->setToolTip("Shape of the vignette.\nPositive values round the falloff (heavier corner darkening);\nnegative values pull it toward the rectangular frame.");
    connectSlider(roundnessParam);
    layout->addWidget(roundnessParam);

    layout->addStretch();
    return controlsWidget;
}

QMap<QString, QVariant> VignetteEffect::getParameters() const {
    QMap<QString, QVariant> params;
    params["amount"]    = static_cast<int>(amountParam    ? amountParam->value()    : 0.0);
    params["midpoint"]  = static_cast<int>(midpointParam  ? midpointParam->value()  : 50.0);
    params["feather"]   = static_cast<int>(featherParam   ? featherParam->value()   : 50.0);
    params["roundness"] = static_cast<int>(roundnessParam ? roundnessParam->value() : 0.0);
    return params;
}

// ============================================================================
// IGpuEffect — shared pipeline interface
// ============================================================================

bool VignetteEffect::initGpuKernels(cl::Context& ctx, cl::Device& dev) {
    try {
        cl::Program prog(ctx, GPU_KERNEL_SOURCE);
        prog.build({dev});
        m_kernel   = cl::Kernel(prog, "applyVignette");
        m_kernel16 = cl::Kernel(prog, "applyVignette16");
        return true;
    }
    // GCOVR_EXCL_START
    catch (const cl::Error& e) {
        qWarning() << "[GpuPipeline] Vignette initGpuKernels failed:"
                   << e.what() << "(err" << e.err() << ")";
        return false;
    }
    // GCOVR_EXCL_STOP
}

bool VignetteEffect::enqueueGpu(cl::CommandQueue& queue,
                                 cl::Buffer& buf, cl::Buffer& /*aux*/,
                                 int w, int h, int stride, bool is16bit,
                                 const QMap<QString, QVariant>& params) {
    const int amount = params.value("amount", 0).toInt();
    if (amount == 0) return true;  // no-op

    const VignetteArgs a = makeArgs(
        amount,
        params.value("midpoint",  50).toInt(),
        params.value("feather",   50).toInt(),
        params.value("roundness",  0).toInt());

    cl::Kernel& k = is16bit ? m_kernel16 : m_kernel;
    setKernelArgs(k, buf, stride, w, h, a);
    queue.enqueueNDRangeKernel(k, cl::NullRange, cl::NDRange(w, h), cl::NullRange);
    return true;
}

QImage VignetteEffect::processImage(const QImage& image, const QMap<QString, QVariant>& parameters) {
    if (image.isNull()) return image;

    const int amount = parameters.value("amount", 0).toInt();
    if (amount == 0) return image;

    const VignetteArgs a = makeArgs(
        amount,
        parameters.value("midpoint",  50).toInt(),
        parameters.value("feather",   50).toInt(),
        parameters.value("roundness",  0).toInt());

    if (image.format() == QImage::Format_RGBX64)
        return processImageGPU16(image, a);
    return processImageGPU(image, a);
}
