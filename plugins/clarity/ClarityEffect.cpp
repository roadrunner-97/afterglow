#include "ClarityEffect.h"
#include "ParamSlider.h"
#include "color_kernels.h"
#include <QDebug>
#include <QVBoxLayout>
#include <algorithm>
#include <mutex>

// ============================================================================
// GPU path (OpenCL)
// Three-pass: horizontal blur → vertical blur → clarity combine.
// Same pipeline shape as UnsharpEffect, but with a large default radius and a
// luminance mask that concentrates the effect on midtones (protects the deep
// shadows and bright highlights from extra contrast / halos).
// ============================================================================

#define CL_HPP_TARGET_OPENCL_VERSION 200
#define CL_HPP_MINIMUM_OPENCL_VERSION 110
#define CL_HPP_ENABLE_EXCEPTIONS
#include <CL/opencl.hpp>
#include "GpuDeviceRegistryOCL.h"
#include "GpuContextBase.h"
#include "blur_kernels.h"

namespace {

// Pixels: QImage::Format_RGB32 (0xFFRRGGBB), stride = bytesPerLine/4
//         QImage::Format_RGBX64 (ushort4),   stride = bytesPerLine/8
// Blur passes (blurH/blurV/blurH16/blurV16) come from the shared header.
// Clarity: result = clamp(orig + amount * mask * (orig − blurred), 0, 1)
//          where mask = max(0, 1 − 2|L − 0.5|) is a triangular tent centred
//          at midtone luminance (Rec.709 weights).  L=0 or L=1 → mask=0 so
//          pure shadows / highlights are untouched; L=0.5 → mask=1.
static const char* GPU_KERNEL_SOURCE = SHARED_BLUR_KERNELS R"CL(

inline float claritMask(float L) {
    return max(0.0f, 1.0f - 2.0f * fabs(L - 0.5f));
}

// 8-bit clarity combine ----------------------------------------------------
__kernel void clarityCombine(__global const uint* original,
                              __global const uint* blurred,
                              __global       uint* output,
                              int stride, int width, int height,
                              float amount)
{
    int x = get_global_id(0), y = get_global_id(1);
    if (x >= width || y >= height) return;

    uint op = original[y * stride + x];
    uint bp = blurred [y * stride + x];

    float or_ = ((op >> 16) & 0xFFu) * (1.0f / 255.0f);
    float og  = ((op >>  8) & 0xFFu) * (1.0f / 255.0f);
    float ob  = ( op        & 0xFFu) * (1.0f / 255.0f);
    float br  = ((bp >> 16) & 0xFFu) * (1.0f / 255.0f);
    float bg  = ((bp >>  8) & 0xFFu) * (1.0f / 255.0f);
    float bb  = ( bp        & 0xFFu) * (1.0f / 255.0f);

    float L = 0.2126f * or_ + 0.7152f * og + 0.0722f * ob;
    float k = amount * claritMask(L);

    float rr = clamp(or_ + k * (or_ - br), 0.0f, 1.0f);
    float rg = clamp(og  + k * (og  - bg), 0.0f, 1.0f);
    float rb = clamp(ob  + k * (ob  - bb), 0.0f, 1.0f);

    output[y * stride + x] = 0xFF000000u
        | ((uint)(rr * 255.0f + 0.5f) << 16)
        | ((uint)(rg * 255.0f + 0.5f) <<  8)
        |  (uint)(rb * 255.0f + 0.5f);
}

// 16-bit clarity combine ---------------------------------------------------
__kernel void clarityCombine16(__global const ushort4* original,
                                __global const ushort4* blurred,
                                __global       ushort4* output,
                                int stride, int width, int height,
                                float amount)
{
    int x = get_global_id(0), y = get_global_id(1);
    if (x >= width || y >= height) return;

    ushort4 op = original[y * stride + x];
    ushort4 bp = blurred [y * stride + x];

    float or_ = op.s0 * (1.0f / 65535.0f);
    float og  = op.s1 * (1.0f / 65535.0f);
    float ob  = op.s2 * (1.0f / 65535.0f);
    float br  = bp.s0 * (1.0f / 65535.0f);
    float bg  = bp.s1 * (1.0f / 65535.0f);
    float bb  = bp.s2 * (1.0f / 65535.0f);

    float L = 0.2126f * or_ + 0.7152f * og + 0.0722f * ob;
    float k = amount * claritMask(L);

    ushort4 res;
    res.s0 = (ushort)(clamp(or_ + k * (or_ - br), 0.0f, 1.0f) * 65535.0f + 0.5f);
    res.s1 = (ushort)(clamp(og  + k * (og  - bg), 0.0f, 1.0f) * 65535.0f + 0.5f);
    res.s2 = (ushort)(clamp(ob  + k * (ob  - bb), 0.0f, 1.0f) * 65535.0f + 0.5f);
    res.s3 = 65535;
    output[y * stride + x] = res;
}

)CL";

struct GpuContext : GpuContextBase<GpuContext> {
    cl::Kernel kernelH;
    cl::Kernel kernelV;
    cl::Kernel kernelClarity;
    cl::Kernel kernelH16;
    cl::Kernel kernelV16;
    cl::Kernel kernelClarity16;

    void init() {
        cl::Device device;
        if (!acquireDevice(device, "Clarity")) return;
        try {
            cl::Program prog(context, GPU_KERNEL_SOURCE);
            prog.build({device});
            kernelH         = cl::Kernel(prog, "blurH");
            kernelV         = cl::Kernel(prog, "blurV");
            kernelClarity   = cl::Kernel(prog, "clarityCombine");
            kernelH16       = cl::Kernel(prog, "blurH16");
            kernelV16       = cl::Kernel(prog, "blurV16");
            kernelClarity16 = cl::Kernel(prog, "clarityCombine16");
            available = true;
            qDebug() << "[GPU] Clarity ready on:"
                     << QString::fromStdString(device.getInfo<CL_DEVICE_NAME>());
        }
        // GCOVR_EXCL_START
        catch (const cl::Error& e) {
            qWarning() << "[GPU] Clarity init failed:" << e.what() << "(err" << e.err() << ")";
        }
        // GCOVR_EXCL_STOP
    }
};

static std::mutex gpuMutex;

static QImage processImageGPU(const QImage& image, int radius, float amount) {
    QImage src = image.convertToFormat(QImage::Format_RGB32);
    const int    width    = src.width();
    const int    height   = src.height();
    const int    stride   = src.bytesPerLine() / 4;
    const size_t bufBytes = static_cast<size_t>(src.bytesPerLine()) * height;

    try {
        std::lock_guard<std::mutex> lock(gpuMutex);
        GpuContext& gpu = GpuContext::instance();
        if (!gpu.available) return {};

        // Upload once; GPU-copy to bufOrig so the combine pass has the original.
        cl::Buffer bufA   (gpu.context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, bufBytes, src.bits());
        cl::Buffer bufOrig(gpu.context, CL_MEM_READ_WRITE, bufBytes);
        cl::Buffer bufB   (gpu.context, CL_MEM_READ_WRITE, bufBytes);
        gpu.queue.enqueueCopyBuffer(bufA, bufOrig, 0, 0, bufBytes);

        const cl::NDRange global(width, height);

        // Horizontal blur: A → B
        gpu.kernelH.setArg(0, bufA); gpu.kernelH.setArg(1, bufB);
        gpu.kernelH.setArg(2, stride); gpu.kernelH.setArg(3, width);
        gpu.kernelH.setArg(4, height); gpu.kernelH.setArg(5, radius);
        gpu.queue.enqueueNDRangeKernel(gpu.kernelH, cl::NullRange, global, cl::NullRange);
        gpu.queue.finish();

        // Vertical blur: B → A  (A now holds the blurred image)
        gpu.kernelV.setArg(0, bufB); gpu.kernelV.setArg(1, bufA);
        gpu.kernelV.setArg(2, stride); gpu.kernelV.setArg(3, width);
        gpu.kernelV.setArg(4, height); gpu.kernelV.setArg(5, radius);
        gpu.queue.enqueueNDRangeKernel(gpu.kernelV, cl::NullRange, global, cl::NullRange);
        gpu.queue.finish();

        // Clarity combine: (original=bufOrig, blurred=bufA) → bufB
        gpu.kernelClarity.setArg(0, bufOrig); gpu.kernelClarity.setArg(1, bufA);
        gpu.kernelClarity.setArg(2, bufB);
        gpu.kernelClarity.setArg(3, stride); gpu.kernelClarity.setArg(4, width);
        gpu.kernelClarity.setArg(5, height);
        gpu.kernelClarity.setArg(6, amount);
        gpu.queue.enqueueNDRangeKernel(gpu.kernelClarity, cl::NullRange, global, cl::NullRange);
        gpu.queue.finish();

        gpu.queue.enqueueReadBuffer(bufB, CL_TRUE, 0, bufBytes, src.bits());
    }
    // GCOVR_EXCL_START
    catch (const cl::Error& e) {
        qWarning() << "[GPU] Clarity kernel failed:" << e.what() << "(err" << e.err() << ")";
        return {};
    }
    // GCOVR_EXCL_STOP
    return src;
}

static QImage processImageGPU16(const QImage& image, int radius, float amount) {
    QImage src = image; // already RGBX64
    const int    width    = src.width();
    const int    height   = src.height();
    const int    stride   = src.bytesPerLine() / 8;
    const size_t bufBytes = static_cast<size_t>(src.bytesPerLine()) * height;

    try {
        std::lock_guard<std::mutex> lock(gpuMutex);
        GpuContext& gpu = GpuContext::instance();
        if (!gpu.available) return {};

        cl::Buffer bufA   (gpu.context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, bufBytes, src.bits());
        cl::Buffer bufOrig(gpu.context, CL_MEM_READ_WRITE, bufBytes);
        cl::Buffer bufB   (gpu.context, CL_MEM_READ_WRITE, bufBytes);
        gpu.queue.enqueueCopyBuffer(bufA, bufOrig, 0, 0, bufBytes);

        const cl::NDRange global(width, height);

        // Horizontal blur: A → B
        gpu.kernelH16.setArg(0, bufA); gpu.kernelH16.setArg(1, bufB);
        gpu.kernelH16.setArg(2, stride); gpu.kernelH16.setArg(3, width);
        gpu.kernelH16.setArg(4, height); gpu.kernelH16.setArg(5, radius);
        gpu.queue.enqueueNDRangeKernel(gpu.kernelH16, cl::NullRange, global, cl::NullRange);
        gpu.queue.finish();

        // Vertical blur: B → A
        gpu.kernelV16.setArg(0, bufB); gpu.kernelV16.setArg(1, bufA);
        gpu.kernelV16.setArg(2, stride); gpu.kernelV16.setArg(3, width);
        gpu.kernelV16.setArg(4, height); gpu.kernelV16.setArg(5, radius);
        gpu.queue.enqueueNDRangeKernel(gpu.kernelV16, cl::NullRange, global, cl::NullRange);
        gpu.queue.finish();

        // Clarity combine: (original=bufOrig, blurred=bufA) → bufB
        gpu.kernelClarity16.setArg(0, bufOrig); gpu.kernelClarity16.setArg(1, bufA);
        gpu.kernelClarity16.setArg(2, bufB);
        gpu.kernelClarity16.setArg(3, stride); gpu.kernelClarity16.setArg(4, width);
        gpu.kernelClarity16.setArg(5, height);
        gpu.kernelClarity16.setArg(6, amount);
        gpu.queue.enqueueNDRangeKernel(gpu.kernelClarity16, cl::NullRange, global, cl::NullRange);
        gpu.queue.finish();

        gpu.queue.enqueueReadBuffer(bufB, CL_TRUE, 0, bufBytes, src.bits());
    }
    // GCOVR_EXCL_START
    catch (const cl::Error& e) {
        qWarning() << "[GPU] Clarity16 kernel failed:" << e.what() << "(err" << e.err() << ")";
        return {};
    }
    // GCOVR_EXCL_STOP
    return src;
}

} // namespace

// ============================================================================
// Pipeline kernel — float4 linear sRGB.  Blur happens in linear; the midtone
// mask is evaluated on sRGB-encoded luminance so its triangular tent
// (centred at 0.5 perceptual grey) behaves like the test path.  We then
// modulate the original linear RGB by the midtone weight * detail to preserve
// hue.  Flow: H(buf→aux) → V(aux→blurBuf) → combine(buf+blurBuf → aux) →
// copy(aux→buf).
// ============================================================================
static const char* PIPELINE_KERNEL_SOURCE = COLOR_KERNELS_SRC SHARED_BLUR_KERNELS_F4 R"CL(

inline float claritMask(float L) {
    return max(0.0f, 1.0f - 2.0f * fabs(L - 0.5f));
}

__kernel void clarityCombineLinear(__global const float4* original,
                                    __global const float4* blurred,
                                    __global       float4* output,
                                    int w, int h,
                                    float amount)
{
    int x = get_global_id(0), y = get_global_id(1);
    if (x >= w || y >= h) return;

    float4 o = original[y * w + x];
    float4 b = blurred [y * w + x];

    // Midtone mask evaluated in sRGB-perceptual space so the tent matches
    // today's behaviour (L=0.5 in sRGB ≈ 0.214 in linear — very different).
    float L_linear = linear_luma(o);
    float L_srgb   = linear_to_srgb(L_linear);
    float k = amount * claritMask(L_srgb);

    // Detail in linear light; unclamped (pack kernel clamps once at the end).
    float r  = o.x + k * (o.x - b.x);
    float g  = o.y + k * (o.y - b.y);
    float bl = o.z + k * (o.z - b.z);
    output[y * w + x] = (float4)(r, g, bl, 1.0f);
}

)CL";

// ============================================================================
// IGpuEffect — shared pipeline interface
// ============================================================================

bool ClarityEffect::initGpuKernels(cl::Context& ctx, cl::Device& dev) {
    try {
        cl::Program prog(ctx, PIPELINE_KERNEL_SOURCE);
        prog.build({dev});
        m_kernelBlurHLinear   = cl::Kernel(prog, "blurHLinear");
        m_kernelBlurVLinear   = cl::Kernel(prog, "blurVLinear");
        m_kernelClarityLinear = cl::Kernel(prog, "clarityCombineLinear");
        m_pipelineCtx         = ctx;
        m_blurBuf             = cl::Buffer();
        m_blurBufW = m_blurBufH = 0;
        return true;
    }
    // GCOVR_EXCL_START
    catch (const cl::Error& e) {
        qWarning() << "[GpuPipeline] Clarity initGpuKernels failed:"
                   << e.what() << "(err" << e.err() << ")";
        return false;
    }
    // GCOVR_EXCL_STOP
}

bool ClarityEffect::enqueueGpu(cl::CommandQueue& queue,
                                cl::Buffer& buf, cl::Buffer& aux,
                                int w, int h,
                                const QMap<QString, QVariant>& params) {
    const int amountPct  = params.value("amount", 0).toInt();
    const int radiusSrc  = params.value("radius", 30).toInt();
    if (amountPct == 0 || radiusSrc == 0) return true;

    const float amount = amountPct / 100.0f;

    // Scale blur radius from source pixels to preview pixels.
    const double scale = params.value("_srcPixelsPerPreviewPixel", 1.0).toDouble();
    int radius = std::max(1, static_cast<int>(radiusSrc / std::max(scale, 1e-6) + 0.5));

    const size_t f4Bytes = static_cast<size_t>(w) * h * sizeof(cl_float4);
    if (m_blurBufW != w || m_blurBufH != h) {
        m_blurBuf  = cl::Buffer(m_pipelineCtx, CL_MEM_READ_WRITE, f4Bytes);
        m_blurBufW = w;
        m_blurBufH = h;
    }

    const cl::NDRange global(w, h);

    // H blur: buf → aux
    m_kernelBlurHLinear.setArg(0, buf);
    m_kernelBlurHLinear.setArg(1, aux);
    m_kernelBlurHLinear.setArg(2, w);
    m_kernelBlurHLinear.setArg(3, h);
    m_kernelBlurHLinear.setArg(4, radius);
    m_kernelBlurHLinear.setArg(5, 1); // isGaussian
    queue.enqueueNDRangeKernel(m_kernelBlurHLinear, cl::NullRange, global, cl::NullRange);

    // V blur: aux → m_blurBuf  (buf still holds the unmodified original)
    m_kernelBlurVLinear.setArg(0, aux);
    m_kernelBlurVLinear.setArg(1, m_blurBuf);
    m_kernelBlurVLinear.setArg(2, w);
    m_kernelBlurVLinear.setArg(3, h);
    m_kernelBlurVLinear.setArg(4, radius);
    m_kernelBlurVLinear.setArg(5, 1);
    queue.enqueueNDRangeKernel(m_kernelBlurVLinear, cl::NullRange, global, cl::NullRange);

    // Combine: (original=buf, blurred=m_blurBuf) → aux
    m_kernelClarityLinear.setArg(0, buf);
    m_kernelClarityLinear.setArg(1, m_blurBuf);
    m_kernelClarityLinear.setArg(2, aux);
    m_kernelClarityLinear.setArg(3, w);
    m_kernelClarityLinear.setArg(4, h);
    m_kernelClarityLinear.setArg(5, amount);
    queue.enqueueNDRangeKernel(m_kernelClarityLinear, cl::NullRange, global, cl::NullRange);

    // Final result must live in buf.
    queue.enqueueCopyBuffer(aux, buf, 0, 0, f4Bytes);
    return true;
}

// ============================================================================
// Effect implementation
// ============================================================================

ClarityEffect::ClarityEffect()
    : controlsWidget(nullptr), amountParam(nullptr), radiusParam(nullptr) {
}

ClarityEffect::~ClarityEffect() {
}

QString ClarityEffect::getName() const        { return "Clarity"; }
QString ClarityEffect::getDescription() const { return "Local midtone contrast enhancement"; }
QString ClarityEffect::getVersion() const     { return "1.0.0"; }

bool ClarityEffect::initialize() {
    qDebug() << "Clarity effect initialized";
    return true;
}

QWidget* ClarityEffect::createControlsWidget() {
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
    amountParam->setToolTip("Midtone local-contrast strength.\nPositive values add \"pop\" to midtones; negative values soften them for a dreamy look.");
    connectSlider(amountParam);
    layout->addWidget(amountParam);

    radiusParam = new ParamSlider("Radius", 10, 100);
    radiusParam->setValue(30);
    radiusParam->setToolTip("Radius (source pixels) of the blur used to define \"local\".\nLarger values affect broader tonal regions; smaller values feel closer to sharpening.");
    connectSlider(radiusParam);
    layout->addWidget(radiusParam);

    layout->addStretch();
    return controlsWidget;
}

QMap<QString, QVariant> ClarityEffect::getParameters() const {
    QMap<QString, QVariant> params;
    params["amount"] = static_cast<int>(amountParam ? amountParam->value() : 0.0);
    params["radius"] = static_cast<int>(radiusParam ? radiusParam->value() : 30.0);
    return params;
}

QImage ClarityEffect::processImage(const QImage& image, const QMap<QString, QVariant>& parameters) {
    if (image.isNull()) return image;

    const int amountPct = parameters.value("amount", 0).toInt();
    const int radius    = parameters.value("radius", 30).toInt();
    if (amountPct == 0 || radius == 0) return image;

    const float amount = amountPct / 100.0f;

    if (image.format() == QImage::Format_RGBX64)
        return processImageGPU16(image, radius, amount);
    return processImageGPU(image, radius, amount);
}
