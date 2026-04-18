#include "UnsharpEffect.h"
#include "ParamSlider.h"
#include "color_kernels.h"
#include <QDebug>
#include <QVBoxLayout>
#include <algorithm>
#include <mutex>

// ============================================================================
// GPU path (OpenCL)
// Three-pass: horizontal blur → vertical blur → unsharp combine
// ============================================================================

#define CL_HPP_TARGET_OPENCL_VERSION 200
#define CL_HPP_MINIMUM_OPENCL_VERSION 110
#define CL_HPP_ENABLE_EXCEPTIONS
#include <CL/opencl.hpp>
#include "GpuDeviceRegistryOCL.h"
#include "GpuContextBase.h"
#include "blur_kernels.h"

namespace {

// Pixels: QImage::Format_RGB32 (0xFFRRGGBB), stride = bytesPerLine/4.
// Blur passes (blurH/blurV/blurH16/blurV16) come from the shared header.
// Unsharp: result = clamp(orig + amount*(orig − blurred), 0, 255)
//          skipped (passes through original) when all channel diffs < threshold.
static const char* GPU_KERNEL_SOURCE = SHARED_BLUR_KERNELS R"CL(

// 8-bit unsharp combine ----------------------------------------------------
__kernel void unsharpCombine(__global const uint* original,
                              __global const uint* blurred,
                              __global       uint* output,
                              int stride, int width, int height,
                              float amount, int threshold)
{
    int x = get_global_id(0), y = get_global_id(1);
    if (x >= width || y >= height) return;

    uint op = original[y * stride + x];
    uint bp = blurred [y * stride + x];

    int or_ = (op >> 16) & 0xFF, og = (op >> 8) & 0xFF, ob = op & 0xFF;
    int br  = (bp >> 16) & 0xFF, bg = (bp >> 8) & 0xFF, bb = bp & 0xFF;

    int dr = or_ - br, dg = og - bg, db = ob - bb;

    // Skip if all differences are below the threshold (avoids sharpening noise)
    if (abs(dr) < threshold && abs(dg) < threshold && abs(db) < threshold) {
        output[y * stride + x] = op;
        return;
    }

    int rr = clamp((int)((float)or_ + amount * (float)dr), 0, 255);
    int rg = clamp((int)((float)og + amount * (float)dg), 0, 255);
    int rb = clamp((int)((float)ob + amount * (float)db), 0, 255);
    output[y * stride + x] = 0xFF000000u | (rr << 16) | (rg << 8) | rb;
}

// 16-bit unsharp combine ---------------------------------------------------
__kernel void unsharpCombine16(__global const ushort4* original,
                                __global const ushort4* blurred,
                                __global       ushort4* output,
                                int stride, int width, int height,
                                float amount, int threshold)
{
    int x = get_global_id(0), y = get_global_id(1);
    if (x >= width || y >= height) return;

    ushort4 op = original[y * stride + x];
    ushort4 bp = blurred [y * stride + x];

    // Scale threshold to 16-bit range
    int thr16 = threshold * 257;

    int or_ = op.s0, og = op.s1, ob = op.s2;
    int br  = bp.s0, bg = bp.s1, bb = bp.s2;
    int dr  = or_ - br, dg = og - bg, db = ob - bb;

    if (abs(dr) < thr16 && abs(dg) < thr16 && abs(db) < thr16) {
        output[y * stride + x] = op;
        return;
    }

    ushort4 res;
    res.s0 = (ushort)clamp((int)((float)or_ + amount * (float)dr), 0, 65535);
    res.s1 = (ushort)clamp((int)((float)og  + amount * (float)dg), 0, 65535);
    res.s2 = (ushort)clamp((int)((float)ob  + amount * (float)db), 0, 65535);
    res.s3 = 65535;
    output[y * stride + x] = res;
}

)CL";

struct GpuContext : GpuContextBase<GpuContext> {
    cl::Kernel kernelH;
    cl::Kernel kernelV;
    cl::Kernel kernelUnsharp;
    cl::Kernel kernelH16;
    cl::Kernel kernelV16;
    cl::Kernel kernelUnsharp16;

    void init() {
        cl::Device device;
        if (!acquireDevice(device, "Unsharp")) return;
        try {
            cl::Program prog(context, GPU_KERNEL_SOURCE);
            prog.build({device});
            kernelH         = cl::Kernel(prog, "blurH");
            kernelV         = cl::Kernel(prog, "blurV");
            kernelUnsharp   = cl::Kernel(prog, "unsharpCombine");
            kernelH16       = cl::Kernel(prog, "blurH16");
            kernelV16       = cl::Kernel(prog, "blurV16");
            kernelUnsharp16 = cl::Kernel(prog, "unsharpCombine16");
            available = true;
            qDebug() << "[GPU] Unsharp ready on:"
                     << QString::fromStdString(device.getInfo<CL_DEVICE_NAME>());
        }
        // GCOVR_EXCL_START
        catch (const cl::Error& e) {
            qWarning() << "[GPU] Unsharp init failed:" << e.what() << "(err" << e.err() << ")";
        }
        // GCOVR_EXCL_STOP
    }
};

static std::mutex gpuMutex;

static QImage processImageGPU(const QImage& image, int radius, float amount, int threshold) {
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
        // Avoids a second PCIe upload of the same 192MB.
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

        // Unsharp combine: (original=bufOrig, blurred=bufA) → bufB
        gpu.kernelUnsharp.setArg(0, bufOrig); gpu.kernelUnsharp.setArg(1, bufA);
        gpu.kernelUnsharp.setArg(2, bufB);
        gpu.kernelUnsharp.setArg(3, stride); gpu.kernelUnsharp.setArg(4, width);
        gpu.kernelUnsharp.setArg(5, height);
        gpu.kernelUnsharp.setArg(6, amount); gpu.kernelUnsharp.setArg(7, threshold);
        gpu.queue.enqueueNDRangeKernel(gpu.kernelUnsharp, cl::NullRange, global, cl::NullRange);
        gpu.queue.finish();

        gpu.queue.enqueueReadBuffer(bufB, CL_TRUE, 0, bufBytes, src.bits());
    }
    // GCOVR_EXCL_START
    catch (const cl::Error& e) {
        qWarning() << "[GPU] Unsharp kernel failed:" << e.what() << "(err" << e.err() << ")";
        return {};
    }
    // GCOVR_EXCL_STOP
    return src;
}

static QImage processImageGPU16(const QImage& image, int radius, float amount, int threshold) {
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

        // Unsharp combine: (original=bufOrig, blurred=bufA) → bufB
        gpu.kernelUnsharp16.setArg(0, bufOrig); gpu.kernelUnsharp16.setArg(1, bufA);
        gpu.kernelUnsharp16.setArg(2, bufB);
        gpu.kernelUnsharp16.setArg(3, stride); gpu.kernelUnsharp16.setArg(4, width);
        gpu.kernelUnsharp16.setArg(5, height);
        gpu.kernelUnsharp16.setArg(6, amount); gpu.kernelUnsharp16.setArg(7, threshold);
        gpu.queue.enqueueNDRangeKernel(gpu.kernelUnsharp16, cl::NullRange, global, cl::NullRange);
        gpu.queue.finish();

        gpu.queue.enqueueReadBuffer(bufB, CL_TRUE, 0, bufBytes, src.bits());
    }
    // GCOVR_EXCL_START
    catch (const cl::Error& e) {
        qWarning() << "[GPU] Unsharp16 kernel failed:" << e.what() << "(err" << e.err() << ")";
        return {};
    }
    // GCOVR_EXCL_STOP
    return src;
}

} // namespace

// ============================================================================
// Pipeline kernel — float4 linear sRGB.  Blur is performed in linear light;
// the unsharp combine also happens in linear (original + amount * detail).
// The threshold gate is evaluated on *sRGB-encoded* per-channel magnitudes
// so its perceptual behaviour matches the sRGB-space test path.
// Flow: H blur (buf→aux), V blur (aux→blurBuf), combine (orig=buf,
// blurred=blurBuf → aux), copy (aux→buf) so result ends in buf.
// ============================================================================
static const char* PIPELINE_KERNEL_SOURCE = COLOR_KERNELS_SRC SHARED_BLUR_KERNELS_F4 R"CL(

__kernel void unsharpCombineLinear(__global const float4* original,
                                    __global const float4* blurred,
                                    __global       float4* output,
                                    int w, int h,
                                    float amount, float threshold_srgb)
{
    int x = get_global_id(0), y = get_global_id(1);
    if (x >= w || y >= h) return;

    float4 o = original[y * w + x];
    float4 b = blurred [y * w + x];

    // Perceptual threshold test: gate on sRGB-encoded magnitude of the
    // original-vs-blurred differences so it matches slider intuition (the
    // slider unit mirrors the 8-bit threshold's per-channel difference).
    float dr_s = linear_to_srgb(o.x) - linear_to_srgb(b.x);
    float dg_s = linear_to_srgb(o.y) - linear_to_srgb(b.y);
    float db_s = linear_to_srgb(o.z) - linear_to_srgb(b.z);
    if (fabs(dr_s) < threshold_srgb
     && fabs(dg_s) < threshold_srgb
     && fabs(db_s) < threshold_srgb) {
        output[y * w + x] = (float4)(o.x, o.y, o.z, 1.0f);
        return;
    }

    // Combine in linear light.  Don't clamp — the pack kernel clamps once.
    float r = o.x + amount * (o.x - b.x);
    float g = o.y + amount * (o.y - b.y);
    float bl = o.z + amount * (o.z - b.z);
    output[y * w + x] = (float4)(r, g, bl, 1.0f);
}

)CL";

// ============================================================================
// IGpuEffect — shared pipeline interface
// ============================================================================

bool UnsharpEffect::initGpuKernels(cl::Context& ctx, cl::Device& dev) {
    try {
        cl::Program prog(ctx, PIPELINE_KERNEL_SOURCE);
        prog.build({dev});
        m_kernelBlurHLinear   = cl::Kernel(prog, "blurHLinear");
        m_kernelBlurVLinear   = cl::Kernel(prog, "blurVLinear");
        m_kernelUnsharpLinear = cl::Kernel(prog, "unsharpCombineLinear");
        m_pipelineCtx         = ctx;  // save for temp buffer allocation
        m_blurBuf             = cl::Buffer();
        m_blurBufW = m_blurBufH = 0;
        return true;
    }
    // GCOVR_EXCL_START
    catch (const cl::Error& e) {
        qWarning() << "[GpuPipeline] Unsharp initGpuKernels failed:"
                   << e.what() << "(err" << e.err() << ")";
        return false;
    }
    // GCOVR_EXCL_STOP
}

bool UnsharpEffect::enqueueGpu(cl::CommandQueue& queue,
                                cl::Buffer& buf, cl::Buffer& aux,
                                int w, int h,
                                const QMap<QString, QVariant>& params) {
    const float amount       = static_cast<float>(params.value("amount", 1.0).toDouble());
    const int   radiusSrc    = params.value("radius", 2).toInt();
    const int   thresholdInt = params.value("threshold", 3).toInt();
    if (amount == 0.0f || radiusSrc == 0) return true;

    // Scale blur radius from source pixels to preview pixels.
    const double scale = params.value("_srcPixelsPerPreviewPixel", 1.0).toDouble();
    int radius = std::max(1, static_cast<int>(radiusSrc / std::max(scale, 1e-6) + 0.5));

    // Threshold slider is defined in 0..20 of 0..255 sRGB-byte units.  Scale
    // to normalised sRGB [0,1] for comparison with linear_to_srgb() output.
    const float threshold_srgb = thresholdInt / 255.0f;

    // Scratch buffer for the blurred image (reused across calls).
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
    m_kernelUnsharpLinear.setArg(0, buf);
    m_kernelUnsharpLinear.setArg(1, m_blurBuf);
    m_kernelUnsharpLinear.setArg(2, aux);
    m_kernelUnsharpLinear.setArg(3, w);
    m_kernelUnsharpLinear.setArg(4, h);
    m_kernelUnsharpLinear.setArg(5, amount);
    m_kernelUnsharpLinear.setArg(6, threshold_srgb);
    queue.enqueueNDRangeKernel(m_kernelUnsharpLinear, cl::NullRange, global, cl::NullRange);

    // Final result must live in buf.
    queue.enqueueCopyBuffer(aux, buf, 0, 0, f4Bytes);
    return true;
}

// ============================================================================
// Effect implementation
// ============================================================================

UnsharpEffect::UnsharpEffect()
    : controlsWidget(nullptr), amountParam(nullptr),
      radiusParam(nullptr), thresholdParam(nullptr) {
}

UnsharpEffect::~UnsharpEffect() {
}

QString UnsharpEffect::getName() const        { return "Unsharp Mask"; }
QString UnsharpEffect::getDescription() const { return "Sharpens by subtracting a blurred mask"; }
QString UnsharpEffect::getVersion() const     { return "1.0.0"; }

bool UnsharpEffect::initialize() {
    qDebug() << "Unsharp Mask effect initialized";
    return true;
}

QWidget* UnsharpEffect::createControlsWidget() {
    if (controlsWidget) return controlsWidget;

    controlsWidget = new QWidget();
    QVBoxLayout* layout = new QVBoxLayout(controlsWidget);
    layout->setContentsMargins(0, 0, 0, 0);
    layout->setSpacing(8);

    amountParam = new ParamSlider("Amount", 0.0, 5.0, 0.1, 1);
    amountParam->setValue(1.0);
    amountParam->setToolTip("Sharpening strength. 1.0 is a standard boost; values above 2 can produce visible edge halos.");
    connect(amountParam, &ParamSlider::editingFinished, this, [this]() {
        emit parametersChanged();
    });
    connect(amountParam, &ParamSlider::valueChanged, this, [this](double) {
        emit liveParametersChanged();
    });
    layout->addWidget(amountParam);

    radiusParam = new ParamSlider("Radius", 1, 15);
    radiusParam->setValue(2);
    radiusParam->setToolTip("Pixel radius of the blur used to find edges. Larger values sharpen broader features; smaller values target fine detail.");
    connect(radiusParam, &ParamSlider::editingFinished, this, [this]() {
        emit parametersChanged();
    });
    connect(radiusParam, &ParamSlider::valueChanged, this, [this](double) {
        emit liveParametersChanged();
    });
    layout->addWidget(radiusParam);

    thresholdParam = new ParamSlider("Threshold", 0, 20);
    thresholdParam->setValue(3);
    thresholdParam->setToolTip("Minimum per-channel difference required before sharpening is applied. Increase to avoid sharpening noise in flat areas.");
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

QMap<QString, QVariant> UnsharpEffect::getParameters() const {
    QMap<QString, QVariant> params;
    params["amount"]    = amountParam    ? amountParam->value()                       : 1.0;
    params["radius"]    = radiusParam    ? static_cast<int>(radiusParam->value())     : 2;
    params["threshold"] = thresholdParam ? static_cast<int>(thresholdParam->value())  : 3;
    return params;
}

QImage UnsharpEffect::processImage(const QImage& image, const QMap<QString, QVariant>& parameters) {
    if (image.isNull()) return image;

    const float amount    = static_cast<float>(parameters.value("amount",    1.0).toDouble());
    const int   radius    = parameters.value("radius",    2).toInt();
    const int   threshold = parameters.value("threshold", 3).toInt();

    if (amount == 0.0f || radius == 0) return image;

    if (image.format() == QImage::Format_RGBX64)
        return processImageGPU16(image, radius, amount, threshold);
    return processImageGPU(image, radius, amount, threshold);
}
