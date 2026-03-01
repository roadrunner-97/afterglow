#include "UnsharpEffect.h"
#include "ParamSlider.h"
#include <QDebug>
#include <QVBoxLayout>
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

namespace {

// Pixels: QImage::Format_RGB32 (0xFFRRGGBB), stride = bytesPerLine/4.
// Blur passes use Gaussian weights (sigma = radius/3).
// Unsharp: result = clamp(orig + amount*(orig − blurred), 0, 255)
//          skipped (passes through original) when all channel diffs < threshold.
static const char* GPU_KERNEL_SOURCE = R"CL(

// 8-bit path ---------------------------------------------------------------
__kernel void blurH(__global const uint* in, __global uint* out,
                    int stride, int width, int height, int radius)
{
    int x = get_global_id(0), y = get_global_id(1);
    if (x >= width || y >= height) return;

    float sigma = max((float)radius / 3.0f, 0.5f);
    float r = 0, g = 0, b = 0, wsum = 0;
    for (int dx = -radius; dx <= radius; dx++) {
        int sx = clamp(x + dx, 0, width - 1);
        uint p = in[y * stride + sx];
        float w = exp(-0.5f * dx * dx / (sigma * sigma));
        r += w * ((p >> 16) & 0xFFu);
        g += w * ((p >>  8) & 0xFFu);
        b += w * ( p        & 0xFFu);
        wsum += w;
    }
    uint ri = (uint)(r/wsum + 0.5f), gi = (uint)(g/wsum + 0.5f), bi = (uint)(b/wsum + 0.5f);
    out[y * stride + x] = 0xFF000000u | (ri << 16) | (gi << 8) | bi;
}

__kernel void blurV(__global const uint* in, __global uint* out,
                    int stride, int width, int height, int radius)
{
    int x = get_global_id(0), y = get_global_id(1);
    if (x >= width || y >= height) return;

    float sigma = max((float)radius / 3.0f, 0.5f);
    float r = 0, g = 0, b = 0, wsum = 0;
    for (int dy = -radius; dy <= radius; dy++) {
        int sy = clamp(y + dy, 0, height - 1);
        uint p = in[sy * stride + x];
        float w = exp(-0.5f * dy * dy / (sigma * sigma));
        r += w * ((p >> 16) & 0xFFu);
        g += w * ((p >>  8) & 0xFFu);
        b += w * ( p        & 0xFFu);
        wsum += w;
    }
    uint ri = (uint)(r/wsum + 0.5f), gi = (uint)(g/wsum + 0.5f), bi = (uint)(b/wsum + 0.5f);
    out[y * stride + x] = 0xFF000000u | (ri << 16) | (gi << 8) | bi;
}

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

// 16-bit path --------------------------------------------------------------
// pixels are QImage::Format_RGBX64 (ushort4 per pixel).
// On little-endian: ushort4.s0=R, .s1=G, .s2=B, .s3=A
// stride = bytesPerLine / 8

__kernel void blurH16(__global const ushort4* in, __global ushort4* out,
                      int stride, int width, int height, int radius)
{
    int x = get_global_id(0), y = get_global_id(1);
    if (x >= width || y >= height) return;

    float sigma = max((float)radius / 3.0f, 0.5f);
    float r = 0, g = 0, b = 0, wsum = 0;
    for (int dx = -radius; dx <= radius; dx++) {
        int sx = clamp(x + dx, 0, width - 1);
        ushort4 p = in[y * stride + sx];
        float w = exp(-0.5f * dx * dx / (sigma * sigma));
        r += w * p.s0; g += w * p.s1; b += w * p.s2;
        wsum += w;
    }
    ushort4 o;
    o.s0 = (ushort)(r/wsum + 0.5f);
    o.s1 = (ushort)(g/wsum + 0.5f);
    o.s2 = (ushort)(b/wsum + 0.5f);
    o.s3 = 65535;
    out[y * stride + x] = o;
}

__kernel void blurV16(__global const ushort4* in, __global ushort4* out,
                      int stride, int width, int height, int radius)
{
    int x = get_global_id(0), y = get_global_id(1);
    if (x >= width || y >= height) return;

    float sigma = max((float)radius / 3.0f, 0.5f);
    float r = 0, g = 0, b = 0, wsum = 0;
    for (int dy = -radius; dy <= radius; dy++) {
        int sy = clamp(y + dy, 0, height - 1);
        ushort4 p = in[sy * stride + x];
        float w = exp(-0.5f * dy * dy / (sigma * sigma));
        r += w * p.s0; g += w * p.s1; b += w * p.s2;
        wsum += w;
    }
    ushort4 o;
    o.s0 = (ushort)(r/wsum + 0.5f);
    o.s1 = (ushort)(g/wsum + 0.5f);
    o.s2 = (ushort)(b/wsum + 0.5f);
    o.s3 = 65535;
    out[y * stride + x] = o;
}

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

struct GpuContext {
    cl::Context      context;
    cl::CommandQueue queue;
    cl::Kernel       kernelH;
    cl::Kernel       kernelV;
    cl::Kernel       kernelUnsharp;
    cl::Kernel       kernelH16;
    cl::Kernel       kernelV16;
    cl::Kernel       kernelUnsharp16;
    bool             available  = false;
    int              m_revision = 0;

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
            qWarning() << "[GPU] Unsharp: no OpenCL device available";
            return;
        }
        try {
            context = cl::Context(device);
            queue   = cl::CommandQueue(context, device);
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
        } catch (const cl::Error& e) {
            qWarning() << "[GPU] Unsharp init failed:" << e.what() << "(err" << e.err() << ")";
        }
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
    } catch (const cl::Error& e) {
        qWarning() << "[GPU] Unsharp kernel failed:" << e.what() << "(err" << e.err() << ")";
        return {};
    }
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
    } catch (const cl::Error& e) {
        qWarning() << "[GPU] Unsharp16 kernel failed:" << e.what() << "(err" << e.err() << ")";
        return {};
    }
    return src;
}

} // namespace

// ============================================================================
// IGpuEffect — shared pipeline interface
// ============================================================================

bool UnsharpEffect::initGpuKernels(cl::Context& ctx, cl::Device& dev) {
    try {
        cl::Program prog(ctx, GPU_KERNEL_SOURCE);
        prog.build({dev});
        m_kernelH         = cl::Kernel(prog, "blurH");
        m_kernelV         = cl::Kernel(prog, "blurV");
        m_kernelUnsharp   = cl::Kernel(prog, "unsharpCombine");
        m_kernelH16       = cl::Kernel(prog, "blurH16");
        m_kernelV16       = cl::Kernel(prog, "blurV16");
        m_kernelUnsharp16 = cl::Kernel(prog, "unsharpCombine16");
        m_pipelineCtx     = ctx;  // save for temp buffer allocation in enqueueGpu
        return true;
    } catch (const cl::Error& e) {
        qWarning() << "[GpuPipeline] Unsharp initGpuKernels failed:"
                   << e.what() << "(err" << e.err() << ")";
        return false;
    }
}

bool UnsharpEffect::enqueueGpu(cl::CommandQueue& queue,
                                cl::Buffer& buf, cl::Buffer& aux,
                                int w, int h, int stride, bool is16bit,
                                const QMap<QString, QVariant>& params) {
    const float amount    = static_cast<float>(params.value("amount",    1.0).toDouble());
    const int   radius    = params.value("radius",    2).toInt();
    const int   threshold = params.value("threshold", 3).toInt();

    if (amount == 0.0f || radius == 0) return true;  // no-op

    // Allocate a third GPU buffer (no host copy — just GPU memory, very fast).
    // Layout:
    //   buf     = input image (preserved as "original" for the combine step)
    //   aux     = H-blur output, then the sharpened result
    //   blurBuf = V-blur output (= blurred image used in combine)
    const size_t bufBytes = static_cast<size_t>(stride) * (is16bit ? 8 : 4) * h;
    cl::Buffer blurBuf(m_pipelineCtx, CL_MEM_READ_WRITE, bufBytes);

    const cl::NDRange global(w, h);

    cl::Kernel& kH  = is16bit ? m_kernelH16 : m_kernelH;
    cl::Kernel& kV  = is16bit ? m_kernelV16 : m_kernelV;
    cl::Kernel& kUS = is16bit ? m_kernelUnsharp16 : m_kernelUnsharp;

    // Step 1: H blur — buf → aux
    kH.setArg(0, buf); kH.setArg(1, aux);
    kH.setArg(2, stride); kH.setArg(3, w); kH.setArg(4, h); kH.setArg(5, radius);
    queue.enqueueNDRangeKernel(kH, cl::NullRange, global, cl::NullRange);

    // Step 2: V blur — aux → blurBuf  (blurBuf = blurred image)
    kV.setArg(0, aux); kV.setArg(1, blurBuf);
    kV.setArg(2, stride); kV.setArg(3, w); kV.setArg(4, h); kV.setArg(5, radius);
    queue.enqueueNDRangeKernel(kV, cl::NullRange, global, cl::NullRange);

    // Step 3: Unsharp combine — (original=buf, blurred=blurBuf) → aux
    kUS.setArg(0, buf); kUS.setArg(1, blurBuf); kUS.setArg(2, aux);
    kUS.setArg(3, stride); kUS.setArg(4, w); kUS.setArg(5, h);
    kUS.setArg(6, amount); kUS.setArg(7, threshold);
    queue.enqueueNDRangeKernel(kUS, cl::NullRange, global, cl::NullRange);

    // Step 4: Copy result (aux) → working buffer (buf)
    queue.enqueueCopyBuffer(aux, buf, 0, 0, bufBytes);

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
