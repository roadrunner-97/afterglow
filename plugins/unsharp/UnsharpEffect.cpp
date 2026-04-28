#include "UnsharpEffect.h"
#include "ParamSlider.h"
#include "color_kernels.h"
#include <QDebug>
#include <QVBoxLayout>
#include <algorithm>

// Three-pass: horizontal blur → vertical blur → unsharp combine

#include "blur_kernels.h"

namespace {

// Pixels: QImage::Format_RGB32 (0xFFRRGGBB), stride = bytesPerLine/4.
// Blur passes (blurH/blurV/blurH16/blurV16) come from the shared header.
// Unsharp: result = clamp(orig + amount*(orig − blurred), 0, 255)
//          skipped (passes through original) when all channel diffs < threshold.

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

void UnsharpEffect::applyParameters(const QMap<QString, QVariant>& parameters) {
    if (amountParam && parameters.contains("amount"))
        amountParam->setValue(parameters.value("amount").toDouble());
    if (radiusParam && parameters.contains("radius"))
        radiusParam->setValue(parameters.value("radius").toDouble());
    if (thresholdParam && parameters.contains("threshold"))
        thresholdParam->setValue(parameters.value("threshold").toDouble());
    emit parametersChanged();
}

