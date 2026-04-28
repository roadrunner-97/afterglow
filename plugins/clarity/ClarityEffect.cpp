#include "ClarityEffect.h"
#include "ParamSlider.h"
#include "color_kernels.h"
#include <QDebug>
#include <QVBoxLayout>
#include <algorithm>

// ============================================================================
// Three-pass: horizontal blur → vertical blur → clarity combine.
// Same pipeline shape as UnsharpEffect, but with a large default radius and a
// luminance mask that concentrates the effect on midtones (protects the deep
// shadows and bright highlights from extra contrast / halos).

#include "blur_kernels.h"

namespace {

// Pixels: QImage::Format_RGB32 (0xFFRRGGBB), stride = bytesPerLine/4
//         QImage::Format_RGBX64 (ushort4),   stride = bytesPerLine/8
// Blur passes (blurH/blurV/blurH16/blurV16) come from the shared header.
// Clarity: result = clamp(orig + amount * mask * (orig − blurred), 0, 1)
//          where mask = max(0, 1 − 2|L − 0.5|) is a triangular tent centred
//          at midtone luminance (Rec.709 weights).  L=0 or L=1 → mask=0 so
//          pure shadows / highlights are untouched; L=0.5 → mask=1.

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

void ClarityEffect::applyParameters(const QMap<QString, QVariant>& parameters) {
    if (amountParam && parameters.contains("amount"))
        amountParam->setValue(parameters.value("amount").toDouble());
    if (radiusParam && parameters.contains("radius"))
        radiusParam->setValue(parameters.value("radius").toDouble());
    emit parametersChanged();
}

