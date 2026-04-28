#include "SplitToningEffect.h"
#include "ParamSlider.h"
#include "color_kernels.h"
#include <QDebug>
#include <QVBoxLayout>


namespace {

// Split-toning: two hue/saturation pairs tint the image's dark and bright
// regions respectively, with a balance slider moving the crossover.
//
//   shadowMask    = clamp((1 - L) - balance*0.5, 0, 1)   peaks in dark pixels
//   highlightMask = clamp(L       + balance*0.5, 0, 1)   peaks in bright pixels
//
// balance = -1 shifts both masks toward shadows (shadow tint dominates);
// balance = +1 shifts both toward highlights.  At balance=0 the masks cross
// at L=0.5.  The target color for each region is the pure-hue RGB scaled by
// the source pixel's luminance, so pure black stays black (zero scales to
// zero) and the tint strength grows with pixel brightness — a convention
// consistent with Lightroom/darktable's split-toning behaviour.

struct SplitArgs {
    float shadowHue;     // [0, 1)
    float shadowSat;     // [0, 1]
    float highlightHue;  // [0, 1)
    float highlightSat;  // [0, 1]
    float balance;       // [-1, 1]
};

static SplitArgs makeArgs(int shadowHueDeg, int shadowSatPct,
                           int highlightHueDeg, int highlightSatPct,
                           int balancePct) {
    SplitArgs a;
    a.shadowHue    = (((shadowHueDeg    % 360) + 360) % 360) / 360.0f;
    a.shadowSat    = shadowSatPct    / 100.0f;
    a.highlightHue = (((highlightHueDeg % 360) + 360) % 360) / 360.0f;
    a.highlightSat = highlightSatPct / 100.0f;
    a.balance      = balancePct      / 100.0f;
    return a;
}

} // namespace

// ============================================================================
// Pipeline kernel — float4 linear sRGB, used by GpuPipeline via enqueueGpu.
// Zone selection (shadow/highlight masks) is on perceptually-placed luminance
// (sRGB-encoded L) so the balance crossover matches the UI expectation.  The
// tint RGB is specified in sRGB (slider = hue wheel), so it is decoded to
// linear before being mixed into the linear pixel.  The tint-intensity
// multiplier uses linear luma so pure black stays black.
// ============================================================================
static const char* PIPELINE_KERNEL_SOURCE = COLOR_KERNELS_SRC R"CL(
// Pure-hue RGB at V=1, S=1 in sRGB space.  Input hue is in [0,1).
inline float3 hueToRgb(float h) {
    float r = fabs(h * 6.0f - 3.0f) - 1.0f;
    float g = 2.0f - fabs(h * 6.0f - 2.0f);
    float b = 2.0f - fabs(h * 6.0f - 4.0f);
    return clamp((float3)(r, g, b), 0.0f, 1.0f);
}

__kernel void applySplitToningLinear(__global float4* pixels,
                                      int   w,
                                      int   h,
                                      float shadowHue,
                                      float shadowSat,
                                      float highlightHue,
                                      float highlightSat,
                                      float balance)
{
    int x = get_global_id(0);
    int y = get_global_id(1);
    if (x >= w || y >= h) return;

    float4 px = pixels[y * w + x];

    // Mask selection uses perceptual L (sRGB-encoded luma).
    float linLum = linear_luma(px);
    float Lp     = linear_to_srgb(linLum);

    float shadowMask    = clamp((1.0f - Lp) - balance * 0.5f, 0.0f, 1.0f);
    float highlightMask = clamp(Lp          + balance * 0.5f, 0.0f, 1.0f);

    // Tint colours are defined on the sRGB hue wheel — decode to linear so
    // the blend stays physically meaningful in linear light.
    float3 sT_srgb = hueToRgb(shadowHue);
    float3 hT_srgb = hueToRgb(highlightHue);
    float3 sT = (float3)(srgb_to_linear(sT_srgb.x),
                         srgb_to_linear(sT_srgb.y),
                         srgb_to_linear(sT_srgb.z));
    float3 hT = (float3)(srgb_to_linear(hT_srgb.x),
                         srgb_to_linear(hT_srgb.y),
                         srgb_to_linear(hT_srgb.z));

    float sStr = shadowSat    * shadowMask;
    float hStr = highlightSat * highlightMask;

    // Tint intensity scales with linear luma so pure black stays black (the
    // tint contribution vanishes for zero-luma pixels).
    float3 rgb = (float3)(px.x, px.y, px.z);
    rgb = rgb * (1.0f - sStr) + sT * linLum * sStr;
    rgb = rgb * (1.0f - hStr) + hT * linLum * hStr;

    // Do not clamp — the final pack kernel clamps once.
    pixels[y * w + x] = (float4)(rgb.x, rgb.y, rgb.z, 1.0f);
}
)CL";

// ============================================================================
// SplitToningEffect
// ============================================================================

SplitToningEffect::SplitToningEffect()
    : controlsWidget(nullptr),
      shadowHueParam(nullptr), shadowSatParam(nullptr),
      highlightHueParam(nullptr), highlightSatParam(nullptr),
      balanceParam(nullptr) {}

SplitToningEffect::~SplitToningEffect() {}

QString SplitToningEffect::getName() const { return "Split Toning"; }
QString SplitToningEffect::getDescription() const {
    return "Independently tint shadows and highlights";
}
QString SplitToningEffect::getVersion() const { return "1.0.0"; }

bool SplitToningEffect::initialize() {
    qDebug() << "Split Toning effect initialized";
    return true;
}

QWidget* SplitToningEffect::createControlsWidget() {
    if (controlsWidget) return controlsWidget;

    controlsWidget = new QWidget();
    QVBoxLayout* layout = new QVBoxLayout(controlsWidget);
    layout->setContentsMargins(0, 0, 0, 0);
    layout->setSpacing(8);

    auto connectSlider = [&](ParamSlider* s) {
        connect(s, &ParamSlider::editingFinished, this, [this]() { emit parametersChanged(); });
        connect(s, &ParamSlider::valueChanged,    this, [this](double) { emit liveParametersChanged(); });
    };

    shadowHueParam = new ParamSlider("Shadow Hue", 0, 359);
    shadowHueParam->setToolTip("Hue angle (degrees) of the tint applied to the darker tones.\n0=red, 60=yellow, 120=green, 180=cyan, 240=blue, 300=magenta.");
    connectSlider(shadowHueParam);
    layout->addWidget(shadowHueParam);

    shadowSatParam = new ParamSlider("Shadow Saturation", 0, 100);
    shadowSatParam->setToolTip("Strength of the shadow tint.\n0 leaves shadows untouched; 100 fully tints them at the chosen hue.");
    connectSlider(shadowSatParam);
    layout->addWidget(shadowSatParam);

    highlightHueParam = new ParamSlider("Highlight Hue", 0, 359);
    highlightHueParam->setToolTip("Hue angle (degrees) of the tint applied to the brighter tones.\n0=red, 60=yellow, 120=green, 180=cyan, 240=blue, 300=magenta.");
    connectSlider(highlightHueParam);
    layout->addWidget(highlightHueParam);

    highlightSatParam = new ParamSlider("Highlight Saturation", 0, 100);
    highlightSatParam->setToolTip("Strength of the highlight tint.\n0 leaves highlights untouched; 100 fully tints them at the chosen hue.");
    connectSlider(highlightSatParam);
    layout->addWidget(highlightSatParam);

    balanceParam = new ParamSlider("Balance", -100, 100);
    balanceParam->setToolTip("Shifts the crossover between shadow and highlight tints.\nNegative values favour the shadow tint; positive values favour the highlight tint.");
    connectSlider(balanceParam);
    layout->addWidget(balanceParam);

    layout->addStretch();
    return controlsWidget;
}

QMap<QString, QVariant> SplitToningEffect::getParameters() const {
    QMap<QString, QVariant> params;
    params["shadowHue"]    = static_cast<int>(shadowHueParam    ? shadowHueParam->value()    : 0.0);
    params["shadowSat"]    = static_cast<int>(shadowSatParam    ? shadowSatParam->value()    : 0.0);
    params["highlightHue"] = static_cast<int>(highlightHueParam ? highlightHueParam->value() : 0.0);
    params["highlightSat"] = static_cast<int>(highlightSatParam ? highlightSatParam->value() : 0.0);
    params["balance"]      = static_cast<int>(balanceParam      ? balanceParam->value()      : 0.0);
    return params;
}

void SplitToningEffect::applyParameters(const QMap<QString, QVariant>& parameters) {
    auto apply = [&](ParamSlider* p, const char* key) {
        if (p && parameters.contains(key))
            p->setValue(parameters.value(key).toDouble());
    };
    apply(shadowHueParam,    "shadowHue");
    apply(shadowSatParam,    "shadowSat");
    apply(highlightHueParam, "highlightHue");
    apply(highlightSatParam, "highlightSat");
    apply(balanceParam,      "balance");
    emit parametersChanged();
}

// ============================================================================
// IGpuEffect — shared pipeline interface
// ============================================================================

bool SplitToningEffect::initGpuKernels(cl::Context& ctx, cl::Device& dev) {
    try {
        cl::Program prog(ctx, PIPELINE_KERNEL_SOURCE);
        prog.build({dev});
        m_kernelLinear = cl::Kernel(prog, "applySplitToningLinear");
        return true;
    }
    // GCOVR_EXCL_START
    catch (const cl::Error& e) {
        qWarning() << "[GpuPipeline] SplitToning initGpuKernels failed:"
                   << e.what() << "(err" << e.err() << ")";
        return false;
    }
    // GCOVR_EXCL_STOP
}

bool SplitToningEffect::enqueueGpu(cl::CommandQueue& queue,
                                    cl::Buffer& buf, cl::Buffer& /*aux*/,
                                    int w, int h,
                                    const QMap<QString, QVariant>& params) {
    const int shadowSat    = params.value("shadowSat",    0).toInt();
    const int highlightSat = params.value("highlightSat", 0).toInt();
    if (shadowSat == 0 && highlightSat == 0) return true;  // no-op

    const SplitArgs a = makeArgs(
        params.value("shadowHue",    0).toInt(), shadowSat,
        params.value("highlightHue", 0).toInt(), highlightSat,
        params.value("balance",      0).toInt());

    m_kernelLinear.setArg(0, buf);
    m_kernelLinear.setArg(1, w);
    m_kernelLinear.setArg(2, h);
    m_kernelLinear.setArg(3, a.shadowHue);
    m_kernelLinear.setArg(4, a.shadowSat);
    m_kernelLinear.setArg(5, a.highlightHue);
    m_kernelLinear.setArg(6, a.highlightSat);
    m_kernelLinear.setArg(7, a.balance);
    queue.enqueueNDRangeKernel(m_kernelLinear, cl::NullRange,
                               cl::NDRange(w, h), cl::NullRange);
    return true;
}

