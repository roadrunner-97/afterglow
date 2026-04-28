#include "ColorBalanceEffect.h"
#include "ParamSlider.h"
#include "color_kernels.h"
#include <QDebug>
#include <QVBoxLayout>
#include <QLabel>


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

void ColorBalanceEffect::applyParameters(const QMap<QString, QVariant>& parameters) {
    auto apply = [&](ParamSlider* p, const char* key) {
        if (p && parameters.contains(key))
            p->setValue(parameters.value(key).toDouble());
    };
    apply(shadowRParam,    "shadowR");
    apply(shadowGParam,    "shadowG");
    apply(shadowBParam,    "shadowB");
    apply(midtoneRParam,   "midtoneR");
    apply(midtoneGParam,   "midtoneG");
    apply(midtoneBParam,   "midtoneB");
    apply(highlightRParam, "highlightR");
    apply(highlightGParam, "highlightG");
    apply(highlightBParam, "highlightB");
    emit parametersChanged();
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

