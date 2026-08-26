#include "SaturationEffect.h"
#include "ParamSlider.h"
#include "color_kernels.h"
#include <QDebug>
#include <QVBoxLayout>

// ============================================================================
// Pipeline kernel — float4 linear sRGB, used by GpuPipeline via enqueueGpu.
//
// Linear-light chroma scaling: subtract Rec.709 luma to isolate chroma, scale
// chroma by a per-pixel factor, add luma back.  Compared to the older
// additive-HSV formulation this avoids two pathologies that surfaced under
// heavy pushes:
//   * additive `s += k` blows up near-neutral pixels (a tiny chroma noise
//     speck with s=0.02 became s=0.22 — an 11× amplification — making
//     shadow/midtone noise erupt as colour blotches);
//   * hard-clamping `s` to 1 posterised the most saturated regions.
// Multiplicative chroma scaling is uniform, gamut-friendly, and stays in
// linear light for the whole operation (no gamma round-trip).
//
// Saturation slider in [-100, +100] → multiplicative factor in [0, 2].
// Vibrancy contributes an additional per-pixel factor weighted by (1 - sLin)
// so already-saturated colours barely move, and attenuated by a Gaussian
// around the orange/peach skin-tone hue band.  Don't clamp outputs — the
// final pack kernel clamps once at readback.
// ============================================================================
static const char *PIPELINE_KERNEL_SOURCE = COLOR_KERNELS_SRC R"CL(
__kernel void adjustSatVibrancyLinear(__global float4* pixels,
                                       int   w,
                                       int   h,
                                       float saturationValue,
                                       float vibrancyValue)
{
    int x = get_global_id(0);
    int y = get_global_id(1);
    if (x >= w || y >= h) return;

    float4 px = pixels[y * w + x];
    float r = px.x, g = px.y, b = px.z;

    float satFactor = 1.0f + saturationValue / 100.0f;

    float vibFactor = 1.0f;
    if (vibrancyValue != 0.0f) {
        float maxC = fmax(fmax(r, g), b);
        float minC = fmin(fmin(r, g), b);
        float delta = maxC - minC;
        float sLin = (maxC > 1e-6f) ? delta / maxC : 0.0f;

        float hue = 0.0f;
        if (delta > 1e-6f) {
            if      (maxC == r) hue = (g - b) / delta;
            else if (maxC == g) hue = (b - r) / delta + 2.0f;
            else                hue = (r - g) / delta + 4.0f;
            hue *= 60.0f;
            if (hue < 0.0f) hue += 360.0f;
        }
        float hueDist = fabs(hue - 20.0f);
        if (hueDist > 180.0f) hueDist = 360.0f - hueDist;
        float skinProtect = exp(-0.5f * hueDist * hueDist / (25.0f * 25.0f));

        float weight = (1.0f - sLin) * (1.0f - 0.7f * skinProtect);
        vibFactor = 1.0f + (vibrancyValue / 100.0f) * weight;
    }

    float factor = satFactor * vibFactor;
    float luma = 0.2126f * r + 0.7152f * g + 0.0722f * b;

    pixels[y * w + x] = (float4)(luma + (r - luma) * factor,
                                 luma + (g - luma) * factor,
                                 luma + (b - luma) * factor,
                                 1.0f);
}
)CL";

// ============================================================================
// IGpuEffect — shared pipeline interface
// ============================================================================

bool SaturationEffect::initGpuKernels(cl::Context &ctx, cl::Device &dev) {
    try {
        cl::Program prog(ctx, PIPELINE_KERNEL_SOURCE);
        prog.build({dev});
        m_kernelLinear = cl::Kernel(prog, "adjustSatVibrancyLinear");
        return true;
    }
    // GCOVR_EXCL_START
    catch (const cl::Error &e) {
        qWarning() << "[GpuPipeline] Saturation initGpuKernels failed:" << e.what() << "(err" << e.err() << ")";
        return false;
    }
    // GCOVR_EXCL_STOP
}

bool SaturationEffect::enqueueGpu(cl::CommandQueue &queue, cl::Buffer &buf, cl::Buffer & /*aux*/, int w, int h,
                                  const QMap<QString, QVariant> &params) {
    const float saturationValue = float(params.value("saturation", 0.0).toDouble());
    const float vibrancyValue   = float(params.value("vibrancy", 0.0).toDouble());
    if (saturationValue == 0.0f && vibrancyValue == 0.0f) return true; // no-op

    m_kernelLinear.setArg(0, buf);
    m_kernelLinear.setArg(1, w);
    m_kernelLinear.setArg(2, h);
    m_kernelLinear.setArg(3, saturationValue);
    m_kernelLinear.setArg(4, vibrancyValue);
    queue.enqueueNDRangeKernel(m_kernelLinear, cl::NullRange,
                               cl::NDRange(static_cast<size_t>(w), static_cast<size_t>(h)), cl::NullRange);
    return true;
}

SaturationEffect::SaturationEffect() : controlsWidget(nullptr), saturationParam(nullptr), vibrancyParam(nullptr) {}

SaturationEffect::~SaturationEffect() {}

QString SaturationEffect::getName() const {
    return "Saturation & Vibrancy";
}

QString SaturationEffect::getDescription() const {
    return "Adjusts saturation and vibrancy of the image";
}

QString SaturationEffect::getVersion() const {
    return "1.0.0";
}

bool SaturationEffect::initialize() {
    qDebug() << "Saturation & Vibrancy effect initialized";
    return true;
}

QWidget *SaturationEffect::createControlsWidget() {
    if (controlsWidget) return controlsWidget;

    controlsWidget      = new QWidget();
    QVBoxLayout *layout = new QVBoxLayout(controlsWidget);
    layout->setContentsMargins(0, 0, 0, 0);
    layout->setSpacing(10);

    saturationParam = new ParamSlider("Saturation", -100.0, 100.0, 0.1, 1);
    saturationParam->setToolTip("Globally boosts or reduces colour intensity across all hues equally.");
    connect(saturationParam, &ParamSlider::editingFinished, this, [this]() { emit parametersChanged(); });
    connect(saturationParam, &ParamSlider::valueChanged, this, [this](double) { emit liveParametersChanged(); });
    layout->addWidget(saturationParam);

    vibrancyParam = new ParamSlider("Vibrancy", -100.0, 100.0, 0.1, 1);
    vibrancyParam->setToolTip(
        "Selectively boosts dull colours while protecting already-saturated tones and skin tones (orange/peach hues).");
    connect(vibrancyParam, &ParamSlider::editingFinished, this, [this]() { emit parametersChanged(); });
    connect(vibrancyParam, &ParamSlider::valueChanged, this, [this](double) { emit liveParametersChanged(); });
    layout->addWidget(vibrancyParam);

    layout->addStretch();
    return controlsWidget;
}

QMap<QString, QVariant> SaturationEffect::getParameters() const {
    QMap<QString, QVariant> params;
    params["saturation"] = saturationParam ? saturationParam->value() : 0.0;
    params["vibrancy"]   = vibrancyParam ? vibrancyParam->value() : 0.0;
    return params;
}

void SaturationEffect::applyParameters(const QMap<QString, QVariant> &parameters) {
    if (saturationParam && parameters.contains("saturation"))
        saturationParam->setValue(parameters.value("saturation").toDouble());
    if (vibrancyParam && parameters.contains("vibrancy"))
        vibrancyParam->setValue(parameters.value("vibrancy").toDouble());
    emit parametersChanged();
}
