#include "SaturationEffect.h"
#include "ParamSlider.h"
#include "color_kernels.h"
#include <QDebug>
#include <QVBoxLayout>

// ============================================================================
// Pipeline kernel — float4 linear sRGB, used by GpuPipeline via enqueueGpu.
// Saturation/vibrancy semantics are defined in sRGB-gamma HSV (the UI was
// designed around that perceptual space).  The linear-light pipeline kernel
// therefore gamma-encodes the input, runs the existing HSV adjust, and
// gamma-decodes the result back to linear.  Don't clamp outputs — the final
// pack kernel clamps once.  (HSV encode/decode is unchanged from the 8/16-bit
// kernels; only the colour-space wrap around it is new.)
// ============================================================================
static const char* PIPELINE_KERNEL_SOURCE = COLOR_KERNELS_SRC R"CL(
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

    // Encode to sRGB-gamma for HSV adjust (matches the slider semantics defined originally in sRGB-gamma space).
    float r = linear_to_srgb(px.x);
    float g = linear_to_srgb(px.y);
    float b = linear_to_srgb(px.z);

    // RGB → HSV
    float maxC  = fmax(fmax(r, g), b);
    float delta = maxC - fmin(fmin(r, g), b);
    float v = maxC;
    float s = (maxC == 0.0f) ? 0.0f : (delta / maxC);

    float hue = 0.0f;
    if (delta != 0.0f) {
        if      (maxC == r) hue = fmod((g - b) / delta, 6.0f);
        else if (maxC == g) hue = (b - r) / delta + 2.0f;
        else                hue = (r - g) / delta + 4.0f;
        hue = fmod(hue * 60.0f, 360.0f);
        if (hue < 0.0f) hue += 360.0f;
    }

    // Vibrancy with skin-tone protection (Gaussian around hue=20°, sigma=25°).
    if (vibrancyValue != 0.0f) {
        float hueDist = fabs(hue - 20.0f);
        if (hueDist > 180.0f) hueDist = 360.0f - hueDist;
        float skinProtect = exp(-0.5f * hueDist * hueDist / (25.0f * 25.0f));
        float weight = (1.0f - s) * (1.0f - 0.7f * skinProtect);
        s = clamp(s + (vibrancyValue / 100.0f) * weight, 0.0f, 1.0f);
    }

    // Global saturation boost after vibrancy.
    if (saturationValue != 0.0f) {
        s = clamp(s + saturationValue / 100.0f, 0.0f, 1.0f);
    }

    // HSV → RGB
    float c  = v * s;
    float xv = c * (1.0f - fabs(fmod(hue / 60.0f, 2.0f) - 1.0f));
    float m  = v - c;
    float rf, gf, bf;
    if      (hue <  60.0f) { rf = c;    gf = xv;   bf = 0.0f; }
    else if (hue < 120.0f) { rf = xv;   gf = c;    bf = 0.0f; }
    else if (hue < 180.0f) { rf = 0.0f; gf = c;    bf = xv;   }
    else if (hue < 240.0f) { rf = 0.0f; gf = xv;   bf = c;    }
    else if (hue < 300.0f) { rf = xv;   gf = 0.0f; bf = c;    }
    else                   { rf = c;    gf = 0.0f; bf = xv;   }

    // Decode back to linear — don't clamp; the final pack kernel clamps once.
    pixels[y * w + x] = (float4)(srgb_to_linear(rf + m),
                                 srgb_to_linear(gf + m),
                                 srgb_to_linear(bf + m),
                                 1.0f);
}
)CL";

// ============================================================================
// IGpuEffect — shared pipeline interface
// ============================================================================

bool SaturationEffect::initGpuKernels(cl::Context& ctx, cl::Device& dev) {
    try {
        cl::Program prog(ctx, PIPELINE_KERNEL_SOURCE);
        prog.build({dev});
        m_kernelLinear = cl::Kernel(prog, "adjustSatVibrancyLinear");
        return true;
    }
    // GCOVR_EXCL_START
    catch (const cl::Error& e) {
        qWarning() << "[GpuPipeline] Saturation initGpuKernels failed:"
                   << e.what() << "(err" << e.err() << ")";
        return false;
    }
    // GCOVR_EXCL_STOP
}

bool SaturationEffect::enqueueGpu(cl::CommandQueue& queue,
                                   cl::Buffer& buf, cl::Buffer& /*aux*/,
                                   int w, int h,
                                   const QMap<QString, QVariant>& params) {
    const float saturationValue = float(params.value("saturation", 0.0).toDouble());
    const float vibrancyValue   = float(params.value("vibrancy",   0.0).toDouble());
    if (saturationValue == 0.0f && vibrancyValue == 0.0f) return true;  // no-op

    m_kernelLinear.setArg(0, buf);
    m_kernelLinear.setArg(1, w);
    m_kernelLinear.setArg(2, h);
    m_kernelLinear.setArg(3, saturationValue);
    m_kernelLinear.setArg(4, vibrancyValue);
    queue.enqueueNDRangeKernel(m_kernelLinear, cl::NullRange,
                               cl::NDRange(w, h), cl::NullRange);
    return true;
}

SaturationEffect::SaturationEffect()
    : controlsWidget(nullptr), saturationParam(nullptr), vibrancyParam(nullptr) {
}

SaturationEffect::~SaturationEffect() {
}

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

QWidget* SaturationEffect::createControlsWidget() {
    if (controlsWidget) return controlsWidget;

    controlsWidget = new QWidget();
    QVBoxLayout* layout = new QVBoxLayout(controlsWidget);
    layout->setContentsMargins(0, 0, 0, 0);
    layout->setSpacing(10);

    saturationParam = new ParamSlider("Saturation", -20.0, 20.0, 0.1, 1);
    saturationParam->setToolTip("Globally boosts or reduces colour intensity across all hues equally.");
    connect(saturationParam, &ParamSlider::editingFinished, this, [this]() {
        emit parametersChanged();
    });
    connect(saturationParam, &ParamSlider::valueChanged, this, [this](double) {
        emit liveParametersChanged();
    });
    layout->addWidget(saturationParam);

    vibrancyParam = new ParamSlider("Vibrancy", -20.0, 20.0, 0.1, 1);
    vibrancyParam->setToolTip("Selectively boosts dull colours while protecting already-saturated tones and skin tones (orange/peach hues).");
    connect(vibrancyParam, &ParamSlider::editingFinished, this, [this]() {
        emit parametersChanged();
    });
    connect(vibrancyParam, &ParamSlider::valueChanged, this, [this](double) {
        emit liveParametersChanged();
    });
    layout->addWidget(vibrancyParam);

    layout->addStretch();
    return controlsWidget;
}

QMap<QString, QVariant> SaturationEffect::getParameters() const {
    QMap<QString, QVariant> params;
    params["saturation"] = saturationParam ? saturationParam->value() : 0.0;
    params["vibrancy"]   = vibrancyParam   ? vibrancyParam->value()   : 0.0;
    return params;
}

void SaturationEffect::applyParameters(const QMap<QString, QVariant>& parameters) {
    if (saturationParam && parameters.contains("saturation"))
        saturationParam->setValue(parameters.value("saturation").toDouble());
    if (vibrancyParam && parameters.contains("vibrancy"))
        vibrancyParam->setValue(parameters.value("vibrancy").toDouble());
    emit parametersChanged();
}

