#include "BrightnessEffect.h"
#include "ParamSlider.h"
#include "color_kernels.h"
#include <QDebug>
#include <QVBoxLayout>

// Pipeline kernel — float4 linear sRGB, used by GpuPipeline via enqueueGpu.
// Brightness and contrast are defined as slider offsets in sRGB-gamma space,
// so the linear-light pipeline kernel gamma-encodes, applies the offsets, then
// decodes back to linear.  Don't clamp here — the final pack kernel clamps.
static const char* PIPELINE_KERNEL_SOURCE = COLOR_KERNELS_SRC R"CL(
__kernel void adjustBrightnessLinear(__global float4* pixels,
                                     int   w,
                                     int   h,
                                     int   brightnessFactor,
                                     float contrastFactor)
{
    int x = get_global_id(0);
    int y = get_global_id(1);
    if (x >= w || y >= h) return;

    float4 px = pixels[y * w + x];

    float r = linear_to_srgb(px.x);
    float g = linear_to_srgb(px.y);
    float b = linear_to_srgb(px.z);

    float bd = brightnessFactor / 255.0f;
    r = r + bd;
    g = g + bd;
    b = b + bd;

    r = (r - 0.5f) * contrastFactor + 0.5f;
    g = (g - 0.5f) * contrastFactor + 0.5f;
    b = (b - 0.5f) * contrastFactor + 0.5f;

    pixels[y * w + x] = (float4)(srgb_to_linear(r),
                                 srgb_to_linear(g),
                                 srgb_to_linear(b),
                                 1.0f);
}
)CL";

BrightnessEffect::BrightnessEffect() = default;
BrightnessEffect::~BrightnessEffect() = default;

QString BrightnessEffect::getName() const {
    return "Brightness & Contrast";
}

QString BrightnessEffect::getDescription() const {
    return "Adjusts brightness and contrast of the image";
}

QString BrightnessEffect::getVersion() const {
    return "2.0.0";
}

bool BrightnessEffect::initialize() {
    qDebug() << "Brightness & Contrast effect initialized";
    return true;
}

QWidget* BrightnessEffect::createControlsWidget() {
    if (controlsWidget) return controlsWidget;

    controlsWidget = new QWidget();
    QVBoxLayout* layout = new QVBoxLayout(controlsWidget);
    layout->setContentsMargins(0, 0, 0, 0);
    layout->setSpacing(8);

    brightnessParam = new ParamSlider("Brightness", -100, 100);
    brightnessParam->setToolTip("Shifts all pixel values brighter (positive) or darker (negative).");
    connect(brightnessParam, &ParamSlider::editingFinished, this, [this]() {
        emit parametersChanged();
    });
    connect(brightnessParam, &ParamSlider::valueChanged, this, [this](double) {
        emit liveParametersChanged();
    });
    layout->addWidget(brightnessParam);

    contrastParam = new ParamSlider("Contrast", -50, 50);
    contrastParam->setToolTip("Expands (positive) or compresses (negative) the tonal range around the midpoint (128).");
    connect(contrastParam, &ParamSlider::editingFinished, this, [this]() {
        emit parametersChanged();
    });
    connect(contrastParam, &ParamSlider::valueChanged, this, [this](double) {
        emit liveParametersChanged();
    });
    layout->addWidget(contrastParam);

    layout->addStretch();
    return controlsWidget;
}

QMap<QString, QVariant> BrightnessEffect::getParameters() const {
    QMap<QString, QVariant> params;
    params["brightness"] = static_cast<int>(brightnessParam ? brightnessParam->value() : 0.0);
    params["contrast"]   = static_cast<int>(contrastParam   ? contrastParam->value()   : 0.0);
    return params;
}

void BrightnessEffect::applyParameters(const QMap<QString, QVariant>& parameters) {
    if (brightnessParam && parameters.contains("brightness"))
        brightnessParam->setValue(parameters.value("brightness").toDouble());
    if (contrastParam && parameters.contains("contrast"))
        contrastParam->setValue(parameters.value("contrast").toDouble());
    emit parametersChanged();
}

bool BrightnessEffect::initGpuKernels(cl::Context& ctx, cl::Device& dev) {
    try {
        cl::Program prog(ctx, PIPELINE_KERNEL_SOURCE);
        prog.build({dev});
        m_kernelLinear = cl::Kernel(prog, "adjustBrightnessLinear");
        return true;
    }
    // GCOVR_EXCL_START
    catch (const cl::Error& e) {
        qWarning() << "[GpuPipeline] Brightness initGpuKernels failed:"
                   << e.what() << "(err" << e.err() << ")";
        return false;
    }
    // GCOVR_EXCL_STOP
}

bool BrightnessEffect::enqueueGpu(cl::CommandQueue& queue,
                                   cl::Buffer& buf, cl::Buffer& /*aux*/,
                                   int w, int h,
                                   const QMap<QString, QVariant>& params) {
    const int brightnessFactor = params.value("brightness", 0).toInt();
    const int contrastInt      = params.value("contrast", 0).toInt();
    if (brightnessFactor == 0 && contrastInt == 0) return true;

    const float contrastFactor = (contrastInt + 100.0f) / 100.0f;

    m_kernelLinear.setArg(0, buf);
    m_kernelLinear.setArg(1, w);
    m_kernelLinear.setArg(2, h);
    m_kernelLinear.setArg(3, brightnessFactor);
    m_kernelLinear.setArg(4, contrastFactor);
    queue.enqueueNDRangeKernel(m_kernelLinear, cl::NullRange,
                               cl::NDRange(w, h), cl::NullRange);
    return true;
}
