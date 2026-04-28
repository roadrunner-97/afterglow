#include "BlurEffect.h"
#include "ParamSlider.h"
#include "blur_kernels.h"
#include "color_kernels.h"
#include <QDebug>
#include <QVBoxLayout>
#include <QLabel>
#include <QComboBox>
#include <algorithm>

// ============================================================================
// Pipeline kernel — float4 linear sRGB.  Blur is done directly in linear light,
// which gives specular highlights their natural bloom / bright-over-dark
// spread.  Two-pass separable: H(buf→aux), V(aux→buf); result in buf.
// ============================================================================
static const char* PIPELINE_KERNEL_SOURCE = SHARED_BLUR_KERNELS_F4;

// ============================================================================
// IGpuEffect — shared pipeline interface
// ============================================================================

bool BlurEffect::initGpuKernels(cl::Context& ctx, cl::Device& dev) {
    try {
        cl::Program prog(ctx, PIPELINE_KERNEL_SOURCE);
        prog.build({dev});
        m_kernelBlurHLinear = cl::Kernel(prog, "blurHLinear");
        m_kernelBlurVLinear = cl::Kernel(prog, "blurVLinear");
        return true;
    }
    // GCOVR_EXCL_START
    catch (const cl::Error& e) {
        qWarning() << "[GpuPipeline] Blur initGpuKernels failed:"
                   << e.what() << "(err" << e.err() << ")";
        return false;
    }
    // GCOVR_EXCL_STOP
}

bool BlurEffect::enqueueGpu(cl::CommandQueue& queue,
                             cl::Buffer& buf, cl::Buffer& aux,
                             int w, int h,
                             const QMap<QString, QVariant>& params) {
    const int radiusSrc = params.value("radius", 0).toInt();
    if (radiusSrc == 0) return true;  // no-op

    // Scale radius from source-pixel units to preview-pixel units so that the
    // perceived blur strength is independent of zoom level.
    const double scale = params.value("_srcPixelsPerPreviewPixel", 1.0).toDouble();
    int radius = std::max(1, static_cast<int>(radiusSrc / std::max(scale, 1e-6) + 0.5));

    const int isGaussian = (params.value("blurType", 0).toInt() == 0) ? 1 : 0;
    const cl::NDRange global(w, h);

    // Horizontal: buf → aux
    m_kernelBlurHLinear.setArg(0, buf);
    m_kernelBlurHLinear.setArg(1, aux);
    m_kernelBlurHLinear.setArg(2, w);
    m_kernelBlurHLinear.setArg(3, h);
    m_kernelBlurHLinear.setArg(4, radius);
    m_kernelBlurHLinear.setArg(5, isGaussian);
    queue.enqueueNDRangeKernel(m_kernelBlurHLinear, cl::NullRange, global, cl::NullRange);

    // Vertical: aux → buf (in-order queue; no finish needed)
    m_kernelBlurVLinear.setArg(0, aux);
    m_kernelBlurVLinear.setArg(1, buf);
    m_kernelBlurVLinear.setArg(2, w);
    m_kernelBlurVLinear.setArg(3, h);
    m_kernelBlurVLinear.setArg(4, radius);
    m_kernelBlurVLinear.setArg(5, isGaussian);
    queue.enqueueNDRangeKernel(m_kernelBlurVLinear, cl::NullRange, global, cl::NullRange);

    return true;
}

// ============================================================================
// Effect implementation
// ============================================================================

BlurEffect::BlurEffect()
    : controlsWidget(nullptr), blurTypeCombo(nullptr),
      radiusParam(nullptr), blurType(0) {
}

BlurEffect::~BlurEffect() {
}

QString BlurEffect::getName() const        { return "Blur"; }
QString BlurEffect::getDescription() const { return "Gaussian and box blur"; }
QString BlurEffect::getVersion() const     { return "1.0.0"; }

bool BlurEffect::initialize() {
    qDebug() << "Blur effect initialized";
    return true;
}

QWidget* BlurEffect::createControlsWidget() {
    if (controlsWidget) return controlsWidget;

    controlsWidget = new QWidget();
    QVBoxLayout* layout = new QVBoxLayout(controlsWidget);
    layout->setContentsMargins(0, 0, 0, 0);
    layout->setSpacing(8);

    // Blur type
    QLabel* typeLabel = new QLabel("Blur type:");
    typeLabel->setStyleSheet("color: #2C2018;");
    layout->addWidget(typeLabel);

    blurTypeCombo = new QComboBox();
    blurTypeCombo->addItem("Gaussian");
    blurTypeCombo->addItem("Box");
    blurTypeCombo->setToolTip("Gaussian: bell-curve weights for a soft, natural-looking blur.\nBox: uniform weights, slightly harder-edged but faster at large radii.");
    blurTypeCombo->setStyleSheet(
        "QComboBox { color: #2C2018; background-color: #F4F1EA;"
        "            border: 1px solid #CCC5B5; border-radius: 3px; padding: 3px; }"
        "QComboBox::drop-down { border: none; }"
        "QComboBox QAbstractItemView { color: #2C2018; background-color: #F4F1EA; }");
    layout->addWidget(blurTypeCombo);

    connect(blurTypeCombo, QOverload<int>::of(&QComboBox::activated), this, [this](int index) {
        blurType = index;
        emit parametersChanged();
    });

    // Radius
    radiusParam = new ParamSlider("Radius", 0, 50);
    radiusParam->setToolTip("Neighbourhood radius in pixels. Higher values produce stronger blurring. 0 = no effect.");
    connect(radiusParam, &ParamSlider::editingFinished, this, [this]() {
        emit parametersChanged();
    });
    connect(radiusParam, &ParamSlider::valueChanged, this, [this](double) {
        emit liveParametersChanged();
    });
    layout->addWidget(radiusParam);

    layout->addStretch();
    return controlsWidget;
}

QMap<QString, QVariant> BlurEffect::getParameters() const {
    QMap<QString, QVariant> params;
    params["blurType"] = blurType;
    params["radius"]   = radiusParam ? static_cast<int>(radiusParam->value()) : 0;
    return params;
}

void BlurEffect::applyParameters(const QMap<QString, QVariant>& parameters) {
    if (parameters.contains("blurType")) {
        blurType = parameters.value("blurType").toInt();
        if (blurTypeCombo) {
            QSignalBlocker block(blurTypeCombo);
            blurTypeCombo->setCurrentIndex(blurType);
        }
    }
    if (radiusParam && parameters.contains("radius"))
        radiusParam->setValue(parameters.value("radius").toDouble());
    emit parametersChanged();
}

