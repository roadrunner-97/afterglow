#include "HotPixelEffect.h"
#include "ParamSlider.h"
#include "color_kernels.h"
#include <QDebug>
#include <QVBoxLayout>

// ============================================================================
// Pipeline kernel — float4 linear sRGB, used by GpuPipeline via enqueueGpu.
// Same 3x3 neighbour-average replacement as the sRGB path, but on linear-light
// float4 pixels.  Threshold is the UI pct / 100, treated as a normalised
// deviation in the linear-channel [0, 1] range.  Reads `input` and writes
// `output` to avoid a read/write race on a single buffer; enqueueGpu copies
// the result back to `buf` so downstream effects see the corrected pixels.
// ============================================================================
static const char* PIPELINE_KERNEL_SOURCE = COLOR_KERNELS_SRC R"CL(
__kernel void hotPixelRemoveLinear(
    __global const float4* input,
    __global       float4* output,
    int w, int h,
    float threshold)
{
    int x = get_global_id(0);
    int y = get_global_id(1);
    if (x >= w || y >= h) return;

    float4 center = input[y * w + x];
    float sumR = 0.0f, sumG = 0.0f, sumB = 0.0f;
    for (int dy = -1; dy <= 1; ++dy) {
        for (int dx = -1; dx <= 1; ++dx) {
            if (dx == 0 && dy == 0) continue;
            int nx = clamp(x + dx, 0, w - 1);
            int ny = clamp(y + dy, 0, h - 1);
            float4 n = input[ny * w + nx];
            sumR += n.x;
            sumG += n.y;
            sumB += n.z;
        }
    }
    float avgR = sumR * 0.125f;
    float avgG = sumG * 0.125f;
    float avgB = sumB * 0.125f;

    float outR = (fabs(center.x - avgR) > threshold) ? avgR : center.x;
    float outG = (fabs(center.y - avgG) > threshold) ? avgG : center.y;
    float outB = (fabs(center.z - avgB) > threshold) ? avgB : center.z;

    output[y * w + x] = (float4)(outR, outG, outB, 1.0f);
}
)CL";

// ============================================================================
// IGpuEffect — shared pipeline interface
// ============================================================================

bool HotPixelEffect::initGpuKernels(cl::Context& ctx, cl::Device& dev) {
    try {
        cl::Program prog(ctx, PIPELINE_KERNEL_SOURCE);
        prog.build({dev});
        m_kernelLinear = cl::Kernel(prog, "hotPixelRemoveLinear");
        return true;
    }
    // GCOVR_EXCL_START
    catch (const cl::Error& e) {
        qWarning() << "[GpuPipeline] HotPixel initGpuKernels failed:"
                   << e.what() << "(err" << e.err() << ")";
        return false;
    }
    // GCOVR_EXCL_STOP
}

bool HotPixelEffect::enqueueGpu(cl::CommandQueue& queue,
                                 cl::Buffer& buf, cl::Buffer& aux,
                                 int w, int h,
                                 const QMap<QString, QVariant>& params) {
    const int thresholdPct = params.value("threshold", 30).toInt();
    if (thresholdPct == 0) return true;  // no-op

    // UI 0–100 → normalised [0, 1] linear-channel deviation.
    const float threshold = thresholdPct / 100.0f;

    m_kernelLinear.setArg(0, buf);    // input
    m_kernelLinear.setArg(1, aux);    // output scratch
    m_kernelLinear.setArg(2, w);
    m_kernelLinear.setArg(3, h);
    m_kernelLinear.setArg(4, threshold);

    queue.enqueueNDRangeKernel(m_kernelLinear, cl::NullRange,
                               cl::NDRange(w, h), cl::NullRange);
    // Copy aux → buf so downstream effects read corrected pixels from buf.
    const size_t bufBytes = static_cast<size_t>(w) * h * sizeof(cl_float4);
    queue.enqueueCopyBuffer(aux, buf, 0, 0, bufBytes);
    return true;
}

// ============================================================================
// Effect implementation
// ============================================================================

HotPixelEffect::HotPixelEffect()
    : controlsWidget(nullptr), thresholdParam(nullptr) {}

HotPixelEffect::~HotPixelEffect() {}

QString HotPixelEffect::getName() const        { return "Hot Pixel Removal"; }
QString HotPixelEffect::getDescription() const { return "Removes hot/dead pixels by replacing outliers with the local neighbourhood average"; }
QString HotPixelEffect::getVersion() const     { return "1.0.0"; }

bool HotPixelEffect::initialize() {
    qDebug() << "HotPixel effect initialized";
    return true;
}

QWidget* HotPixelEffect::createControlsWidget() {
    if (controlsWidget) return controlsWidget;

    controlsWidget = new QWidget();
    QVBoxLayout* layout = new QVBoxLayout(controlsWidget);
    layout->setContentsMargins(0, 0, 0, 0);
    layout->setSpacing(8);

    // Threshold slider: 0–100 (maps internally to per-channel deviation 0–255 / 0–65535)
    thresholdParam = new ParamSlider("Threshold", 0, 100);
    thresholdParam->setValue(30);
    thresholdParam->setToolTip("How far a pixel must deviate from its 8 neighbours to be considered a hot/dead pixel and replaced with the local average.\nLower values = more aggressive correction. Typical range: 20–40.");
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

QMap<QString, QVariant> HotPixelEffect::getParameters() const {
    QMap<QString, QVariant> params;
    params["threshold"] = thresholdParam ? static_cast<int>(thresholdParam->value()) : 30;
    return params;
}

void HotPixelEffect::applyParameters(const QMap<QString, QVariant>& parameters) {
    if (thresholdParam && parameters.contains("threshold"))
        thresholdParam->setValue(parameters.value("threshold").toDouble());
    emit parametersChanged();
}

