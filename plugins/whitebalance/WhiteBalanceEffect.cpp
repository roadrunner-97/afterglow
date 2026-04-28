#include "WhiteBalanceEffect.h"
#include "ParamSlider.h"
#include "color_kernels.h"
#include <QDebug>
#include <QVBoxLayout>
#include <cmath>
#include <algorithm>


namespace {

// ---------------------------------------------------------------------------
// Kernel — receives pre-computed per-channel multipliers from the CPU.
// The Kang formula math stays on the host; the kernel just applies the
// three multipliers.  This keeps the kernel trivially simple and avoids
// reimplementing polynomial evaluation on the GPU.
//
// 8-bit format:  QImage::Format_RGB32  (uint = 0xFFRRGGBB)
// 16-bit format: QImage::Format_RGBX64 (ushort4, LE: s0=R s1=G s2=B s3=X)
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Kang et al. (2002) Planckian locus → linear sRGB (D65 reference white).
// Used on the CPU to compute correction multipliers from a pair of K values.
// ---------------------------------------------------------------------------
static void kangToRGB(float T, float& r, float& g, float& b) {
    T = std::max(1000.0f, std::min(15000.0f, T));

    float x, y;
    if (T <= 4000.0f) {
        x = -0.2661239e9f/(T*T*T) - 0.2343580e6f/(T*T) + 0.8776956e3f/T + 0.179910f;
        y = (T <= 2222.0f)
            ? (-1.1063814f*(x*x*x) - 1.34811020f*(x*x) + 2.18555832f*x - 0.20219683f)
            : (-0.9549476f*(x*x*x) - 1.37418593f*(x*x) + 2.09137015f*x - 0.16748867f);
    } else {
        x = -3.0258469e9f/(T*T*T) + 2.1070379e6f/(T*T) + 0.2226347e3f/T + 0.240390f;
        y = 3.0817580f*(x*x*x) - 5.8733867f*(x*x) + 3.75112997f*x - 0.37001483f;
    }

    float X = x / y;
    float Z = (1.0f - x - y) / y;
    r = std::max(1e-6f,  3.2406f*X - 1.5372f - 0.4986f*Z);
    g = std::max(1e-6f, -0.9689f*X + 1.8758f + 0.0415f*Z);
    b = std::max(1e-6f,  0.0557f*X - 0.2040f + 1.0570f*Z);
}

// ---------------------------------------------------------------------------
// Compute the three per-channel multipliers to shift the apparent white point
// from shotK to targetK, then layer in the tint correction.
//
// Convention: targetK > shotK → warmer image (more amber, less blue).
//             tint > 0 → magenta shift (less green).
//             When targetK == shotK and tint == 0: all multipliers are 1.
// ---------------------------------------------------------------------------
static void computeWBMuls(float shotK, float targetK, float tint,
                           float& rMul, float& gMul, float& bMul)
{
    float rs, gs, bs, rt, gt, bt;
    kangToRGB(shotK,   rs, gs, bs);
    kangToRGB(targetK, rt, gt, bt);

    // Normalise both to G = 1 so we only shift chromaticity, not exposure.
    rs /= gs;  bs /= gs;   // gs → 1
    rt /= gt;  bt /= gt;   // gt → 1

    // The pixel-space multiplier: when shotK == targetK, rs==rt → ratio = 1.
    rMul = rs / rt;
    gMul = 1.0f;
    bMul = bs / bt;

    // Tint axis (green ↔ magenta), orthogonal to temperature.
    // tint ∈ [−100, +100]:  +100 = 15% less green (magenta), −100 = 15% more green.
    gMul *= 1.0f - tint / 100.0f * 0.15f;

    // Safety clamp: avoid divide-by-near-zero or extreme blowouts.
    rMul = std::max(0.05f, std::min(20.0f, rMul));
    gMul = std::max(0.05f, std::min(20.0f, gMul));
    bMul = std::max(0.05f, std::min(20.0f, bMul));
}


} // namespace

// ============================================================================
// Pipeline kernel — float4 linear sRGB, used by GpuPipeline via enqueueGpu.
// The Kang-derived multipliers are already in linear sRGB primaries, so the
// linear-light kernel is a plain per-channel multiply — no gamma decode/encode
// ceremony.  Don't clamp outputs; the final pack kernel clamps once.
// ============================================================================
static const char* PIPELINE_KERNEL_SOURCE = COLOR_KERNELS_SRC R"CL(
__kernel void applyWBLinear(__global float4* pixels,
                             int   w,
                             int   h,
                             float rMul,
                             float gMul,
                             float bMul)
{
    int x = get_global_id(0);
    int y = get_global_id(1);
    if (x >= w || y >= h) return;

    float4 px = pixels[y * w + x];
    pixels[y * w + x] = (float4)(px.x * rMul,
                                 px.y * gMul,
                                 px.z * bMul,
                                 1.0f);
}
)CL";

// ============================================================================
// IGpuEffect — shared pipeline interface
// ============================================================================

bool WhiteBalanceEffect::initGpuKernels(cl::Context& ctx, cl::Device& dev) {
    try {
        cl::Program prog(ctx, PIPELINE_KERNEL_SOURCE);
        prog.build({dev});
        m_kernelLinear = cl::Kernel(prog, "applyWBLinear");
        return true;
    }
    // GCOVR_EXCL_START
    catch (const cl::Error& e) {
        qWarning() << "[GpuPipeline] WhiteBalance initGpuKernels failed:"
                   << e.what() << "(err" << e.err() << ")";
        return false;
    }
    // GCOVR_EXCL_STOP
}

bool WhiteBalanceEffect::enqueueGpu(cl::CommandQueue& queue,
                                    cl::Buffer& buf, cl::Buffer& /*aux*/,
                                    int w, int h,
                                    const QMap<QString, QVariant>& params) {
    const float shotK   = float(params.value("shot_temp",   5500.0).toDouble());
    const float targetK = float(params.value("temperature", 5500.0).toDouble());
    const float tint    = float(params.value("tint",        0.0).toDouble());

    // No-op shortcut: same source/target temp and zero tint → all muls are 1.
    if (shotK == targetK && tint == 0.0f) return true;

    float rMul, gMul, bMul;
    computeWBMuls(shotK, targetK, tint, rMul, gMul, bMul);

    m_kernelLinear.setArg(0, buf);
    m_kernelLinear.setArg(1, w);
    m_kernelLinear.setArg(2, h);
    m_kernelLinear.setArg(3, rMul);
    m_kernelLinear.setArg(4, gMul);
    m_kernelLinear.setArg(5, bMul);
    queue.enqueueNDRangeKernel(m_kernelLinear, cl::NullRange,
                               cl::NDRange(w, h), cl::NullRange);
    return true;
}

// ============================================================================
// PhotoEditorEffect
// ============================================================================

WhiteBalanceEffect::WhiteBalanceEffect()
    : controlsWidget(nullptr), temperatureParam(nullptr), tintParam(nullptr)
    , m_shotK(5500.0f)
{
}

WhiteBalanceEffect::~WhiteBalanceEffect() {
}

QString WhiteBalanceEffect::getName() const    { return "White Balance"; }
QString WhiteBalanceEffect::getDescription() const { return "Adjusts color temperature and tint"; }
QString WhiteBalanceEffect::getVersion() const { return "1.0.0"; }
bool    WhiteBalanceEffect::initialize()       { qDebug() << "White Balance effect initialized"; return true; }

void WhiteBalanceEffect::onImageLoaded(const ImageMetadata& meta) {
    // Use the as-shot temperature from metadata if available; keep 5500 K otherwise.
    if (meta.colorTempK >= 1500.0f && meta.colorTempK <= 14000.0f)
        m_shotK = meta.colorTempK;
    else
        m_shotK = 5500.0f;

    // Update the slider to the new default without firing any signals
    // (ParamSlider::setValue blocks signals internally).
    if (temperatureParam)
        temperatureParam->setValue(m_shotK);
}

QWidget* WhiteBalanceEffect::createControlsWidget() {
    if (controlsWidget) return controlsWidget;

    controlsWidget = new QWidget();
    QVBoxLayout* layout = new QVBoxLayout(controlsWidget);
    layout->setContentsMargins(0, 0, 0, 0);
    layout->setSpacing(10);

    // Temperature: 2000 K – 12000 K, 50 K steps, default = as-shot (m_shotK).
    // Slider right = warmer (amber), left = cooler (blue).
    temperatureParam = new ParamSlider("Temperature", 2000.0, 12000.0, 50.0, 0);
    temperatureParam->setValue(m_shotK);
    temperatureParam->setToolTip("Colour temperature of the light source in Kelvin.\nLower = cooler/blue (e.g. overcast sky ~7000 K), higher = warmer/amber (e.g. tungsten ~3200 K).\nDefault is the as-shot value read from the RAW file metadata.");
    connect(temperatureParam, &ParamSlider::editingFinished, this, [this]() {
        emit parametersChanged();
    });
    connect(temperatureParam, &ParamSlider::valueChanged, this, [this](double) {
        emit liveParametersChanged();
    });
    layout->addWidget(temperatureParam);

    // Tint: −100 (green) … +100 (magenta), 1-step, default 0.
    tintParam = new ParamSlider("Tint", -100.0, 100.0, 1.0, 0);
    tintParam->setToolTip("Fine-tunes the green–magenta axis, orthogonal to colour temperature.\nPositive = magenta shift (less green), negative = green shift.\nUse to correct fluorescent or mixed-light casts after setting temperature.");
    connect(tintParam, &ParamSlider::editingFinished, this, [this]() {
        emit parametersChanged();
    });
    connect(tintParam, &ParamSlider::valueChanged, this, [this](double) {
        emit liveParametersChanged();
    });
    layout->addWidget(tintParam);

    layout->addStretch();
    return controlsWidget;
}

QMap<QString, QVariant> WhiteBalanceEffect::getParameters() const {
    QMap<QString, QVariant> params;
    params["temperature"] = temperatureParam ? temperatureParam->value() : static_cast<double>(m_shotK);
    params["tint"]        = tintParam        ? tintParam->value()        : 0.0;
    params["shot_temp"]   = static_cast<double>(m_shotK);
    return params;
}

void WhiteBalanceEffect::applyParameters(const QMap<QString, QVariant>& parameters) {
    // shot_temp is image-derived metadata, not a user-editable control —
    // skip restoring it from a saved file (the loaded image's own metadata wins).
    if (temperatureParam && parameters.contains("temperature"))
        temperatureParam->setValue(parameters.value("temperature").toDouble());
    if (tintParam && parameters.contains("tint"))
        tintParam->setValue(parameters.value("tint").toDouble());
    emit parametersChanged();
}

