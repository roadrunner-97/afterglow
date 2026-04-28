#include "VignetteEffect.h"
#include "ParamSlider.h"
#include "color_kernels.h"
#include <QDebug>
#include <QVBoxLayout>
#include <cmath>


namespace {

// Distance metric: L^p norm on isotropic (pixel-space) coordinates — the
// same scale (halfD = half the image diagonal) is used for both axes, so at
// p=2 the level curves are true circles, matching a real lens's circular
// image circle regardless of the frame's aspect ratio.  Higher p pushes the
// shape toward an axis-aligned square; lower p toward an axis-aligned
// diamond.
//
// In the UI:
//   roundness = +100  →  p = 0.5  (very round/diamond, corners darkened heavily)
//   roundness =    0  →  p = 2    (circle through the image corners)
//   roundness = -100  →  p = 8    (rectangular, hugs the frame)
//
// cornerD is passed in so we can normalise d to [0..1] across the roundness
// range — the host pre-computes cornerD as the d value at the image corner
// (|nx|=halfW/halfD, |ny|=halfH/halfD), so dn=1 at the corner for any p.
// smoothstep from (midpoint-feather/2) to (midpoint+feather/2) on that
// normalised distance gives the transition weight t; factor=1+amount*t
// multiplies RGB (amount<0 darkens corners, amount>0 lightens).

struct VignetteArgs {
    float amount;    // -1..1
    float midpoint;  // 0..1
    float feather;   // 0..1
    float p;         // L^p exponent
};

static VignetteArgs makeArgs(int amount, int midpoint, int feather, int roundness) {
    VignetteArgs a;
    a.amount   = amount   / 100.0f;
    a.midpoint = midpoint / 100.0f;
    a.feather  = feather  / 100.0f;
    float rn   = roundness / 100.0f;                 // [-1, 1]
    a.p        = std::exp2(1.0f - rn * 3.0f);        // roundness=0 → p=2
    return a;
}

} // namespace

// ============================================================================
// Pipeline kernel — float4 linear sRGB, used by GpuPipeline via enqueueGpu.
// Vignette is a pure linear-light multiply on RGB: compute an isotropic L^p
// distance from image centre, smoothstep-feathered around a midpoint, then
// scale each channel by factor = 1 + amount * t.  No clamping — the final
// pack kernel clamps once before readback.
//
// Coordinates are source-anchored via _cropX0/_cropY0/_srcPixelsPerPreviewPixel
// (passed as srcX0/srcY0/srcPPP) so the vignette stays centred on the source
// image regardless of zoom, pan, or aspect-mismatched fit-to-window padding.
// In Commit mode srcPPP=1 and srcX0=srcY0=0 so the transform is the identity.
// ============================================================================
static const char* PIPELINE_KERNEL_SOURCE = COLOR_KERNELS_SRC R"CL(
__kernel void applyVignetteLinear(__global float4* pixels,
                                   int   w,
                                   int   h,
                                   float cx,
                                   float cy,
                                   float halfW,
                                   float halfH,
                                   float amount,
                                   float midpoint,
                                   float feather,
                                   float p,
                                   float cornerD,
                                   float srcX0,
                                   float srcY0,
                                   float srcPPP)
{
    int x = get_global_id(0);
    int y = get_global_id(1);
    if (x >= w || y >= h) return;

    float sx = srcX0 + ((float)x + 0.5f) * srcPPP;
    float sy = srcY0 + ((float)y + 0.5f) * srcPPP;

    float nx = fabs((sx - cx) / halfW);
    float ny = fabs((sy - cy) / halfH);
    float d  = native_powr(native_powr(nx, p) + native_powr(ny, p), 1.0f / p);
    float dn = d / cornerD;

    float edge0 = midpoint - feather * 0.5f;
    float edge1 = midpoint + feather * 0.5f + 1e-5f;
    float t = smoothstep(edge0, edge1, dn);
    float factor = 1.0f + amount * t;

    float4 px = pixels[y * w + x];
    pixels[y * w + x] = (float4)(px.x * factor,
                                 px.y * factor,
                                 px.z * factor,
                                 1.0f);
}
)CL";

// ============================================================================
// VignetteEffect
// ============================================================================

VignetteEffect::VignetteEffect()
    : controlsWidget(nullptr), amountParam(nullptr), midpointParam(nullptr),
      featherParam(nullptr), roundnessParam(nullptr) {
}

VignetteEffect::~VignetteEffect() {
}

QString VignetteEffect::getName() const { return "Vignette"; }
QString VignetteEffect::getDescription() const {
    return "Radial darkening or lightening around the image center";
}
QString VignetteEffect::getVersion() const { return "1.0.0"; }

bool VignetteEffect::initialize() {
    qDebug() << "Vignette effect initialized";
    return true;
}

QWidget* VignetteEffect::createControlsWidget() {
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
    amountParam->setToolTip("Strength and direction of the vignette.\nNegative darkens the corners; positive lightens them.");
    connectSlider(amountParam);
    layout->addWidget(amountParam);

    midpointParam = new ParamSlider("Midpoint", 0, 100);
    midpointParam->setValue(50);
    midpointParam->setToolTip("Centre of the transition, as a fraction of the way from image centre to corner.\nLower values move the vignette inward.");
    connectSlider(midpointParam);
    layout->addWidget(midpointParam);

    featherParam = new ParamSlider("Feather", 0, 100);
    featherParam->setValue(50);
    featherParam->setToolTip("Softness of the transition.\n0 is a hard edge; 100 fades across the full image.");
    connectSlider(featherParam);
    layout->addWidget(featherParam);

    roundnessParam = new ParamSlider("Roundness", -100, 100);
    roundnessParam->setToolTip("Shape of the vignette.\nPositive values round the falloff (heavier corner darkening);\nnegative values pull it toward the rectangular frame.");
    connectSlider(roundnessParam);
    layout->addWidget(roundnessParam);

    layout->addStretch();
    return controlsWidget;
}

QMap<QString, QVariant> VignetteEffect::getParameters() const {
    QMap<QString, QVariant> params;
    params["amount"]    = static_cast<int>(amountParam    ? amountParam->value()    : 0.0);
    params["midpoint"]  = static_cast<int>(midpointParam  ? midpointParam->value()  : 50.0);
    params["feather"]   = static_cast<int>(featherParam   ? featherParam->value()   : 50.0);
    params["roundness"] = static_cast<int>(roundnessParam ? roundnessParam->value() : 0.0);
    return params;
}

void VignetteEffect::applyParameters(const QMap<QString, QVariant>& parameters) {
    auto apply = [&](ParamSlider* p, const char* key) {
        if (p && parameters.contains(key))
            p->setValue(parameters.value(key).toDouble());
    };
    apply(amountParam,    "amount");
    apply(midpointParam,  "midpoint");
    apply(featherParam,   "feather");
    apply(roundnessParam, "roundness");
    emit parametersChanged();
}

// ============================================================================
// IGpuEffect — shared pipeline interface
// ============================================================================

bool VignetteEffect::initGpuKernels(cl::Context& ctx, cl::Device& dev) {
    try {
        cl::Program prog(ctx, PIPELINE_KERNEL_SOURCE);
        prog.build({dev});
        m_kernelLinear = cl::Kernel(prog, "applyVignetteLinear");
        return true;
    }
    // GCOVR_EXCL_START
    catch (const cl::Error& e) {
        qWarning() << "[GpuPipeline] Vignette initGpuKernels failed:"
                   << e.what() << "(err" << e.err() << ")";
        return false;
    }
    // GCOVR_EXCL_STOP
}

bool VignetteEffect::enqueueGpu(cl::CommandQueue& queue,
                                 cl::Buffer& buf, cl::Buffer& /*aux*/,
                                 int w, int h,
                                 const QMap<QString, QVariant>& params) {
    const int amount = params.value("amount", 0).toInt();
    if (amount == 0) return true;  // no-op

    const VignetteArgs a = makeArgs(
        amount,
        params.value("midpoint",  50).toInt(),
        params.value("feather",   50).toInt(),
        params.value("roundness",  0).toInt());

    // Compute geometry in source-image coordinates so the vignette stays
    // anchored to the source centre regardless of preview crop / aspect.
    // In Commit mode srcW=w, srcH=h, srcPPP=1, srcX0=srcY0=0 — same as before.
    const float srcPPP = static_cast<float>(
        params.value("_srcPixelsPerPreviewPixel", 1.0).toDouble());
    const float srcX0  = static_cast<float>(params.value("_cropX0", 0.0).toDouble());
    const float srcY0  = static_cast<float>(params.value("_cropY0", 0.0).toDouble());
    const int   srcW   = params.value("_srcW", w).toInt();
    const int   srcH   = params.value("_srcH", h).toInt();

    // Post-crop vignette: centre and extents are derived from the user crop
    // rectangle (normalised 0..1 source coords) so the falloff is anchored to
    // the crop boundary rather than the full source frame.  Default values
    // {0,0,1,1} reproduce the previous full-frame behaviour when no crop is set.
    const double uX0 = params.value("_userCropX0", 0.0).toDouble();
    const double uY0 = params.value("_userCropY0", 0.0).toDouble();
    const double uX1 = params.value("_userCropX1", 1.0).toDouble();
    const double uY1 = params.value("_userCropY1", 1.0).toDouble();

    // Convert user-crop rect to source-pixel coords.
    const float cropCxSrc    = static_cast<float>((uX0 + uX1) * 0.5 * srcW);
    const float cropCySrc    = static_cast<float>((uY0 + uY1) * 0.5 * srcH);
    const float cropHalfWSrc = static_cast<float>((uX1 - uX0) * 0.5 * srcW);
    const float cropHalfHSrc = static_cast<float>((uY1 - uY0) * 0.5 * srcH);

    // TODO: _userCropAngle is not yet applied to the vignette falloff.  The
    // full rotation-aware math rotates the nx/ny axes by the crop angle before
    // taking the L^p norm; this can be added later without changing the param
    // schema.

    // Transform crop centre and extents from source-pixel coords to
    // current-buffer coords using the same srcX0/srcPPP translation the kernel
    // already uses for each pixel.
    const float cx    = (cropCxSrc - srcX0) / srcPPP;
    const float cy    = (cropCySrc - srcY0) / srcPPP;
    const float halfW = std::max(cropHalfWSrc / srcPPP, 1.0f);
    const float halfH = std::max(cropHalfHSrc / srcPPP, 1.0f);

    const float halfD = std::sqrt(halfW * halfW + halfH * halfH);
    // Isotropic normalisation — same scale on both axes so p=2 yields a true
    // circle regardless of aspect.  cornerD normalises dn to 1 at the corner.
    const float nxCorner = halfW / halfD;
    const float nyCorner = halfH / halfD;
    const float cornerD  = std::pow(std::pow(nxCorner, a.p) +
                                    std::pow(nyCorner, a.p), 1.0f / a.p);

    m_kernelLinear.setArg(0,  buf);
    m_kernelLinear.setArg(1,  w);
    m_kernelLinear.setArg(2,  h);
    m_kernelLinear.setArg(3,  cx);     // cx (crop centre in buffer coords)
    m_kernelLinear.setArg(4,  cy);     // cy (crop centre in buffer coords)
    m_kernelLinear.setArg(5,  halfD);  // halfW scale (isotropic, crop-relative)
    m_kernelLinear.setArg(6,  halfD);  // halfH scale (isotropic, crop-relative)
    m_kernelLinear.setArg(7,  a.amount);
    m_kernelLinear.setArg(8,  a.midpoint);
    m_kernelLinear.setArg(9,  a.feather);
    m_kernelLinear.setArg(10, a.p);
    m_kernelLinear.setArg(11, cornerD);
    m_kernelLinear.setArg(12, srcX0);
    m_kernelLinear.setArg(13, srcY0);
    m_kernelLinear.setArg(14, srcPPP);

    queue.enqueueNDRangeKernel(m_kernelLinear, cl::NullRange,
                               cl::NDRange(w, h), cl::NullRange);
    return true;
}

