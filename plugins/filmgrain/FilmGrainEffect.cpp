#include "FilmGrainEffect.h"
#include "ParamSlider.h"
#include "color_kernels.h"
#include <QCheckBox>
#include <QDebug>
#include <QVBoxLayout>


namespace {

// Value-noise film grain, evaluated in source-image pixel space so the
// pattern stays anchored to the image regardless of zoom / pan.
//
// The kernel's global id (x, y) is a PREVIEW pixel.  We map it back to a
// source-image coordinate using the crop origin (srcX0, srcY0) and the
// preview-to-source pixel ratio (srcPPP) that GpuPipeline already supplies
// to all effects.  The noise lattice lives at intervals of `size` source
// pixels, and bilinear+smootherstep interpolation produces smoothly varying
// values rather than the old per-block on/off pattern.
//
// If lumWeight is non-zero the noise is multiplied by 4*L*(1-L) — a tent
// peaking at midtones, vanishing at pure black / white — which mimics how
// real film grain is least visible in deep shadows / highlights.

struct GrainArgs {
    float    size;        // lattice spacing in source pixels, ≥1
    float    amount;      // 0..1 (slider 0..40 → /100)
    int      lumWeight;   // 0 or 1
    unsigned seed;        // hashed user seed
    float    srcX0;       // source-pixel origin of preview pixel (0, 0)
    float    srcY0;
    float    srcPPP;      // source pixels per preview pixel (1.0 when run on full-res)
};

static GrainArgs makeArgs(int amount, int size, bool lumWeight, int userSeed,
                          double srcX0 = 0.0, double srcY0 = 0.0, double srcPPP = 1.0) {
    GrainArgs a;
    a.size      = static_cast<float>(size < 1 ? 1 : size);
    a.amount    = amount / 100.0f;
    a.lumWeight = lumWeight ? 1 : 0;
    // Mix the user's seed into a well-distributed 32-bit value so adjacent
    // integers (0, 1, 2, …) produce visibly different patterns.
    a.seed      = static_cast<unsigned>(userSeed) * 2654435761u + 0xDEADBEEFu;
    a.srcX0     = static_cast<float>(srcX0);
    a.srcY0     = static_cast<float>(srcY0);
    a.srcPPP    = static_cast<float>(srcPPP);
    return a;
}

} // namespace

// ============================================================================
// Pipeline kernel — float4 linear sRGB, used by GpuPipeline via enqueueGpu.
// Grain is added in sRGB-encoded ("density") space for behavioural parity
// with the 8-bit / 16-bit kernels above and with real film behaviour
// (amplitude is perceptually meaningful in gamma space, not linear).  Image
// coordinates are source-anchored via _cropX0/_cropY0/_srcPixelsPerPreviewPixel
// so the grain pattern doesn't crawl with zoom / pan.
// ============================================================================
static const char* PIPELINE_KERNEL_SOURCE = COLOR_KERNELS_SRC R"CL(
inline uint pcg_hash(uint x, uint y, uint seed) {
    uint h = x * 374761393u + y * 668265263u + seed * 3266489917u;
    h = (h ^ (h >> 13)) * 1274126177u;
    return h ^ (h >> 16);
}

inline float hash_lattice(int x, int y, uint seed) {
    return (float)pcg_hash((uint)x, (uint)y, seed) * (2.0f / 4294967295.0f) - 1.0f;
}

inline float smootherstep_f(float t) {
    return t * t * t * (t * (t * 6.0f - 15.0f) + 10.0f);
}

inline float value_noise(float x, float y, uint seed) {
    int   x0 = (int)floor(x);
    int   y0 = (int)floor(y);
    float fx = x - (float)x0;
    float fy = y - (float)y0;
    float v00 = hash_lattice(x0,     y0,     seed);
    float v10 = hash_lattice(x0 + 1, y0,     seed);
    float v01 = hash_lattice(x0,     y0 + 1, seed);
    float v11 = hash_lattice(x0 + 1, y0 + 1, seed);
    float u = smootherstep_f(fx);
    float v = smootherstep_f(fy);
    return mix(mix(v00, v10, u), mix(v01, v11, u), v);
}

inline float fbm_aa(float x, float y, uint seed, float srcPPP, float sizef) {
    const int OCTAVES = 3;
    float acc     = 0.0f;
    float ampSum  = 0.0f;
    float freq    = 1.0f;
    float amp     = 1.0f;
    uint octaveSeeds[3] = { seed,
                            seed ^ 0x85EBCA6Bu,
                            seed ^ 0xC2B2AE35u };
    for (int i = 0; i < OCTAVES; ++i) {
        float latticePrevPx = sizef / (freq * srcPPP);
        float w = smoothstep(0.5f, 1.0f, latticePrevPx);
        if (w > 0.0f) {
            acc    += value_noise(x * freq, y * freq, octaveSeeds[i]) * amp * w;
            ampSum += amp * w;
        }
        freq *= 2.0f;
        amp  *= 0.5f;
    }
    if (ampSum < 0.001f) return 0.0f;
    return acc / ampSum;
}

inline float grain_sample(int px, int py,
                          float srcX0, float srcY0, float srcPPP,
                          float sizef, uint seed) {
    float sx = srcX0 + ((float)px + 0.5f) * srcPPP;
    float sy = srcY0 + ((float)py + 0.5f) * srcPPP;
    return fbm_aa(sx / sizef, sy / sizef, seed, srcPPP, sizef);
}

__kernel void applyFilmGrainLinear(__global float4* pixels,
                                    int   w,
                                    int   h,
                                    float sizef,
                                    float amount,
                                    int   lumWeight,
                                    uint  seed,
                                    float srcX0,
                                    float srcY0,
                                    float srcPPP)
{
    int x = get_global_id(0);
    int y = get_global_id(1);
    if (x >= w || y >= h) return;

    float4 px = pixels[y * w + x];

    // Encode to sRGB to add grain in gamma / density space — amplitude is
    // perceptually meaningful there and matches the film-grain feel.
    float4 srgb = linear_to_srgb4(px);

    float n = grain_sample(x, y, srcX0, srcY0, srcPPP, sizef, seed);
    float wgt = 1.0f;
    if (lumWeight) {
        // Luma weighting in sRGB space (matches the standalone 8-bit kernel that defined the UI semantics)
        // (which use 0.299/0.587/0.114 on already-gamma data).
        float L = 0.299f * srgb.x + 0.587f * srgb.y + 0.114f * srgb.z;
        wgt = 4.0f * L * (1.0f - L);
    }
    float d = n * amount * wgt;
    // Clamp in sRGB space before decoding: srgb_to_linear uses native_powr
    // which is undefined for negative values, and values > 1 would push
    // beyond what sRGB represents.
    srgb.x = clamp(srgb.x + d, 0.0f, 1.0f);
    srgb.y = clamp(srgb.y + d, 0.0f, 1.0f);
    srgb.z = clamp(srgb.z + d, 0.0f, 1.0f);

    float4 lin = srgb_to_linear4(srgb);
    pixels[y * w + x] = (float4)(lin.x, lin.y, lin.z, 1.0f);
}
)CL";

// ============================================================================
// FilmGrainEffect
// ============================================================================

FilmGrainEffect::FilmGrainEffect()
    : controlsWidget(nullptr), amountParam(nullptr), sizeParam(nullptr),
      seedParam(nullptr), lumWeightBox(nullptr) {
}

FilmGrainEffect::~FilmGrainEffect() {
}

QString FilmGrainEffect::getName() const { return "Film Grain"; }
QString FilmGrainEffect::getDescription() const {
    return "Additive per-pixel noise with optional luminance weighting";
}
QString FilmGrainEffect::getVersion() const { return "1.0.0"; }

bool FilmGrainEffect::initialize() {
    qDebug() << "FilmGrain effect initialized";
    return true;
}

QWidget* FilmGrainEffect::createControlsWidget() {
    if (controlsWidget) return controlsWidget;

    controlsWidget = new QWidget();
    QVBoxLayout* layout = new QVBoxLayout(controlsWidget);
    layout->setContentsMargins(0, 0, 0, 0);
    layout->setSpacing(8);

    auto connectSlider = [&](ParamSlider* s) {
        connect(s, &ParamSlider::editingFinished, this, [this]() { emit parametersChanged(); });
        connect(s, &ParamSlider::valueChanged,    this, [this](double) { emit liveParametersChanged(); });
    };

    amountParam = new ParamSlider("Amount", 0, 40);
    amountParam->setToolTip("Strength of the grain.\n0 disables the effect.");
    connectSlider(amountParam);
    layout->addWidget(amountParam);

    sizeParam = new ParamSlider("Size", 1, 50);
    sizeParam->setValue(8);
    sizeParam->setToolTip("Grain size in source-image pixels.\n"
                          "Larger values produce a coarser pattern.\n"
                          "Anchored to the image — grain does not scale with zoom.");
    connectSlider(sizeParam);
    layout->addWidget(sizeParam);

    seedParam = new ParamSlider("Seed", 0, 999);
    seedParam->setValue(0);
    seedParam->setToolTip("PRNG seed — change to get a different grain pattern.");
    connectSlider(seedParam);
    layout->addWidget(seedParam);

    lumWeightBox = new QCheckBox("Luminance-weighted");
    lumWeightBox->setChecked(true);
    lumWeightBox->setToolTip("When enabled, grain is strongest in the midtones\nand fades toward pure black or white — like real film.");
    connect(lumWeightBox, &QCheckBox::toggled, this, [this](bool) { emit parametersChanged(); });
    layout->addWidget(lumWeightBox);

    layout->addStretch();
    return controlsWidget;
}

QMap<QString, QVariant> FilmGrainEffect::getParameters() const {
    QMap<QString, QVariant> params;
    params["amount"]    = static_cast<int>(amountParam ? amountParam->value() : 0.0);
    params["size"]      = static_cast<int>(sizeParam   ? sizeParam->value()   : 8.0);
    params["seed"]      = static_cast<int>(seedParam   ? seedParam->value()   : 0.0);
    params["lumWeight"] = lumWeightBox ? lumWeightBox->isChecked() : true;
    return params;
}

void FilmGrainEffect::applyParameters(const QMap<QString, QVariant>& parameters) {
    if (amountParam && parameters.contains("amount"))
        amountParam->setValue(parameters.value("amount").toDouble());
    if (sizeParam && parameters.contains("size"))
        sizeParam->setValue(parameters.value("size").toDouble());
    if (seedParam && parameters.contains("seed"))
        seedParam->setValue(parameters.value("seed").toDouble());
    if (lumWeightBox && parameters.contains("lumWeight")) {
        QSignalBlocker block(lumWeightBox);
        lumWeightBox->setChecked(parameters.value("lumWeight").toBool());
    }
    emit parametersChanged();
}

// ============================================================================
// IGpuEffect — shared pipeline interface
// ============================================================================

bool FilmGrainEffect::initGpuKernels(cl::Context& ctx, cl::Device& dev) {
    try {
        cl::Program prog(ctx, PIPELINE_KERNEL_SOURCE);
        prog.build({dev});
        m_kernelLinear = cl::Kernel(prog, "applyFilmGrainLinear");
        return true;
    }
    // GCOVR_EXCL_START
    catch (const cl::Error& e) {
        qWarning() << "[GpuPipeline] FilmGrain initGpuKernels failed:"
                   << e.what() << "(err" << e.err() << ")";
        return false;
    }
    // GCOVR_EXCL_STOP
}

bool FilmGrainEffect::enqueueGpu(cl::CommandQueue& queue,
                                  cl::Buffer& buf, cl::Buffer& /*aux*/,
                                  int w, int h,
                                  const QMap<QString, QVariant>& params) {
    const int amount = params.value("amount", 0).toInt();
    if (amount == 0) return true;  // no-op

    const GrainArgs a = makeArgs(
        amount,
        params.value("size", 8).toInt(),
        params.value("lumWeight", true).toBool(),
        params.value("seed", 0).toInt(),
        params.value("_cropX0", 0.0).toDouble(),
        params.value("_cropY0", 0.0).toDouble(),
        params.value("_srcPixelsPerPreviewPixel", 1.0).toDouble());

    m_kernelLinear.setArg(0,  buf);
    m_kernelLinear.setArg(1,  w);
    m_kernelLinear.setArg(2,  h);
    m_kernelLinear.setArg(3,  a.size);
    m_kernelLinear.setArg(4,  a.amount);
    m_kernelLinear.setArg(5,  a.lumWeight);
    m_kernelLinear.setArg(6,  a.seed);
    m_kernelLinear.setArg(7,  a.srcX0);
    m_kernelLinear.setArg(8,  a.srcY0);
    m_kernelLinear.setArg(9,  a.srcPPP);
    queue.enqueueNDRangeKernel(m_kernelLinear, cl::NullRange,
                               cl::NDRange(w, h), cl::NullRange);
    return true;
}

