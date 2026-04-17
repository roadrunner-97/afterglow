#include "FilmGrainEffect.h"
#include "ParamSlider.h"
#include <QCheckBox>
#include <QDebug>
#include <QVBoxLayout>
#include <mutex>

// ============================================================================
// GPU path (OpenCL)
// ============================================================================

#define CL_HPP_TARGET_OPENCL_VERSION 200
#define CL_HPP_MINIMUM_OPENCL_VERSION 110
#define CL_HPP_ENABLE_EXCEPTIONS
#include <CL/opencl.hpp>
#include "GpuDeviceRegistryOCL.h"
#include "GpuContextBase.h"

namespace {

// Hash-based per-pixel noise.
//
// Pixel coordinates are quantised to grain-size blocks so neighbouring pixels
// within a block share a noise value — a cheap way to get grain size > 1 px
// without a separate blur pass.  The hash mixes the block coordinates with a
// fixed seed via a small PCG-style integer hash, yielding a uniform [-1, 1]
// noise value that is added to each RGB channel.
//
// If lumWeight is non-zero the noise is multiplied by 4*L*(1-L), a tent
// function peaking at midtones (L=0.5) and vanishing at shadows/highlights.
// This mimics how real film grain is least visible in pure-black or pure-white
// regions.
//
// amount is the additive noise range in normalised [0..1] intensity units; at
// amount=1 a midtone pixel with lumWeight can swing by up to ±1 (fully clamped).
static const char* GPU_KERNEL_SOURCE = R"CL(
inline uint pcg_hash(uint x, uint y, uint seed) {
    uint h = x * 374761393u + y * 668265263u + seed * 3266489917u;
    h = (h ^ (h >> 13)) * 1274126177u;
    return h ^ (h >> 16);
}

inline float grain_noise(int x, int y, int size, uint seed) {
    uint bx = (uint)(x / size);
    uint by = (uint)(y / size);
    uint h  = pcg_hash(bx, by, seed);
    // map to [-1, 1]
    return (float)h * (2.0f / 4294967295.0f) - 1.0f;
}

__kernel void applyFilmGrain(__global uint* pixels,
                              int   stride,
                              int   width,
                              int   height,
                              int   size,
                              float amount,
                              int   lumWeight,
                              uint  seed)
{
    int x = get_global_id(0);
    int y = get_global_id(1);
    if (x >= width || y >= height) return;

    uint pixel = pixels[y * stride + x];
    float r = ((pixel >> 16) & 0xFFu) / 255.0f;
    float g = ((pixel >>  8) & 0xFFu) / 255.0f;
    float b = ( pixel        & 0xFFu) / 255.0f;

    float n = grain_noise(x, y, size, seed);
    float w = 1.0f;
    if (lumWeight) {
        float L = 0.299f * r + 0.587f * g + 0.114f * b;
        w = 4.0f * L * (1.0f - L);
    }
    float d = n * amount * w;

    r = clamp(r + d, 0.0f, 1.0f);
    g = clamp(g + d, 0.0f, 1.0f);
    b = clamp(b + d, 0.0f, 1.0f);

    uint ri = (uint)(r * 255.0f + 0.5f);
    uint gi = (uint)(g * 255.0f + 0.5f);
    uint bi = (uint)(b * 255.0f + 0.5f);
    pixels[y * stride + x] = 0xFF000000u | (ri << 16) | (gi << 8) | bi;
}

__kernel void applyFilmGrain16(__global ushort4* pixels,
                                int   stride,
                                int   width,
                                int   height,
                                int   size,
                                float amount,
                                int   lumWeight,
                                uint  seed)
{
    int x = get_global_id(0);
    int y = get_global_id(1);
    if (x >= width || y >= height) return;

    ushort4 px = pixels[y * stride + x];
    float r = px.s0 / 65535.0f;
    float g = px.s1 / 65535.0f;
    float b = px.s2 / 65535.0f;

    float n = grain_noise(x, y, size, seed);
    float w = 1.0f;
    if (lumWeight) {
        float L = 0.299f * r + 0.587f * g + 0.114f * b;
        w = 4.0f * L * (1.0f - L);
    }
    float d = n * amount * w;

    r = clamp(r + d, 0.0f, 1.0f);
    g = clamp(g + d, 0.0f, 1.0f);
    b = clamp(b + d, 0.0f, 1.0f);

    px.s0 = (ushort)(r * 65535.0f + 0.5f);
    px.s1 = (ushort)(g * 65535.0f + 0.5f);
    px.s2 = (ushort)(b * 65535.0f + 0.5f);
    pixels[y * stride + x] = px;
}
)CL";

struct GpuContext : GpuContextBase<GpuContext> {
    cl::Kernel kernel;
    cl::Kernel kernel16;

    void init() {
        cl::Device device;
        if (!acquireDevice(device, "FilmGrain")) return;
        try {
            cl::Program prog(context, GPU_KERNEL_SOURCE);
            prog.build({device});
            kernel   = cl::Kernel(prog, "applyFilmGrain");
            kernel16 = cl::Kernel(prog, "applyFilmGrain16");
            available = true;
            qDebug() << "[GPU] FilmGrain ready on:"
                     << QString::fromStdString(device.getInfo<CL_DEVICE_NAME>());
        }
        // GCOVR_EXCL_START
        catch (const cl::Error& e) {
            qWarning() << "[GPU] FilmGrain init failed:" << e.what() << "(err" << e.err() << ")";
        }
        // GCOVR_EXCL_STOP
    }
};

static std::mutex gpuMutex;

struct GrainArgs {
    int      size;        // block side in pixels, ≥1
    float    amount;      // 0..1 (slider 0..100 → /100)
    int      lumWeight;   // 0 or 1
    unsigned seed;
};

static GrainArgs makeArgs(int amount, int size, bool lumWeight) {
    GrainArgs a;
    a.size      = size < 1 ? 1 : size;
    a.amount    = amount / 100.0f;
    a.lumWeight = lumWeight ? 1 : 0;
    a.seed      = 0xDEADBEEFu;
    return a;
}

static void setKernelArgs(cl::Kernel& k, cl::Buffer& buf,
                           int stride, int w, int h, const GrainArgs& a) {
    k.setArg(0, buf);
    k.setArg(1, stride);
    k.setArg(2, w);
    k.setArg(3, h);
    k.setArg(4, a.size);
    k.setArg(5, a.amount);
    k.setArg(6, a.lumWeight);
    k.setArg(7, a.seed);
}

static QImage processImageGPU(const QImage& image, const GrainArgs& a) {
    QImage result = image.convertToFormat(QImage::Format_RGB32);
    const int    width    = result.width();
    const int    height   = result.height();
    const int    stride   = result.bytesPerLine() / 4;
    const size_t bufBytes = static_cast<size_t>(result.bytesPerLine()) * height;

    try {
        std::lock_guard<std::mutex> lock(gpuMutex);
        GpuContext& gpu = GpuContext::instance();
        if (!gpu.available) return {};

        cl::Buffer buf(gpu.context,
                       CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                       bufBytes, result.bits());

        setKernelArgs(gpu.kernel, buf, stride, width, height, a);
        gpu.queue.enqueueNDRangeKernel(gpu.kernel, cl::NullRange,
                                       cl::NDRange(width, height), cl::NullRange);
        gpu.queue.finish();
        gpu.queue.enqueueReadBuffer(buf, CL_TRUE, 0, bufBytes, result.bits());
    }
    // GCOVR_EXCL_START
    catch (const cl::Error& e) {
        qWarning() << "[GPU] FilmGrain kernel failed:" << e.what() << "(err" << e.err() << ")";
        return {};
    }
    // GCOVR_EXCL_STOP
    return result;
}

static QImage processImageGPU16(const QImage& image, const GrainArgs& a) {
    QImage result = image;
    const int    width    = result.width();
    const int    height   = result.height();
    const int    stride   = result.bytesPerLine() / 8;
    const size_t bufBytes = static_cast<size_t>(result.bytesPerLine()) * height;

    try {
        std::lock_guard<std::mutex> lock(gpuMutex);
        GpuContext& gpu = GpuContext::instance();
        if (!gpu.available) return {};

        cl::Buffer buf(gpu.context,
                       CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                       bufBytes, result.bits());

        setKernelArgs(gpu.kernel16, buf, stride, width, height, a);
        gpu.queue.enqueueNDRangeKernel(gpu.kernel16, cl::NullRange,
                                       cl::NDRange(width, height), cl::NullRange);
        gpu.queue.finish();
        gpu.queue.enqueueReadBuffer(buf, CL_TRUE, 0, bufBytes, result.bits());
    }
    // GCOVR_EXCL_START
    catch (const cl::Error& e) {
        qWarning() << "[GPU] FilmGrain16 kernel failed:" << e.what() << "(err" << e.err() << ")";
        return {};
    }
    // GCOVR_EXCL_STOP
    return result;
}

} // namespace

// ============================================================================
// FilmGrainEffect
// ============================================================================

FilmGrainEffect::FilmGrainEffect()
    : controlsWidget(nullptr), amountParam(nullptr), sizeParam(nullptr),
      lumWeightBox(nullptr) {
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

    amountParam = new ParamSlider("Amount", 0, 100);
    amountParam->setToolTip("Strength of the grain.\n0 disables the effect.");
    connectSlider(amountParam);
    layout->addWidget(amountParam);

    sizeParam = new ParamSlider("Size", 1, 5);
    sizeParam->setValue(1);
    sizeParam->setToolTip("Grain size in pixels.\nLarger values produce a coarser pattern.");
    connectSlider(sizeParam);
    layout->addWidget(sizeParam);

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
    params["size"]      = static_cast<int>(sizeParam   ? sizeParam->value()   : 1.0);
    params["lumWeight"] = lumWeightBox ? lumWeightBox->isChecked() : true;
    return params;
}

// ============================================================================
// IGpuEffect — shared pipeline interface
// ============================================================================

bool FilmGrainEffect::initGpuKernels(cl::Context& ctx, cl::Device& dev) {
    try {
        cl::Program prog(ctx, GPU_KERNEL_SOURCE);
        prog.build({dev});
        m_kernel   = cl::Kernel(prog, "applyFilmGrain");
        m_kernel16 = cl::Kernel(prog, "applyFilmGrain16");
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
                                  int w, int h, int stride, bool is16bit,
                                  const QMap<QString, QVariant>& params) {
    const int amount = params.value("amount", 0).toInt();
    if (amount == 0) return true;  // no-op

    const GrainArgs a = makeArgs(
        amount,
        params.value("size", 1).toInt(),
        params.value("lumWeight", true).toBool());

    cl::Kernel& k = is16bit ? m_kernel16 : m_kernel;
    setKernelArgs(k, buf, stride, w, h, a);
    queue.enqueueNDRangeKernel(k, cl::NullRange, cl::NDRange(w, h), cl::NullRange);
    return true;
}

QImage FilmGrainEffect::processImage(const QImage& image, const QMap<QString, QVariant>& parameters) {
    if (image.isNull()) return image;

    const int amount = parameters.value("amount", 0).toInt();
    if (amount == 0) return image;

    const GrainArgs a = makeArgs(
        amount,
        parameters.value("size", 1).toInt(),
        parameters.value("lumWeight", true).toBool());

    if (image.format() == QImage::Format_RGBX64)
        return processImageGPU16(image, a);
    return processImageGPU(image, a);
}
