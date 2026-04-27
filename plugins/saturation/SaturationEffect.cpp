#include "SaturationEffect.h"
#include "ParamSlider.h"
#include "color_kernels.h"
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

// The kernel mirrors the CPU HSV logic exactly.
// Pixels are in QImage::Format_RGB32: each uint = 0xFFRRGGBB.
// stride = bytesPerLine/4 (handles any row padding Qt may add).
// Vibrancy: boost dull colors proportionally to (1 - current_saturation),
// protecting skin tones via a Gaussian falloff centred on hue=20° (orange/peach).
// Saturation: global equal boost, applied after vibrancy.
static const char* GPU_KERNEL_SOURCE = R"CL(
// 8-bit path: pixels are QImage::Format_RGB32 (uint = 0xFFRRGGBB)
__kernel void adjustSatVibrancy(__global uint* pixels,
                                 int   stride,
                                 int   width,
                                 int   height,
                                 float saturationValue,
                                 float vibrancyValue)
{
    int x = get_global_id(0);
    int y = get_global_id(1);
    if (x >= width || y >= height) return;

    uint pixel = pixels[y * stride + x];
    float r = ((pixel >> 16) & 0xFFu) / 255.0f;
    float g = ((pixel >>  8) & 0xFFu) / 255.0f;
    float b = ( pixel        & 0xFFu) / 255.0f;

    // RGB → HSV
    float maxC  = fmax(fmax(r, g), b);
    float delta = maxC - fmin(fmin(r, g), b);
    float v = maxC;
    float s = (maxC == 0.0f) ? 0.0f : (delta / maxC);

    float h = 0.0f;
    if (delta != 0.0f) {
        if      (maxC == r) h = fmod((g - b) / delta, 6.0f);
        else if (maxC == g) h = (b - r) / delta + 2.0f;
        else                h = (r - g) / delta + 4.0f;
        h = fmod(h * 60.0f, 360.0f);
        if (h < 0.0f) h += 360.0f;
    }

    // Vibrancy: weight = (1 - s) * skin_protection
    // Skin tones sit around hue=20°; Gaussian protects them (sigma=25°, 70% max suppression).
    if (vibrancyValue != 0) {
        float hueDist = fabs(h - 20.0f);
        if (hueDist > 180.0f) hueDist = 360.0f - hueDist;
        float skinProtect = exp(-0.5f * hueDist * hueDist / (25.0f * 25.0f));
        float weight = (1.0f - s) * (1.0f - 0.7f * skinProtect);
        s = clamp(s + (vibrancyValue / 100.0f) * weight, 0.0f, 1.0f);
    }

    // Saturation: global equal boost applied after vibrancy
    if (saturationValue != 0.0f) {
        s = clamp(s + saturationValue / 100.0f, 0.0f, 1.0f);
    }

    // HSV → RGB
    float c  = v * s;
    float xv = c * (1.0f - fabs(fmod(h / 60.0f, 2.0f) - 1.0f));
    float m  = v - c;
    float rf, gf, bf;
    if      (h <  60.0f) { rf = c;    gf = xv;   bf = 0.0f; }
    else if (h < 120.0f) { rf = xv;   gf = c;    bf = 0.0f; }
    else if (h < 180.0f) { rf = 0.0f; gf = c;    bf = xv;   }
    else if (h < 240.0f) { rf = 0.0f; gf = xv;   bf = c;    }
    else if (h < 300.0f) { rf = xv;   gf = 0.0f; bf = c;    }
    else                 { rf = c;    gf = 0.0f; bf = xv;   }

    uint ri = (uint)(clamp(rf + m, 0.0f, 1.0f) * 255.0f + 0.5f);
    uint gi = (uint)(clamp(gf + m, 0.0f, 1.0f) * 255.0f + 0.5f);
    uint bi = (uint)(clamp(bf + m, 0.0f, 1.0f) * 255.0f + 0.5f);
    pixels[y * stride + x] = 0xFF000000u | (ri << 16) | (gi << 8) | bi;
}

// 16-bit path: pixels are QImage::Format_RGBX64 (ushort4 per pixel).
// On little-endian: ushort4.s0=R, .s1=G, .s2=B, .s3=A
// stride = bytesPerLine / 8
__kernel void adjustSatVibrancy16(__global ushort4* pixels,
                                   int   stride,
                                   int   width,
                                   int   height,
                                   float saturationValue,
                                   float vibrancyValue)
{
    int x = get_global_id(0);
    int y = get_global_id(1);
    if (x >= width || y >= height) return;

    ushort4 px = pixels[y * stride + x];
    float r = px.s0 / 65535.0f;
    float g = px.s1 / 65535.0f;
    float b = px.s2 / 65535.0f;

    // RGB → HSV (identical to 8-bit kernel)
    float maxC  = fmax(fmax(r, g), b);
    float delta = maxC - fmin(fmin(r, g), b);
    float v = maxC;
    float s = (maxC == 0.0f) ? 0.0f : (delta / maxC);

    float h = 0.0f;
    if (delta != 0.0f) {
        if      (maxC == r) h = fmod((g - b) / delta, 6.0f);
        else if (maxC == g) h = (b - r) / delta + 2.0f;
        else                h = (r - g) / delta + 4.0f;
        h = fmod(h * 60.0f, 360.0f);
        if (h < 0.0f) h += 360.0f;
    }

    if (vibrancyValue != 0) {
        float hueDist = fabs(h - 20.0f);
        if (hueDist > 180.0f) hueDist = 360.0f - hueDist;
        float skinProtect = exp(-0.5f * hueDist * hueDist / (25.0f * 25.0f));
        float weight = (1.0f - s) * (1.0f - 0.7f * skinProtect);
        s = clamp(s + (vibrancyValue / 100.0f) * weight, 0.0f, 1.0f);
    }

    if (saturationValue != 0.0f) {
        s = clamp(s + saturationValue / 100.0f, 0.0f, 1.0f);
    }

    // HSV → RGB
    float c  = v * s;
    float xv = c * (1.0f - fabs(fmod(h / 60.0f, 2.0f) - 1.0f));
    float m  = v - c;
    float rf, gf, bf;
    if      (h <  60.0f) { rf = c;    gf = xv;   bf = 0.0f; }
    else if (h < 120.0f) { rf = xv;   gf = c;    bf = 0.0f; }
    else if (h < 180.0f) { rf = 0.0f; gf = c;    bf = xv;   }
    else if (h < 240.0f) { rf = 0.0f; gf = xv;   bf = c;    }
    else if (h < 300.0f) { rf = xv;   gf = 0.0f; bf = c;    }
    else                 { rf = c;    gf = 0.0f; bf = xv;   }

    px.s0 = (ushort)(clamp(rf + m, 0.0f, 1.0f) * 65535.0f + 0.5f);
    px.s1 = (ushort)(clamp(gf + m, 0.0f, 1.0f) * 65535.0f + 0.5f);
    px.s2 = (ushort)(clamp(bf + m, 0.0f, 1.0f) * 65535.0f + 0.5f);
    // px.s3 (alpha) unchanged
    pixels[y * stride + x] = px;
}
)CL";

struct GpuContext : GpuContextBase<GpuContext> {
    cl::Kernel kernel;
    cl::Kernel kernel16;

    void init() {
        cl::Device device;
        if (!acquireDevice(device, "Saturation")) return;
        try {
            cl::Program prog(context, GPU_KERNEL_SOURCE);
            prog.build({device});
            kernel   = cl::Kernel(prog, "adjustSatVibrancy");
            kernel16 = cl::Kernel(prog, "adjustSatVibrancy16");
            available = true;
            qDebug() << "[GPU] Saturation ready on:"
                     << QString::fromStdString(device.getInfo<CL_DEVICE_NAME>());
        }
        // GCOVR_EXCL_START
        catch (const cl::Error& e) {
            qWarning() << "[GPU] Saturation init failed:" << e.what() << "(err" << e.err() << ")";
        }
        // GCOVR_EXCL_STOP
    }
};

static std::mutex gpuMutex;

static QImage processImageGPU(const QImage& image, float saturationValue, float vibrancyValue) {
    QImage result = image.convertToFormat(QImage::Format_RGB32);
    const int width  = result.width();
    const int height = result.height();
    const int stride = result.bytesPerLine() / 4;
    const size_t bufBytes = static_cast<size_t>(result.bytesPerLine()) * height;

    try {
        std::lock_guard<std::mutex> lock(gpuMutex);
        GpuContext& gpu = GpuContext::instance();
        if (!gpu.available) return {};

        cl::Buffer buf(gpu.context,
                       CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                       bufBytes, result.bits());

        gpu.kernel.setArg(0, buf);
        gpu.kernel.setArg(1, stride);
        gpu.kernel.setArg(2, width);
        gpu.kernel.setArg(3, height);
        gpu.kernel.setArg(4, saturationValue);
        gpu.kernel.setArg(5, vibrancyValue);

        gpu.queue.enqueueNDRangeKernel(gpu.kernel,
                                       cl::NullRange,
                                       cl::NDRange(width, height),
                                       cl::NullRange);
        gpu.queue.finish();
        gpu.queue.enqueueReadBuffer(buf, CL_TRUE, 0, bufBytes, result.bits());
    }
    // GCOVR_EXCL_START
    catch (const cl::Error& e) {
        qWarning() << "[GPU] Kernel failed:" << e.what() << "(err" << e.err() << ")";
        return {};
    }
    // GCOVR_EXCL_STOP
    return result;
}

static QImage processImageGPU16(const QImage& image, float saturationValue, float vibrancyValue) {
    QImage result = image; // already RGBX64
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

        gpu.kernel16.setArg(0, buf);
        gpu.kernel16.setArg(1, stride);
        gpu.kernel16.setArg(2, width);
        gpu.kernel16.setArg(3, height);
        gpu.kernel16.setArg(4, saturationValue);
        gpu.kernel16.setArg(5, vibrancyValue);

        gpu.queue.enqueueNDRangeKernel(gpu.kernel16, cl::NullRange,
                                       cl::NDRange(width, height), cl::NullRange);
        gpu.queue.finish();
        gpu.queue.enqueueReadBuffer(buf, CL_TRUE, 0, bufBytes, result.bits());
    }
    // GCOVR_EXCL_START
    catch (const cl::Error& e) {
        qWarning() << "[GPU] Saturation16 kernel failed:" << e.what() << "(err" << e.err() << ")";
        return {};
    }
    // GCOVR_EXCL_STOP
    return result;
}

} // namespace

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

    // Encode to sRGB-gamma for HSV adjust (matches processImage UI behaviour).
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

QImage SaturationEffect::processImage(const QImage &image, const QMap<QString, QVariant> &parameters) {
    if (image.isNull()) return image;

    float saturationFactor = static_cast<float>(parameters.value("saturation", 0.0).toDouble());
    float vibrancyFactor   = static_cast<float>(parameters.value("vibrancy",   0.0).toDouble());
    if (saturationFactor == 0.0f && vibrancyFactor == 0.0f) return image;

    if (image.format() == QImage::Format_RGBX64)
        return processImageGPU16(image, saturationFactor, vibrancyFactor);
    return processImageGPU(image, saturationFactor, vibrancyFactor);
}
