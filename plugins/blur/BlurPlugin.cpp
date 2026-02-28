#include "BlurPlugin.h"
#include "ParamSlider.h"
#include <QDebug>
#include <QVBoxLayout>
#include <QLabel>
#include <QComboBox>
#include <algorithm>
#include <cmath>
#include <mutex>
#include <vector>

// ============================================================================
// GPU path (OpenCL) — two-pass separable blur (horizontal then vertical)
// ============================================================================
#ifdef HAVE_OPENCL

#define CL_HPP_TARGET_OPENCL_VERSION 200
#define CL_HPP_MINIMUM_OPENCL_VERSION 110
#define CL_HPP_ENABLE_EXCEPTIONS
#include <CL/opencl.hpp>

namespace {

// Pixels are QImage::Format_RGB32: each uint = 0xFFRRGGBB.
// stride = bytesPerLine/4.
// Gaussian sigma = radius/3 (edge weight ≈ exp(-4.5) ≈ 0.01 — negligible).
// Box blur: uniform weights, equivalent to sigma = ∞.
static const char* GPU_KERNEL_SOURCE = R"CL(

__kernel void blurHorizontal(__global const uint* input,
                              __global       uint* output,
                              int stride, int width, int height,
                              int radius, int isGaussian)
{
    int x = get_global_id(0);
    int y = get_global_id(1);
    if (x >= width || y >= height) return;

    float sigma = max((float)radius / 3.0f, 0.5f);
    float r = 0.0f, g = 0.0f, b = 0.0f, wsum = 0.0f;

    for (int dx = -radius; dx <= radius; dx++) {
        int sx = clamp(x + dx, 0, width - 1);
        uint pixel = input[y * stride + sx];
        float w = isGaussian ? exp(-0.5f * dx * dx / (sigma * sigma)) : 1.0f;
        r += w * ((pixel >> 16) & 0xFFu);
        g += w * ((pixel >>  8) & 0xFFu);
        b += w * ( pixel        & 0xFFu);
        wsum += w;
    }

    uint ri = (uint)(r / wsum + 0.5f);
    uint gi = (uint)(g / wsum + 0.5f);
    uint bi = (uint)(b / wsum + 0.5f);
    output[y * stride + x] = 0xFF000000u | (ri << 16) | (gi << 8) | bi;
}

__kernel void blurVertical(__global const uint* input,
                            __global       uint* output,
                            int stride, int width, int height,
                            int radius, int isGaussian)
{
    int x = get_global_id(0);
    int y = get_global_id(1);
    if (x >= width || y >= height) return;

    float sigma = max((float)radius / 3.0f, 0.5f);
    float r = 0.0f, g = 0.0f, b = 0.0f, wsum = 0.0f;

    for (int dy = -radius; dy <= radius; dy++) {
        int sy = clamp(y + dy, 0, height - 1);
        uint pixel = input[sy * stride + x];
        float w = isGaussian ? exp(-0.5f * dy * dy / (sigma * sigma)) : 1.0f;
        r += w * ((pixel >> 16) & 0xFFu);
        g += w * ((pixel >>  8) & 0xFFu);
        b += w * ( pixel        & 0xFFu);
        wsum += w;
    }

    uint ri = (uint)(r / wsum + 0.5f);
    uint gi = (uint)(g / wsum + 0.5f);
    uint bi = (uint)(b / wsum + 0.5f);
    output[y * stride + x] = 0xFF000000u | (ri << 16) | (gi << 8) | bi;
}

)CL";

struct GpuContext {
    cl::Context      context;
    cl::CommandQueue queue;
    cl::Kernel       kernelH;   // horizontal pass
    cl::Kernel       kernelV;   // vertical pass
    bool             available = false;

    static GpuContext& instance() {
        static std::once_flag flag;
        static GpuContext ctx;
        std::call_once(flag, [&]{ ctx.init(); });
        return ctx;
    }

private:
    void init() {
        try {
            std::vector<cl::Platform> platforms;
            cl::Platform::get(&platforms);
            if (platforms.empty()) { qWarning() << "[GPU] No OpenCL platforms"; return; }

            cl::Device device;
            bool found = false;
            for (auto& p : platforms) {
                std::vector<cl::Device> devs;
                try { p.getDevices(CL_DEVICE_TYPE_GPU, &devs); } catch (...) {}
                if (!devs.empty()) { device = devs[0]; found = true; break; }
            }
            if (!found) {
                for (auto& p : platforms) {
                    std::vector<cl::Device> devs;
                    try { p.getDevices(CL_DEVICE_TYPE_ALL, &devs); } catch (...) {}
                    if (!devs.empty()) { device = devs[0]; found = true; break; }
                }
            }
            if (!found) { qWarning() << "[GPU] No OpenCL devices"; return; }

            context = cl::Context(device);
            queue   = cl::CommandQueue(context, device);
            cl::Program prog(context, GPU_KERNEL_SOURCE);
            prog.build({device});
            kernelH   = cl::Kernel(prog, "blurHorizontal");
            kernelV   = cl::Kernel(prog, "blurVertical");
            available = true;
            qDebug() << "[GPU] Blur OpenCL ready on:"
                     << QString::fromStdString(device.getInfo<CL_DEVICE_NAME>());
        } catch (const cl::Error& e) {
            qWarning() << "[GPU] Blur init failed:" << e.what() << "(err" << e.err() << ")";
        }
    }
};

static std::mutex gpuMutex;

static QImage processImageGPU(const QImage& image, int radius, bool gaussian) {
    GpuContext& gpu = GpuContext::instance();
    if (!gpu.available) return {};

    QImage result = image.convertToFormat(QImage::Format_RGB32);
    const int    width    = result.width();
    const int    height   = result.height();
    const int    stride   = result.bytesPerLine() / 4;
    const size_t bufBytes = static_cast<size_t>(result.bytesPerLine()) * height;

    try {
        std::lock_guard<std::mutex> lock(gpuMutex);

        // Two buffers — ping-pong between horizontal and vertical passes
        cl::Buffer bufA(gpu.context,
                        CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                        bufBytes, result.bits());
        cl::Buffer bufB(gpu.context, CL_MEM_READ_WRITE, bufBytes);

        const int isGaussian = gaussian ? 1 : 0;

        // Horizontal: A → B
        gpu.kernelH.setArg(0, bufA);
        gpu.kernelH.setArg(1, bufB);
        gpu.kernelH.setArg(2, stride);
        gpu.kernelH.setArg(3, width);
        gpu.kernelH.setArg(4, height);
        gpu.kernelH.setArg(5, radius);
        gpu.kernelH.setArg(6, isGaussian);
        gpu.queue.enqueueNDRangeKernel(gpu.kernelH, cl::NullRange,
                                       cl::NDRange(width, height), cl::NullRange);
        gpu.queue.finish();

        // Vertical: B → A
        gpu.kernelV.setArg(0, bufB);
        gpu.kernelV.setArg(1, bufA);
        gpu.kernelV.setArg(2, stride);
        gpu.kernelV.setArg(3, width);
        gpu.kernelV.setArg(4, height);
        gpu.kernelV.setArg(5, radius);
        gpu.kernelV.setArg(6, isGaussian);
        gpu.queue.enqueueNDRangeKernel(gpu.kernelV, cl::NullRange,
                                       cl::NDRange(width, height), cl::NullRange);
        gpu.queue.finish();

        // Read result back from A
        gpu.queue.enqueueReadBuffer(bufA, CL_TRUE, 0, bufBytes, result.bits());
    } catch (const cl::Error& e) {
        qWarning() << "[GPU] Blur kernel failed:" << e.what() << "(err" << e.err() << ")";
        return {};
    }
    return result;
}

} // namespace

#endif // HAVE_OPENCL

// ============================================================================
// CPU path — two-pass separable blur
// ============================================================================

static QImage blurCPU(const QImage& image, int radius, bool gaussian) {
    QImage src = image.convertToFormat(QImage::Format_RGB32);
    const int w = src.width();
    const int h = src.height();

    // Precompute normalised 1D kernel
    const int ksize = 2 * radius + 1;
    std::vector<float> kernel(ksize);
    const float sigma = std::max(radius / 3.0f, 0.5f);
    float wsum = 0.0f;
    for (int i = 0; i < ksize; i++) {
        const int d = i - radius;
        kernel[i] = gaussian ? std::exp(-0.5f * d * d / (sigma * sigma)) : 1.0f;
        wsum += kernel[i];
    }
    for (auto& v : kernel) v /= wsum;

    // Horizontal pass: src → tmp
    QImage tmp(w, h, QImage::Format_RGB32);
    for (int y = 0; y < h; y++) {
        const QRgb* srcRow = reinterpret_cast<const QRgb*>(src.constScanLine(y));
        QRgb*       dstRow = reinterpret_cast<QRgb*>(tmp.scanLine(y));
        for (int x = 0; x < w; x++) {
            float r = 0, g = 0, b = 0;
            for (int k = 0; k < ksize; k++) {
                const int sx = std::clamp(x + k - radius, 0, w - 1);
                const QRgb px = srcRow[sx];
                r += kernel[k] * qRed(px);
                g += kernel[k] * qGreen(px);
                b += kernel[k] * qBlue(px);
            }
            dstRow[x] = qRgb(static_cast<int>(r + 0.5f),
                             static_cast<int>(g + 0.5f),
                             static_cast<int>(b + 0.5f));
        }
    }

    // Vertical pass: tmp → dst
    QImage dst(w, h, QImage::Format_RGB32);
    for (int y = 0; y < h; y++) {
        QRgb* dstRow = reinterpret_cast<QRgb*>(dst.scanLine(y));
        for (int x = 0; x < w; x++) {
            float r = 0, g = 0, b = 0;
            for (int k = 0; k < ksize; k++) {
                const int sy = std::clamp(y + k - radius, 0, h - 1);
                const QRgb px = reinterpret_cast<const QRgb*>(tmp.constScanLine(sy))[x];
                r += kernel[k] * qRed(px);
                g += kernel[k] * qGreen(px);
                b += kernel[k] * qBlue(px);
            }
            dstRow[x] = qRgb(static_cast<int>(r + 0.5f),
                             static_cast<int>(g + 0.5f),
                             static_cast<int>(b + 0.5f));
        }
    }

    return dst;
}

// ============================================================================
// Plugin implementation
// ============================================================================

BlurPlugin::BlurPlugin()
    : controlsWidget(nullptr), blurTypeCombo(nullptr),
      radiusParam(nullptr), blurType(0) {
}

BlurPlugin::~BlurPlugin() {
}

QString BlurPlugin::getName() const        { return "Blur"; }
QString BlurPlugin::getDescription() const { return "Gaussian and box blur"; }
QString BlurPlugin::getVersion() const     { return "1.0.0"; }

bool BlurPlugin::initialize() {
    qDebug() << "Blur plugin initialized";
    return true;
}

QWidget* BlurPlugin::createControlsWidget() {
    if (controlsWidget) return controlsWidget;

    controlsWidget = new QWidget();
    QVBoxLayout* layout = new QVBoxLayout(controlsWidget);
    layout->setContentsMargins(0, 0, 0, 0);
    layout->setSpacing(8);

    // Blur type
    QLabel* typeLabel = new QLabel("Blur type:");
    typeLabel->setStyleSheet("color: #e0e0e0;");
    layout->addWidget(typeLabel);

    blurTypeCombo = new QComboBox();
    blurTypeCombo->addItem("Gaussian");
    blurTypeCombo->addItem("Box");
    blurTypeCombo->setStyleSheet(
        "QComboBox { color: #e0e0e0; background-color: #444444;"
        "            border: 1px solid #666666; border-radius: 3px; padding: 3px; }"
        "QComboBox::drop-down { border: none; }"
        "QComboBox QAbstractItemView { color: #e0e0e0; background-color: #444444; }");
    layout->addWidget(blurTypeCombo);

    connect(blurTypeCombo, QOverload<int>::of(&QComboBox::activated), this, [this](int index) {
        blurType = index;
        emit parametersChanged();
    });

    // Radius
    radiusParam = new ParamSlider("Radius", 0, 50);
    connect(radiusParam, &ParamSlider::valueChanged, this, [this](double) {
        emit parametersChanged();
    });
    layout->addWidget(radiusParam);

    layout->addStretch();
    return controlsWidget;
}

QMap<QString, QVariant> BlurPlugin::getParameters() const {
    QMap<QString, QVariant> params;
    params["blurType"] = blurType;
    params["radius"]   = radiusParam ? static_cast<int>(radiusParam->value()) : 0;
    return params;
}

QImage BlurPlugin::processImage(const QImage& image, const QMap<QString, QVariant>& parameters) {
    if (image.isNull()) return image;

    const int radius    = parameters.value("radius",   0).toInt();
    const int type      = parameters.value("blurType", 0).toInt();
    const bool gaussian = (type == 0);

    if (radius == 0) return image;

#ifdef HAVE_OPENCL
    QImage gpuResult = processImageGPU(image, radius, gaussian);
    if (!gpuResult.isNull()) return gpuResult;
    qDebug() << "[GPU] Falling back to CPU for blur";
#endif

    return blurCPU(image, radius, gaussian);
}

extern "C" {
    PhotoEditorPlugin* createPlugin()             { return new BlurPlugin(); }
    void destroyPlugin(PhotoEditorPlugin* plugin)  { delete plugin; }
}
