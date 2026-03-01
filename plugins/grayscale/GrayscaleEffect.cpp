#include "GrayscaleEffect.h"
#include <QCheckBox>
#include <QVBoxLayout>
#include <QDebug>
#include <mutex>

// ============================================================================
// GPU path (OpenCL)
// ============================================================================

#define CL_HPP_TARGET_OPENCL_VERSION 200
#define CL_HPP_MINIMUM_OPENCL_VERSION 110
#define CL_HPP_ENABLE_EXCEPTIONS
#include <CL/opencl.hpp>
#include "GpuDeviceRegistryOCL.h"

namespace {

static const char* GPU_KERNEL_SOURCE = R"CL(
// 8-bit path: pixels are QImage::Format_RGB32 (uint = 0xFFRRGGBB)
__kernel void grayscale(__global uint* pixels, int stride, int width, int height)
{
    int x = get_global_id(0);
    int y = get_global_id(1);
    if (x >= width || y >= height) return;

    uint pixel = pixels[y * stride + x];
    uint r = (pixel >> 16) & 0xFFu;
    uint g = (pixel >>  8) & 0xFFu;
    uint b =  pixel        & 0xFFu;

    // Luminosity formula, rounded
    uint gray = (uint)(0.299f * r + 0.587f * g + 0.114f * b + 0.5f);
    pixels[y * stride + x] = 0xFF000000u | (gray << 16) | (gray << 8) | gray;
}

// 16-bit path: pixels are QImage::Format_RGBX64 (ushort4 per pixel).
// On little-endian: ushort4.s0=R, .s1=G, .s2=B, .s3=A
// stride = bytesPerLine / 8
__kernel void grayscale16(__global ushort4* pixels, int stride, int width, int height)
{
    int x = get_global_id(0);
    int y = get_global_id(1);
    if (x >= width || y >= height) return;

    ushort4 px = pixels[y * stride + x];
    ushort gray = (ushort)(0.299f * px.s0 + 0.587f * px.s1 + 0.114f * px.s2 + 0.5f);
    px.s0 = gray; px.s1 = gray; px.s2 = gray;
    // px.s3 (alpha) unchanged
    pixels[y * stride + x] = px;
}
)CL";

struct GpuContext {
    cl::Context      context;
    cl::CommandQueue queue;
    cl::Kernel       kernel;
    cl::Kernel       kernel16;
    bool             available = false;
    int              m_revision = 0;

    // Must be called with gpuMutex held.
    static GpuContext& instance() {
        static GpuContext ctx;
        int rev = GpuDeviceRegistry::instance().revision();
        if (ctx.m_revision != rev) {
            ctx            = GpuContext{};
            ctx.m_revision = rev;
            ctx.init();
        }
        return ctx;
    }

private:
    void init() {
        cl::Device   device;
        cl::Platform platform;
        if (!GpuDeviceRegistryOCL::getSelectedDevice(device, platform)) {
            qWarning() << "[GPU] Grayscale: no OpenCL device available";
            return;
        }
        try {
            context = cl::Context(device);
            queue   = cl::CommandQueue(context, device);
            cl::Program prog(context, GPU_KERNEL_SOURCE);
            prog.build({device});
            kernel    = cl::Kernel(prog, "grayscale");
            kernel16  = cl::Kernel(prog, "grayscale16");
            available = true;
            qDebug() << "[GPU] Grayscale ready on:"
                     << QString::fromStdString(device.getInfo<CL_DEVICE_NAME>());
        } catch (const cl::Error& e) {
            qWarning() << "[GPU] Grayscale init failed:" << e.what() << "(err" << e.err() << ")";
        }
    }
};

static std::mutex gpuMutex;

static QImage processImageGPU(const QImage& image) {
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

        gpu.kernel.setArg(0, buf);
        gpu.kernel.setArg(1, stride);
        gpu.kernel.setArg(2, width);
        gpu.kernel.setArg(3, height);

        gpu.queue.enqueueNDRangeKernel(gpu.kernel, cl::NullRange,
                                       cl::NDRange(width, height), cl::NullRange);
        gpu.queue.finish();
        gpu.queue.enqueueReadBuffer(buf, CL_TRUE, 0, bufBytes, result.bits());
    } catch (const cl::Error& e) {
        qWarning() << "[GPU] Grayscale kernel failed:" << e.what() << "(err" << e.err() << ")";
        return {};
    }
    return result;
}

} // namespace

// ============================================================================
// IGpuEffect — shared pipeline interface
// ============================================================================

bool GrayscaleEffect::initGpuKernels(cl::Context& ctx, cl::Device& dev) {
    try {
        cl::Program prog(ctx, GPU_KERNEL_SOURCE);
        prog.build({dev});
        m_kernel   = cl::Kernel(prog, "grayscale");
        m_kernel16 = cl::Kernel(prog, "grayscale16");
        return true;
    } catch (const cl::Error& e) {
        qWarning() << "[GpuPipeline] Grayscale initGpuKernels failed:"
                   << e.what() << "(err" << e.err() << ")";
        return false;
    }
}

bool GrayscaleEffect::enqueueGpu(cl::CommandQueue& queue,
                                  cl::Buffer& buf, cl::Buffer& /*aux*/,
                                  int w, int h, int stride, bool is16bit,
                                  const QMap<QString, QVariant>& /*params*/) {
    if (!m_active) return true;  // no-op: pass buffer through unchanged
    cl::Kernel& k = is16bit ? m_kernel16 : m_kernel;
    k.setArg(0, buf);
    k.setArg(1, stride);
    k.setArg(2, w);
    k.setArg(3, h);
    queue.enqueueNDRangeKernel(k, cl::NullRange, cl::NDRange(w, h), cl::NullRange);
    return true;
}

GrayscaleEffect::GrayscaleEffect() {}

GrayscaleEffect::~GrayscaleEffect() {}

QString GrayscaleEffect::getName() const {
    return "Grayscale";
}

QString GrayscaleEffect::getDescription() const {
    return "Converts the image to grayscale";
}

QString GrayscaleEffect::getVersion() const {
    return "2.0.0";
}

bool GrayscaleEffect::initialize() {
    qDebug() << "Grayscale effect initialized";
    return true;
}

QWidget* GrayscaleEffect::createControlsWidget() {
    QWidget* w = new QWidget();
    QVBoxLayout* layout = new QVBoxLayout(w);
    layout->setContentsMargins(0, 2, 0, 2);

    QCheckBox* check = new QCheckBox("Convert to Grayscale");
    check->setStyleSheet("color: #2C2018;");
    check->setChecked(m_active);
    connect(check, &QCheckBox::toggled, this, [this](bool on) {
        m_active = on;
        emit parametersChanged();
    });
    layout->addWidget(check);
    return w;
}

QImage GrayscaleEffect::processImage(const QImage &image, const QMap<QString, QVariant> &) {
    if (image.isNull() || !m_active) return image;
    // GPU path operates on 8-bit RGB32; converts RGBX64 input automatically
    return processImageGPU(image);
}
