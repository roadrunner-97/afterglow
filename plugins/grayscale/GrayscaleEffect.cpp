#include "GrayscaleEffect.h"
#include "color_kernels.h"
#include <QCheckBox>
#include <QVBoxLayout>
#include <QDebug>

// ============================================================================
// Pipeline kernel — float4 linear sRGB, used by GpuPipeline via enqueueGpu.
// Grayscale in linear light: L = 0.2126 R + 0.7152 G + 0.0722 B (Rec. 709
// linear luma).  No gamma dance — the full linear dynamic range is preserved.
// ============================================================================
static const char* PIPELINE_KERNEL_SOURCE = COLOR_KERNELS_SRC R"CL(
__kernel void grayscaleLinear(__global float4* pixels, int w, int h)
{
    int x = get_global_id(0);
    int y = get_global_id(1);
    if (x >= w || y >= h) return;

    float4 px = pixels[y * w + x];
    float L = linear_luma(px);
    pixels[y * w + x] = (float4)(L, L, L, 1.0f);
}
)CL";

// ============================================================================
// IGpuEffect — shared pipeline interface
// ============================================================================

bool GrayscaleEffect::initGpuKernels(cl::Context& ctx, cl::Device& dev) {
    try {
        cl::Program prog(ctx, PIPELINE_KERNEL_SOURCE);
        prog.build({dev});
        m_kernelLinear = cl::Kernel(prog, "grayscaleLinear");
        return true;
    }
    // GCOVR_EXCL_START
    catch (const cl::Error& e) {
        qWarning() << "[GpuPipeline] Grayscale initGpuKernels failed:"
                   << e.what() << "(err" << e.err() << ")";
        return false;
    }
    // GCOVR_EXCL_STOP
}

bool GrayscaleEffect::enqueueGpu(cl::CommandQueue& queue,
                                  cl::Buffer& buf, cl::Buffer& /*aux*/,
                                  int w, int h,
                                  const QMap<QString, QVariant>& /*params*/) {
    if (!m_active) return true;  // no-op when checkbox is unchecked

    m_kernelLinear.setArg(0, buf);
    m_kernelLinear.setArg(1, w);
    m_kernelLinear.setArg(2, h);
    queue.enqueueNDRangeKernel(m_kernelLinear, cl::NullRange,
                               cl::NDRange(w, h), cl::NullRange);
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
    check->setToolTip("Converts the image to grayscale using the perceptual luminosity formula:\n29.9% red + 58.7% green + 11.4% blue.");
    check->setChecked(m_active);
    connect(check, &QCheckBox::toggled, this, [this](bool on) {
        m_active = on;
        emit parametersChanged();
    });
    layout->addWidget(check);
    return w;
}

