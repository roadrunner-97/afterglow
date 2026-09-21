#include "LongExposureStack.h"

#include <QtGlobal>
#include <algorithm>

namespace {

static const char *MAXIMUM_KERNEL = R"CL(
__kernel void aggregate_maximum(__global float4 *accumulator,
                                __global const float4 *frame,
                                int pixelCount)
{
    int i = get_global_id(0);
    if (i >= pixelCount) return;
    float4 a = accumulator[i];
    float4 b = frame[i];
    accumulator[i] = (float4)(fmax(a.x, b.x), fmax(a.y, b.y), fmax(a.z, b.z), 1.0f);
}
)CL";

class PerChannelMaximumStrategy final : public IStackAggregationStrategy {
public:
    QString id() const override {
        return QStringLiteral("per-channel-maximum");
    }
    QString displayName() const override {
        return QStringLiteral("Per-channel maximum");
    }
    QString cacheVersion() const override {
        return QStringLiteral("maximum-float32-v1");
    }
    bool supportsAssociativeBlockCache() const override {
        return true;
    }
    size_t accumulatorBytes(int width, int height) const override {
        return static_cast<size_t>(width) * static_cast<size_t>(height) * sizeof(cl_float4);
    }

    bool initialize(cl::Context &context, cl::Device &device, QString *error) override {
        if (m_contextHandle == context()) return true;
        try {
            cl::Program program(context, MAXIMUM_KERNEL);
            program.build({device});
            m_kernel = cl::Kernel(program, "aggregate_maximum");
            m_contextHandle = context();
            return true;
        }
        // OpenCL compiler/runtime failures cannot be induced portably in a
        // unit test once a valid selected device and fixed kernel are used.
        // GCOVR_EXCL_START
        catch (const cl::Error &e) {
            if (error)
                *error = QStringLiteral("Could not initialize the maximum stack method: %1 (%2)")
                             .arg(QString::fromLatin1(e.what()))
                             .arg(e.err());
            return false;
        }
        // GCOVR_EXCL_STOP
    }

    bool enqueue(cl::CommandQueue &queue, cl::Buffer &accumulator, const cl::Buffer &frame, int width, int height,
                 bool seeded, QString *error) override {
        try {
            const size_t bytes = static_cast<size_t>(width) * static_cast<size_t>(height) * sizeof(cl_float4);
            if (!seeded) {
                queue.enqueueCopyBuffer(frame, accumulator, 0, 0, bytes);
                return true;
            }
            const int pixelCount = width * height;
            m_kernel.setArg(0, accumulator);
            m_kernel.setArg(1, frame);
            m_kernel.setArg(2, pixelCount);
            queue.enqueueNDRangeKernel(m_kernel, cl::NullRange, cl::NDRange(static_cast<size_t>(pixelCount)));
            return true;
        }
        // GCOVR_EXCL_START
        catch (const cl::Error &e) {
            if (error)
                *error = QStringLiteral("Could not add a frame to the maximum stack: %1 (%2)")
                             .arg(QString::fromLatin1(e.what()))
                             .arg(e.err());
            return false;
        }
        // GCOVR_EXCL_STOP
    }

    bool resolve(cl::CommandQueue &queue, const cl::Buffer &accumulator, cl::Buffer &linearResult, int width,
                 int height, QString *error) override {
        try {
            queue.enqueueCopyBuffer(accumulator, linearResult, 0, 0, accumulatorBytes(width, height));
            return true;
        }
        // GCOVR_EXCL_START
        catch (const cl::Error &e) {
            if (error)
                *error = QStringLiteral("Could not resolve the maximum stack: %1 (%2)")
                             .arg(QString::fromLatin1(e.what()))
                             .arg(e.err());
            return false;
        }
        // GCOVR_EXCL_STOP
    }

private:
    cl::Kernel  m_kernel;
    cl_context m_contextHandle = nullptr;
};

} // namespace

namespace LongExposureStack {

bool blendLighten(QImage *accumulator, const QImage &frame, QString *error) {
    if (!accumulator) {
        if (error) *error = QStringLiteral("No stack accumulator was provided.");
        return false;
    }
    if (frame.isNull()) {
        if (error) *error = QStringLiteral("The frame is empty.");
        return false;
    }
    if (accumulator->isNull()) {
        *accumulator = frame.convertToFormat(QImage::Format_RGB32);
        return true;
    }
    if (accumulator->size() != frame.size()) {
        if (error)
            *error = QStringLiteral("Frame dimensions %1 × %2 do not match the stack's %3 × %4.")
                         .arg(frame.width())
                         .arg(frame.height())
                         .arg(accumulator->width())
                         .arg(accumulator->height());
        return false;
    }

    const QImage source = frame.convertToFormat(QImage::Format_RGB32);
    if (accumulator->format() != QImage::Format_RGB32)
        *accumulator = accumulator->convertToFormat(QImage::Format_RGB32);

    for (int y = 0; y < source.height(); ++y) {
        const auto *src = reinterpret_cast<const QRgb *>(source.constScanLine(y));
        auto       *dst = reinterpret_cast<QRgb *>(accumulator->scanLine(y));
        for (int x = 0; x < source.width(); ++x) {
            dst[x] = qRgb(qMax(qRed(dst[x]), qRed(src[x])), qMax(qGreen(dst[x]), qGreen(src[x])),
                          qMax(qBlue(dst[x]), qBlue(src[x])));
        }
    }
    return true;
}

QStringList aggregationMethodIds() {
    return {QStringLiteral("per-channel-maximum")};
}

QString aggregationMethodDisplayName(const QString &methodId) {
    if (methodId == QStringLiteral("per-channel-maximum")) return QStringLiteral("Per-channel maximum");
    return methodId;
}

std::unique_ptr<IStackAggregationStrategy> createAggregationStrategy(const StackAggregationConfig &config,
                                                                      QString *error) {
    if (config.methodId == QStringLiteral("per-channel-maximum"))
        return std::make_unique<PerChannelMaximumStrategy>();
    if (error) *error = QStringLiteral("Unknown stack aggregation method: %1").arg(config.methodId);
    return {};
}

} // namespace LongExposureStack
