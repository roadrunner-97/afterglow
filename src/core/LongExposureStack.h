#ifndef LONGEXPOSURESTACK_H
#define LONGEXPOSURESTACK_H

#define CL_HPP_TARGET_OPENCL_VERSION 200
#define CL_HPP_MINIMUM_OPENCL_VERSION 110
#define CL_HPP_ENABLE_EXCEPTIONS
#include <CL/opencl.hpp>

#include <QMap>
#include <QImage>
#include <QString>
#include <QVariant>
#include <cstddef>
#include <memory>

// A named decision instead of a bare bool leaves room for future per-frame
// policies (for example masked contribution or foreground-only frames)
// without changing the stack project/UI contract.
enum class StackFrameDecision { Include = 0, Exclude = 1 };

struct StackFrame {
    QString            path;
    StackFrameDecision decision = StackFrameDecision::Include;
};

struct StackAggregationConfig {
    QString                 methodId = QStringLiteral("per-channel-maximum");
    QMap<QString, QVariant> parameters;
};

class IStackAggregationStrategy {
public:
    virtual ~IStackAggregationStrategy() = default; // GCOVR_EXCL_LINE — compiler-emitted deleting destructor

    virtual QString id() const                            = 0;
    virtual QString displayName() const                   = 0;
    virtual QString cacheVersion() const                  = 0;
    virtual bool    supportsAssociativeBlockCache() const = 0;
    virtual int     preferredBlockSize() const {
        return 16;
    }
    virtual size_t accumulatorBytes(int width, int height) const = 0;

    virtual bool initialize(cl::Context &context, cl::Device &device, QString *error = nullptr) = 0;
    virtual bool enqueue(cl::CommandQueue &queue, cl::Buffer &accumulator, const cl::Buffer &frame, int width,
                         int height, bool seeded, QString *error = nullptr)                     = 0;
    virtual bool resolve(cl::CommandQueue &queue, const cl::Buffer &accumulator, cl::Buffer &linearResult, int width,
                         int height, QString *error = nullptr)                                  = 0;
};

namespace LongExposureStack {

// Merge `frame` into `accumulator` with the conventional star-trail
// "lighten" operation: each output colour channel keeps the larger value.
// The first frame seeds an empty accumulator. Returns false for null or
// differently-sized input and leaves an existing accumulator untouched.
bool blendLighten(QImage *accumulator, const QImage &frame, QString *error = nullptr);

QStringList                                aggregationMethodIds();
QString                                    aggregationMethodDisplayName(const QString &methodId);
std::unique_ptr<IStackAggregationStrategy> createAggregationStrategy(const StackAggregationConfig &config,
                                                                     QString                      *error = nullptr);

} // namespace LongExposureStack

#endif // LONGEXPOSURESTACK_H
