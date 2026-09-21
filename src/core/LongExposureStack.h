#ifndef LONGEXPOSURESTACK_H
#define LONGEXPOSURESTACK_H

#include <QImage>
#include <QString>

// A named decision instead of a bare bool leaves room for future per-frame
// policies (for example masked contribution or foreground-only frames)
// without changing the stack project/UI contract.
enum class StackFrameDecision { Include = 0, Exclude = 1 };

struct StackFrame {
    QString            path;
    StackFrameDecision decision = StackFrameDecision::Include;
};

namespace LongExposureStack {

// Merge `frame` into `accumulator` with the conventional star-trail
// "lighten" operation: each output colour channel keeps the larger value.
// The first frame seeds an empty accumulator. Returns false for null or
// differently-sized input and leaves an existing accumulator untouched.
bool blendLighten(QImage *accumulator, const QImage &frame, QString *error = nullptr);

} // namespace LongExposureStack

#endif // LONGEXPOSURESTACK_H
