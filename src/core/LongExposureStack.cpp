#include "LongExposureStack.h"

#include <QtGlobal>

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

} // namespace LongExposureStack
