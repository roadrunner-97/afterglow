#ifndef IINTERACTIVEEFFECT_H
#define IINTERACTIVEEFFECT_H

#include <QCursor>
#include <QPointF>
#include <QSize>
#include <algorithm>
#include <cmath>

class QMouseEvent;
class QPainter;
class QKeyEvent;

// Snapshot of how the source image is currently displayed in the viewport.
// Mirrors ViewportWidget's convention: center is normalised (0..1) on both
// axes, zoom is a multiplier over the fit-to-widget scale.
struct ViewportTransform {
    QSize   imageSize;
    QSize   viewportSize;
    QPointF center{0.5, 0.5};
    float   zoom          = 1.0f;
    float   imageAngleDeg = 0.0f;
    QPointF imagePivotNorm{0.5, 0.5};

    // Pixels of screen per pixel of source.
    float displayScale() const {
        if (imageSize.isEmpty() || viewportSize.isEmpty()) return 1.0f;
        const float fit = std::min(static_cast<float>(viewportSize.width()) / static_cast<float>(imageSize.width()),
                                   static_cast<float>(viewportSize.height()) / static_cast<float>(imageSize.height()));
        return fit * zoom;
    }

    QPointF sourceToScreen(QPointF srcPx) const {
        if (imageSize.isEmpty() || viewportSize.isEmpty()) return {};
        const float ds      = displayScale();
        const float regionW = static_cast<float>(viewportSize.width()) / ds;
        const float regionH = static_cast<float>(viewportSize.height()) / ds;
        const float x0      = static_cast<float>(center.x()) * static_cast<float>(imageSize.width()) - regionW * 0.5f;
        const float y0      = static_cast<float>(center.y()) * static_cast<float>(imageSize.height()) - regionH * 0.5f;
        return {(srcPx.x() - x0) * ds, (srcPx.y() - y0) * ds};
    }

    QPointF screenToSource(QPointF screenPx) const {
        if (imageSize.isEmpty() || viewportSize.isEmpty()) return {};
        const float ds      = displayScale();
        const float regionW = static_cast<float>(viewportSize.width()) / ds;
        const float regionH = static_cast<float>(viewportSize.height()) / ds;
        const float x0      = static_cast<float>(center.x()) * static_cast<float>(imageSize.width()) - regionW * 0.5f;
        const float y0      = static_cast<float>(center.y()) * static_cast<float>(imageSize.height()) - regionH * 0.5f;
        return {x0 + screenPx.x() / ds, y0 + screenPx.y() / ds};
    }

    // Mapping for controls anchored to image content. The ordinary methods
    // above intentionally remain unrotated because crop's frame stays
    // axis-aligned while the image rotates beneath it.
    QPointF rotatedSourceToScreen(QPointF srcPx) const {
        const QPointF point = sourceToScreen(srcPx);
        const QPointF pivot =
            sourceToScreen({imagePivotNorm.x() * imageSize.width(), imagePivotNorm.y() * imageSize.height()});
        const double  radians = static_cast<double>(imageAngleDeg) * 3.14159265358979323846 / 180.0;
        const double  c = std::cos(radians), s = std::sin(radians);
        const QPointF delta = point - pivot;
        return pivot + QPointF(c * delta.x() + s * delta.y(), -s * delta.x() + c * delta.y());
    }

    QPointF screenToRotatedSource(QPointF screenPx) const {
        const QPointF pivot =
            sourceToScreen({imagePivotNorm.x() * imageSize.width(), imagePivotNorm.y() * imageSize.height()});
        const double  radians = static_cast<double>(imageAngleDeg) * 3.14159265358979323846 / 180.0;
        const double  c = std::cos(radians), s = std::sin(radians);
        const QPointF delta     = screenPx - pivot;
        const QPointF unrotated = pivot + QPointF(c * delta.x() - s * delta.y(), s * delta.x() + c * delta.y());
        return screenToSource(unrotated);
    }
};

// Mixin interface for effects that draw on-image overlays and consume
// mouse events (crop handles, rotation grip, line-straighten, future
// gradient / radial-filter handles, etc.).
//
// Mouse handlers return true when they claimed the event — the viewport
// falls through to its own pan behaviour on false.
class IInteractiveEffect {
public:
    virtual ~IInteractiveEffect() = default; // GCOVR_EXCL_LINE

    virtual void paintOverlay(QPainter &painter, const ViewportTransform &vt)  = 0;
    virtual bool mousePress(QMouseEvent *event, const ViewportTransform &vt)   = 0;
    virtual bool mouseMove(QMouseEvent *event, const ViewportTransform &vt)    = 0;
    virtual bool mouseRelease(QMouseEvent *event, const ViewportTransform &vt) = 0;
    // GCOVR_EXCL_START
    virtual bool keyPress(QKeyEvent * /*event*/) {
        return false;
    }
    virtual QCursor cursorFor(QPointF /*screenPx*/, const ViewportTransform & /*vt*/) {
        return {};
    }
    // GCOVR_EXCL_STOP
};

#endif // IINTERACTIVEEFFECT_H
