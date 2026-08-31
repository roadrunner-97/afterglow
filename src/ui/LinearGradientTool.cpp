#include "LinearGradientTool.h"

#include <QKeyEvent>
#include <QLinearGradient>
#include <QMouseEvent>
#include <QPainter>
#include <QPainterPath>
#include <cmath>

namespace {
double distanceSquared(QPointF a, QPointF b) {
    const QPointF d = a - b;
    return QPointF::dotProduct(d, d);
}
} // namespace

LinearGradientTool::LinearGradientTool(QObject *parent) : QObject(parent) {}

bool LinearGradientTool::hasMask() const {
    return m_mask.has_value();
}

const LinearGradientMask *LinearGradientTool::mask() const {
    return m_mask ? &*m_mask : nullptr;
}

bool LinearGradientTool::isCreating() const {
    return m_creationMode;
}

bool LinearGradientTool::isOverlayVisible() const {
    return m_overlayVisible;
}

void LinearGradientTool::beginCreation() {
    m_creationMode = true;
    m_drag         = Drag::None;
    emit creationModeChanged(true);
}

void LinearGradientTool::setMask(const LinearGradientMask &mask) {
    m_mask = mask;
    m_beforeGesture.reset();
    m_drag         = Drag::None;
    m_creationMode = false;
    emit maskChanged();
    emit creationModeChanged(false);
}

void LinearGradientTool::clearMask() {
    if (!m_mask && !m_creationMode) return;
    m_mask.reset();
    m_beforeGesture.reset();
    m_drag         = Drag::None;
    m_creationMode = false;
    emit maskChanged();
    emit creationModeChanged(false);
}

void LinearGradientTool::setInverted(bool inverted) {
    if (!m_mask || m_mask->isInverted() == inverted) return;
    m_mask->setInverted(inverted);
    emit maskChanged();
    emit gestureFinished();
}

void LinearGradientTool::setOverlayVisible(bool visible) {
    if (m_overlayVisible == visible) return;
    m_overlayVisible = visible;
    emit maskChanged();
}

QPointF LinearGradientTool::screenToNormalized(QPointF screen, const ViewportTransform &vt) const {
    const QPointF source = vt.screenToRotatedSource(screen);
    if (vt.imageSize.isEmpty()) return {};
    return {source.x() / vt.imageSize.width(), source.y() / vt.imageSize.height()};
}

QPointF LinearGradientTool::normalizedToScreen(QPointF normalized, const ViewportTransform &vt) const {
    return vt.rotatedSourceToScreen({normalized.x() * vt.imageSize.width(), normalized.y() * vt.imageSize.height()});
}

LinearGradientTool::Handles LinearGradientTool::handles(const ViewportTransform &vt) const {
    if (!m_mask) return {};
    const QPointF center = normalizedToScreen(m_mask->center(), vt);
    const QPointF basisX = normalizedToScreen(m_mask->center() + QPointF(1.0, 0.0), vt) - center;
    const QPointF basisY = normalizedToScreen(m_mask->center() + QPointF(0.0, 1.0), vt) - center;

    // The mask direction is a normal (a covector) in normalized image
    // coordinates. Transform it with the inverse transpose so the on-screen
    // guides remain parallel to the actual constant-weight lines on images
    // whose width and height differ.
    const double determinant = basisX.x() * basisY.y() - basisY.x() * basisX.y();
    if (std::abs(determinant) < 1e-9) return {center, center, center};
    const QPointF direction = m_mask->direction();
    const QPointF screenNormal((basisY.y() * direction.x() - basisX.y() * direction.y()) / determinant,
                               (-basisY.x() * direction.x() + basisX.x() * direction.y()) / determinant);
    const double  normalSquared = QPointF::dotProduct(screenNormal, screenNormal);
    const QPointF offset =
        normalSquared > 0.0 ? screenNormal * (m_mask->featherHalfWidth() / normalSquared) : QPointF();
    return {center - offset, center, center + offset};
}

LinearGradientTool::Drag LinearGradientTool::hitTest(QPointF screen, const ViewportTransform &vt) const {
    if (!m_mask) return Drag::None;
    const Handles h  = handles(vt);
    const double  r2 = HIT_RADIUS * HIT_RADIUS;
    if (distanceSquared(screen, h.center) <= r2) return Drag::Move;
    if (distanceSquared(screen, h.start) <= r2) return Drag::Start;
    if (distanceSquared(screen, h.end) <= r2) return Drag::End;
    return Drag::None;
}

void LinearGradientTool::setFromScreenEndpoints(QPointF start, QPointF end, const ViewportTransform &vt) {
    const QPointF screenAxis      = end - start;
    const double  screenLength    = std::hypot(screenAxis.x(), screenAxis.y());
    const QPointF normalizedStart = screenToNormalized(start, vt);
    const QPointF normalizedEnd   = screenToNormalized(end, vt);
    const QPointF center          = (normalizedStart + normalizedEnd) * 0.5;
    const QPointF centerScreen    = normalizedToScreen(center, vt);
    const QPointF basisX          = normalizedToScreen(center + QPointF(1.0, 0.0), vt) - centerScreen;
    const QPointF basisY          = normalizedToScreen(center + QPointF(0.0, 1.0), vt) - centerScreen;
    QPointF       direction(basisX.x() * screenAxis.x() + basisX.y() * screenAxis.y(),
                            basisY.x() * screenAxis.x() + basisY.y() * screenAxis.y());
    const double  directionLength = std::hypot(direction.x(), direction.y());
    direction                     = directionLength > 0.0 ? direction / directionLength : QPointF(0.0, 1.0);
    const double halfWidth        = std::abs(QPointF::dotProduct(normalizedEnd - center, direction));
    const bool   inverted         = m_mask && m_mask->isInverted();
    m_mask.emplace(center, direction, screenLength > 0.0 ? halfWidth : 0.0, inverted);
}

void LinearGradientTool::cancelGesture() {
    if (m_drag == Drag::None && !m_creationMode) return;
    m_mask         = m_beforeGesture;
    m_drag         = Drag::None;
    m_creationMode = false;
    emit maskChanged();
    emit creationModeChanged(false);
}

void LinearGradientTool::paintOverlay(QPainter &painter, const ViewportTransform &vt) {
    if (!m_mask || vt.imageSize.isEmpty()) return;
    const Handles h    = handles(vt);
    QPointF       axis = h.end - h.start;
    const double  len  = std::hypot(axis.x(), axis.y());
    if (len == 0.0) axis = {0.0, 1.0};
    else axis /= len;
    const QPointF perpendicular(-axis.y(), axis.x());
    const double  extent = std::hypot(vt.viewportSize.width(), vt.viewportSize.height());

    painter.save();
    painter.setRenderHint(QPainter::Antialiasing);

    const QColor shadow(0, 0, 0, 190);
    const QColor line(240, 240, 240, 225);
    const QColor accent(84, 190, 210, 255);
    const QColor transparent(84, 190, 210, 0);
    const QColor affected(84, 190, 210, 48);

    if (m_overlayVisible) {
        QLinearGradient feather(h.start, h.end);
        feather.setSpread(QGradient::PadSpread);
        if (m_mask->isInverted()) {
            feather.setColorAt(0.0, affected);
            feather.setColorAt(1.0, transparent);
        } else {
            feather.setColorAt(0.0, transparent);
            feather.setColorAt(1.0, affected);
        }
        painter.fillRect(QRectF(QPointF(0.0, 0.0), QSizeF(vt.viewportSize)), feather);
    }

    auto drawBoundary = [&](QPointF point, Qt::PenStyle style, const QColor &color) {
        painter.setPen(QPen(shadow, 3.0, style));
        painter.drawLine(point - perpendicular * extent, point + perpendicular * extent);
        painter.setPen(QPen(color, 1.0, style));
        painter.drawLine(point - perpendicular * extent, point + perpendicular * extent);
    };
    drawBoundary(h.start, Qt::DashLine, line);
    drawBoundary(h.center, Qt::SolidLine, accent);
    drawBoundary(h.end, Qt::DashLine, accent);

    painter.setPen(QPen(shadow, 3.0));
    painter.drawLine(h.start, h.end);
    painter.setPen(QPen(accent, 1.5));
    painter.drawLine(h.start, h.end);

    auto drawHandle = [&](QPointF point, bool filled) {
        painter.setPen(QPen(shadow, 3.0));
        painter.setBrush(filled ? accent : QColor(30, 30, 30, 220));
        painter.drawEllipse(point, 6.0, 6.0);
        painter.setPen(QPen(line, 1.0));
        painter.drawEllipse(point, 6.0, 6.0);
    };
    drawHandle(h.start, false);
    drawHandle(h.center, true);
    drawHandle(h.end, true);

    const QPointF arrowTip  = h.end + axis * 18.0;
    const QPointF arrowBase = h.end + axis * 8.0;
    QPolygonF     arrow;
    arrow << arrowTip << arrowBase + perpendicular * 5.0 << arrowBase - perpendicular * 5.0;
    painter.setPen(Qt::NoPen);
    painter.setBrush(accent);
    painter.drawPolygon(arrow);
    painter.restore();
}

bool LinearGradientTool::mousePress(QMouseEvent *event, const ViewportTransform &vt) {
    if (event->button() != Qt::LeftButton || vt.imageSize.isEmpty()) return false;
    const QPointF normalized = screenToNormalized(event->position(), vt);
    if (m_creationMode) {
        m_beforeGesture = m_mask;
        m_anchor        = event->position();
        m_drag          = Drag::Create;
        setFromScreenEndpoints(m_anchor, m_anchor, vt);
        emit maskChanged();
        return true;
    }

    m_drag = hitTest(event->position(), vt);
    if (m_drag == Drag::None) return false;
    m_beforeGesture = m_mask;
    m_dragStart     = normalized;
    if (m_mask) {
        const Handles h = handles(vt);
        m_anchor        = m_drag == Drag::Start ? h.end : h.start;
    }
    return true;
}

bool LinearGradientTool::mouseMove(QMouseEvent *event, const ViewportTransform &vt) {
    if (m_drag == Drag::None || !m_mask) return false;
    const QPointF normalized = screenToNormalized(event->position(), vt);
    if (m_drag == Drag::Create) setFromScreenEndpoints(m_anchor, event->position(), vt);
    else if (m_drag == Drag::Start) setFromScreenEndpoints(event->position(), m_anchor, vt);
    else if (m_drag == Drag::End) setFromScreenEndpoints(m_anchor, event->position(), vt);
    else if (m_beforeGesture) {
        const QPointF delta = normalized - m_dragStart;
        m_mask->setCenter(m_beforeGesture->center() + delta);
    }
    emit maskChanged();
    return true;
}

bool LinearGradientTool::mouseRelease(QMouseEvent *event, const ViewportTransform &) {
    if (event->button() != Qt::LeftButton || m_drag == Drag::None) return false;
    m_drag         = Drag::None;
    m_creationMode = false;
    m_beforeGesture.reset();
    emit creationModeChanged(false);
    emit gestureFinished();
    return true;
}

bool LinearGradientTool::keyPress(QKeyEvent *event) {
    if (event->key() == Qt::Key_Escape && (m_drag != Drag::None || m_creationMode)) {
        cancelGesture();
        return true;
    }
    if ((event->key() == Qt::Key_Delete || event->key() == Qt::Key_Backspace) && m_mask) {
        clearMask();
        emit gestureFinished();
        return true;
    }
    return false;
}

QCursor LinearGradientTool::cursorFor(QPointF screenPx, const ViewportTransform &vt) {
    if (m_creationMode) return QCursor(Qt::CrossCursor);
    switch (hitTest(screenPx, vt)) {
    case Drag::Move:
        return QCursor(Qt::SizeAllCursor);
    case Drag::Start:
    case Drag::End:
        return QCursor(Qt::OpenHandCursor);
    default:
        return {};
    }
}
