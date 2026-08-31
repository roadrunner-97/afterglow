#include "LinearGradientMask.h"

#include <algorithm>
#include <cmath>

namespace {
QPointF normalizedDirection(QPointF direction) {
    const double length = std::hypot(direction.x(), direction.y());
    if (length == 0.0) return {0.0, 1.0};
    return direction / length;
}
} // namespace

LinearGradientMask::LinearGradientMask(QPointF center, QPointF direction, double featherHalfWidth, bool inverted)
    : m_center(center), m_direction(normalizedDirection(direction)), m_featherHalfWidth(std::abs(featherHalfWidth)),
      m_inverted(inverted) {}

QPointF LinearGradientMask::center() const {
    return m_center;
}

QPointF LinearGradientMask::direction() const {
    return m_direction;
}

double LinearGradientMask::featherHalfWidth() const {
    return m_featherHalfWidth;
}

bool LinearGradientMask::isInverted() const {
    return m_inverted;
}

void LinearGradientMask::setCenter(QPointF center) {
    m_center = center;
}

void LinearGradientMask::setDirection(QPointF direction) {
    m_direction = normalizedDirection(direction);
}

void LinearGradientMask::setFeatherHalfWidth(double halfWidth) {
    m_featherHalfWidth = std::abs(halfWidth);
}

void LinearGradientMask::setInverted(bool inverted) {
    m_inverted = inverted;
}

double LinearGradientMask::weightAt(QPointF sourcePoint) const {
    const QPointF delta      = sourcePoint - m_center;
    const double  projection = QPointF::dotProduct(delta, m_direction);
    double        weight;
    if (m_featherHalfWidth == 0.0) weight = projection >= 0.0 ? 1.0 : 0.0;
    else weight = std::clamp(0.5 + projection / (2.0 * m_featherHalfWidth), 0.0, 1.0);
    return m_inverted ? 1.0 - weight : weight;
}
