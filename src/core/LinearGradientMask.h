#ifndef LINEARGRADIENTMASK_H
#define LINEARGRADIENTMASK_H

#include <QPointF>

// Analytical linear-gradient mask in normalized, uncropped source-image
// coordinates. direction points from the unaffected side toward the affected
// side. featherHalfWidth is the distance from the center to either edge of the
// transition band.
class LinearGradientMask {
public:
    LinearGradientMask(QPointF center = {0.5, 0.5}, QPointF direction = {0.0, 1.0}, double featherHalfWidth = 0.25,
                       bool inverted = false);

    QPointF center() const;
    QPointF direction() const;
    double  featherHalfWidth() const;
    bool    isInverted() const;

    void setCenter(QPointF center);
    void setDirection(QPointF direction);
    void setFeatherHalfWidth(double halfWidth);
    void setInverted(bool inverted);

    // Returns [0, 1]. A zero-width feather is a hard transition whose center
    // belongs to the affected side.
    double weightAt(QPointF sourcePoint) const;

private:
    QPointF m_center;
    QPointF m_direction;
    double  m_featherHalfWidth;
    bool    m_inverted;
};

#endif // LINEARGRADIENTMASK_H
