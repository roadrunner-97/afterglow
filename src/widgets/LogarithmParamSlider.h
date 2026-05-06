#ifndef LOGARITHMPARAMSLIDER_H
#define LOGARITHMPARAMSLIDER_H

#include "ParamSlider.h"

/**
 * @brief Parameter slider whose handle position maps logarithmically to value.
 *
 * Use for parameters that span more than ~one order of magnitude where the user
 * cares about relative (multiplicative) change rather than absolute steps —
 * e.g. blur radius, exposure, frequency cutoffs. Both @p min and @p max must be
 * positive; for symmetric/signed-log behavior use ParamSlider instead.
 *
 * The underlying QSlider runs at integer positions @c [0, resolution]; the
 * mapping is @c value = min · (max/min)^(t/resolution). The spinbox accepts
 * direct numeric entry across the full @c [min, max] range.
 *
 * Inherits the velocity-based drag, double-click reset, and Shift-fine-grain
 * behavior from ParamSlider.
 */
class LogarithmParamSlider : public ParamSlider {
    Q_OBJECT

public:
    explicit LogarithmParamSlider(const QString& label,
                                  double min, double max,
                                  int decimals = 2,
                                  int resolution = 1000,
                                  QWidget* parent = nullptr);

protected:
    double sliderToValue(int sliderInt) const override;
    int    valueToSlider(double v) const override;

private:
    double m_min;
    double m_max;
    int    m_resolution;
    double m_logRatio;  // log(max / min)
};

#endif // LOGARITHMPARAMSLIDER_H
