#include "LogarithmParamSlider.h"

#include <QtGlobal>
#include <algorithm>
#include <cmath>

LogarithmParamSlider::LogarithmParamSlider(const QString& label,
                                           double min, double max,
                                           int decimals, int resolution,
                                           QWidget* parent)
    : ParamSlider(label,
                  Setup{
                      0,
                      resolution,
                      min, max,
                      std::pow(10.0, -decimals),
                      decimals
                  },
                  parent)
    , m_min(min)
    , m_max(max)
    , m_resolution(resolution)
    , m_logRatio(std::log(max / min))
{
    Q_ASSERT(min > 0.0);
    Q_ASSERT(max > min);
    Q_ASSERT(resolution > 0);

    // 0 isn't representable on a positive-log scale; anchor the handle at min instead of mid.
    setValue(min);
}

double LogarithmParamSlider::sliderToValue(int sliderInt) const {
    const double t = std::clamp(static_cast<double>(sliderInt) / m_resolution, 0.0, 1.0);
    return m_min * std::exp(t * m_logRatio);
}

int LogarithmParamSlider::valueToSlider(double v) const {
    if (v <= m_min) return 0;
    if (v >= m_max) return m_resolution;
    const double t = std::log(v / m_min) / m_logRatio;
    return static_cast<int>(std::round(t * m_resolution));
}
