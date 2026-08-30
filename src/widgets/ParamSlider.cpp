#include "ParamSlider.h"

#include <QHBoxLayout>
#include <QVBoxLayout>
#include <QLabel>
#include <QSlider>
#include <QDoubleSpinBox>
#include <QMouseEvent>
#include <QSignalBlocker>
#include <algorithm>
#include <cmath>

ParamSlider::ParamSlider(const QString &label, double min, double max, double step, int decimals, QWidget *parent)
    : ParamSlider(label,
                  Setup{static_cast<int>(std::round(min / step)), static_cast<int>(std::round(max / step)), min, max,
                        step, decimals},
                  parent) {
    m_scaleFactor = 1.0 / step;
}

ParamSlider::ParamSlider(const QString &label, const Setup &s, QWidget *parent)
    : QWidget(parent), m_labelPrefix(label) {
    buildUi(label, s);
    wireSignals();
    m_defaultValue = value();
}

void ParamSlider::buildUi(const QString &label, const Setup &s) {
    QVBoxLayout *outer = new QVBoxLayout(this);
    outer->setContentsMargins(0, 0, 0, 0);
    outer->setSpacing(4);

    m_label = new QLabel();
    outer->addWidget(m_label);

    QHBoxLayout *row = new QHBoxLayout();
    row->setSpacing(6);

    m_slider = new QSlider(Qt::Horizontal);
    m_slider->setRange(s.sliderMin, s.sliderMax);
    m_slider->setValue(0);
    m_slider->installEventFilter(this);
    row->addWidget(m_slider);

    m_spinBox = new QDoubleSpinBox();
    m_spinBox->setRange(s.spinMin, s.spinMax);
    m_spinBox->setSingleStep(s.spinStep);
    m_spinBox->setDecimals(s.spinDecimals);
    m_spinBox->setValue(0.0);
    row->addWidget(m_spinBox);

    outer->addLayout(row);

    updateLabel(0.0);
    (void)label;
}

void ParamSlider::wireSignals() {
    // Slider int → user value: sync spinbox, update label, emit (real-time)
    connect(m_slider, QOverload<int>::of(&QSlider::valueChanged), this, [this](int intVal) {
        double v = sliderToValue(intVal);
        updateLabel(v);
        QSignalBlocker bk(m_spinBox);
        m_spinBox->setValue(v);
        emit valueChanged(v);
    });

    // Slider released → emit editingFinished. With our custom drag handler this no
    // longer fires from real input (we consume press/release), but keep the wiring
    // for callers / tests that invoke sliderReleased programmatically.
    connect(m_slider, &QSlider::sliderReleased, this, [this]() { emit editingFinished(); });

    // Spinbox → slider sync (no emit — avoids double-fire)
    connect(m_spinBox, QOverload<double>::of(&QDoubleSpinBox::valueChanged), this, [this](double v) {
        updateLabel(v);
        QSignalBlocker bk(m_slider);
        m_slider->setValue(valueToSlider(v));
    });

    // Spinbox commit (Enter or focus loss)
    connect(m_spinBox, &QDoubleSpinBox::editingFinished, this, [this]() {
        emit valueChanged(m_spinBox->value());
        emit editingFinished();
    });
}

double ParamSlider::value() const {
    return m_spinBox->value();
}

void ParamSlider::setValue(double v) {
    QSignalBlocker bk1(m_spinBox);
    QSignalBlocker bk2(m_slider);
    m_spinBox->setValue(v);
    m_slider->setValue(valueToSlider(v));
    updateLabel(v);
}

void ParamSlider::setDefaultValue(double v) {
    setValue(v);
    m_defaultValue = value();
}

double ParamSlider::sliderToValue(int sliderInt) const {
    return sliderInt / m_scaleFactor;
}

int ParamSlider::valueToSlider(double v) const {
    return static_cast<int>(std::round(v * m_scaleFactor));
}

void ParamSlider::updateLabel(double v) {
    if (m_spinBox->decimals() > 0) {
        m_label->setText(QString("%1: %2").arg(m_labelPrefix).arg(v, 0, 'f', m_spinBox->decimals()));
    } else {
        m_label->setText(QString("%1: %2").arg(m_labelPrefix).arg(static_cast<int>(std::round(v))));
    }
}

bool ParamSlider::eventFilter(QObject *watched, QEvent *event) {
    if (watched == m_slider) {
        switch (event->type()) {
        case QEvent::MouseButtonDblClick: {
            auto *me = static_cast<QMouseEvent *>(event);
            if (me->button() == Qt::LeftButton) {
                m_dragging = false;
                setValue(m_defaultValue);
                emit valueChanged(m_defaultValue);
                emit editingFinished();
                return true;
            }
            break;
        }
        case QEvent::MouseButtonPress:
            return handleDragPress(static_cast<QMouseEvent *>(event));
        case QEvent::MouseMove:
            return handleDragMove(static_cast<QMouseEvent *>(event));
        case QEvent::MouseButtonRelease:
            return handleDragRelease(static_cast<QMouseEvent *>(event));
        default:
            break;
        }
    }
    return QWidget::eventFilter(watched, event);
}

bool ParamSlider::handleDragPress(QMouseEvent *me) {
    if (me->button() != Qt::LeftButton) return false;
    m_dragging       = true;
    m_dragMoved      = false;
    m_dragStartInt   = m_slider->value();
    m_dragSliderPosF = static_cast<double>(m_dragStartInt);
    m_dragLastPos    = me->pos();
    m_dragLastTimeMs = me->timestamp();
    return true; // consume — don't let QSlider do click-to-jump
}

bool ParamSlider::handleDragMove(QMouseEvent *me) {
    if (!m_dragging) return false;

    const QPoint pos   = me->pos();
    const int    dx_px = pos.x() - m_dragLastPos.x();
    if (dx_px == 0) {
        m_dragLastTimeMs = me->timestamp();
        return true;
    }

    // Time delta — clamp to >=4 ms so a flurry of same-tick events doesn't blow up the gain.
    const qulonglong now   = me->timestamp();
    const double     dt_ms = std::max<double>(4.0, static_cast<double>(now - m_dragLastTimeMs));

    // Velocity (px/sec) → gain. Slow ≈ 0.25×, typical ≈ 1.25×, fast flick ≈ 6×.
    const double speed_pxps = std::abs(dx_px) * 1000.0 / dt_ms;
    double       gain       = std::clamp(0.25 + speed_pxps / 600.0, 0.25, 6.0);
    if (me->modifiers() & Qt::ShiftModifier) {
        gain = std::min(gain, 0.4); // forced fine-grain
    }

    // Convert px → slider int units. At gain=1, dragging the full slider width sweeps the full range.
    const int    sliderRange    = m_slider->maximum() - m_slider->minimum();
    const int    sliderUsablePx = std::max(1, m_slider->width());
    const double intsPerPx      = static_cast<double>(sliderRange) / static_cast<double>(sliderUsablePx);

    m_dragSliderPosF += dx_px * intsPerPx * gain;
    m_dragSliderPosF  = std::clamp(m_dragSliderPosF, static_cast<double>(m_slider->minimum()),
                                   static_cast<double>(m_slider->maximum()));

    const int newInt = static_cast<int>(std::round(m_dragSliderPosF));
    if (newInt != m_slider->value()) {
        m_dragMoved = true;
        m_slider->setValue(newInt); // fires valueChanged → user-facing emit
    }

    m_dragLastPos    = pos;
    m_dragLastTimeMs = now;
    return true;
}

bool ParamSlider::handleDragRelease(QMouseEvent *me) {
    if (me->button() != Qt::LeftButton || !m_dragging) return false;
    const bool moved = m_dragMoved;
    m_dragging       = false;
    m_dragMoved      = false;
    if (moved) {
        emit editingFinished();
    }
    return true;
}
