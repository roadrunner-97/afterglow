#include "ParamSlider.h"

#include <QHBoxLayout>
#include <QVBoxLayout>
#include <QLabel>
#include <QSlider>
#include <QDoubleSpinBox>
#include <cmath>

ParamSlider::ParamSlider(const QString& label,
                         double min, double max,
                         double step, int decimals,
                         QWidget* parent)
    : QWidget(parent)
    , m_labelPrefix(label)
    , m_scaleFactor(1.0 / step)
{
    QVBoxLayout* outer = new QVBoxLayout(this);
    outer->setContentsMargins(0, 0, 0, 0);
    outer->setSpacing(4);

    m_label = new QLabel();
    m_label->setStyleSheet("color: #e0e0e0;");
    outer->addWidget(m_label);

    QHBoxLayout* row = new QHBoxLayout();
    row->setSpacing(6);

    m_slider = new QSlider(Qt::Horizontal);
    m_slider->setRange(static_cast<int>(std::round(min * m_scaleFactor)),
                       static_cast<int>(std::round(max * m_scaleFactor)));
    m_slider->setValue(0);
    row->addWidget(m_slider);

    m_spinBox = new QDoubleSpinBox();
    m_spinBox->setRange(min, max);
    m_spinBox->setSingleStep(step);
    m_spinBox->setDecimals(decimals);
    m_spinBox->setValue(0.0);
    m_spinBox->setStyleSheet("color: #e0e0e0; background-color: #444444;");
    row->addWidget(m_spinBox);

    outer->addLayout(row);

    updateLabel(0.0);

    // Slider → spinbox sync + label + emit (real-time)
    connect(m_slider, QOverload<int>::of(&QSlider::valueChanged), this, [this](int intVal) {
        double v = intVal / m_scaleFactor;
        updateLabel(v);
        m_spinBox->blockSignals(true);
        m_spinBox->setValue(v);
        m_spinBox->blockSignals(false);
        emit valueChanged(v);
    });

    // Slider released → emit editingFinished
    connect(m_slider, &QSlider::sliderReleased, this, [this]() {
        emit editingFinished();
    });

    // Spinbox → slider sync + label only (no emit — avoids double-fire)
    connect(m_spinBox, QOverload<double>::of(&QDoubleSpinBox::valueChanged), this, [this](double v) {
        updateLabel(v);
        m_slider->blockSignals(true);
        m_slider->setValue(static_cast<int>(std::round(v * m_scaleFactor)));
        m_slider->blockSignals(false);
    });

    // Spinbox → emit on editing finished
    connect(m_spinBox, &QDoubleSpinBox::editingFinished, this, [this]() {
        emit valueChanged(m_spinBox->value());
        emit editingFinished();
    });
}

double ParamSlider::value() const {
    return m_spinBox->value();
}

void ParamSlider::setValue(double v) {
    m_spinBox->blockSignals(true);
    m_slider->blockSignals(true);
    m_spinBox->setValue(v);
    m_slider->setValue(static_cast<int>(std::round(v * m_scaleFactor)));
    m_spinBox->blockSignals(false);
    m_slider->blockSignals(false);
    updateLabel(v);
}

void ParamSlider::updateLabel(double v) {
    if (m_spinBox->decimals() > 0) {
        m_label->setText(QString("%1: %2").arg(m_labelPrefix)
                                          .arg(v, 0, 'f', m_spinBox->decimals()));
    } else {
        m_label->setText(QString("%1: %2").arg(m_labelPrefix)
                                          .arg(static_cast<int>(std::round(v))));
    }
}
