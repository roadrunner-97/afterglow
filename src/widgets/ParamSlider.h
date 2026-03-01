#ifndef PARAMSLIDER_H
#define PARAMSLIDER_H

#include <QWidget>
#include <QEvent>

class QLabel;
class QSlider;
class QDoubleSpinBox;

/**
 * @brief A labeled slider+spinbox pair for a single numeric parameter.
 *
 * Encapsulates the label, QSlider, and QDoubleSpinBox trio so plugins
 * don't have to repeat the wiring boilerplate. The slider always updates
 * in real-time; the spinbox fires only on editingFinished.
 *
 * Usage:
 *   auto* p = new ParamSlider("Brightness", -100, 100);       // integer steps
 *   auto* p = new ParamSlider("Saturation", -20.0, 20.0, 0.1, 1); // 0.1 steps, 1 decimal
 *   connect(p, &ParamSlider::valueChanged, this, [this](double v) { ... });
 */
class ParamSlider : public QWidget {
    Q_OBJECT

public:
    explicit ParamSlider(const QString& label,
                         double min, double max,
                         double step = 1.0, int decimals = 0,
                         QWidget* parent = nullptr);

    double value() const;
    void setValue(double v);

signals:
    void valueChanged(double value);  // fires on every slider drag / spinbox sync
    void editingFinished();           // fires only on slider release or spinbox commit

protected:
    bool eventFilter(QObject* watched, QEvent* event) override;

private:
    QLabel*         m_label;
    QSlider*        m_slider;
    QDoubleSpinBox* m_spinBox;
    QString         m_labelPrefix;
    double          m_scaleFactor; // = 1.0 / step; slider runs at integer × scale

    void updateLabel(double v);
};

#endif // PARAMSLIDER_H
