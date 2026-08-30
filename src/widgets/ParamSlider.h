#ifndef PARAMSLIDER_H
#define PARAMSLIDER_H

#include <QWidget>
#include <QEvent>
#include <QPoint>

class QLabel;
class QSlider;
class QDoubleSpinBox;

/**
 * @brief A labeled slider+spinbox pair for a single numeric parameter.
 *
 * Mouse drag uses velocity-based gain rather than absolute mouse-position
 * tracking: a slow drag is precise (~0.25× the slider's nominal pixel
 * sensitivity), a fast flick is coarse (up to ~6×). Click-to-jump on the
 * groove is disabled — every press anchors at the current value and
 * accumulates from there. Hold Shift during a drag for forced fine-grain.
 * Double-click resets to the control's configured default value.
 *
 * Subclass and override sliderToValue()/valueToSlider() (plus pass a custom
 * Setup to the protected constructor) for non-linear mappings —
 * see LogarithmParamSlider.
 *
 * Usage:
 *   auto* p = new ParamSlider("Brightness", -100, 100);             // integer steps
 *   auto* p = new ParamSlider("Saturation", -20.0, 20.0, 0.1, 1);   // 0.1 steps
 *   connect(p, &ParamSlider::valueChanged, this, [this](double v) { ... });
 */
class ParamSlider : public QWidget {
    Q_OBJECT

public:
    explicit ParamSlider(const QString &label, double min, double max, double step = 1.0, int decimals = 0,
                         QWidget *parent = nullptr);

    double value() const;
    void   setValue(double v);
    void   setDefaultValue(double v); // sets both the current value and double-click reset point

signals:
    void valueChanged(double value); // fires on every slider drag / spinbox sync
    void editingFinished();          // fires only on slider release (with change) or spinbox commit

protected:
    // Subclass hook — describes the slider's integer range and the spinbox bounds.
    struct Setup {
        int    sliderMin;
        int    sliderMax;
        double spinMin;
        double spinMax;
        double spinStep;
        int    spinDecimals;
    };
    ParamSlider(const QString &label, const Setup &s, QWidget *parent);

    // Default mapping is linear via the public constructor's step.
    // Override these in subclasses (e.g. LogarithmParamSlider) for non-linear curves.
    virtual double sliderToValue(int sliderInt) const;
    virtual int    valueToSlider(double v) const;

    bool eventFilter(QObject *watched, QEvent *event) override;

private:
    void buildUi(const QString &label, const Setup &s);
    void wireSignals();
    void updateLabel(double v);

    bool handleDragPress(QMouseEvent *me);
    bool handleDragMove(QMouseEvent *me);
    bool handleDragRelease(QMouseEvent *me);

    QLabel         *m_label;
    QSlider        *m_slider;
    QDoubleSpinBox *m_spinBox;
    QString         m_labelPrefix;

    // Default linear mapping uses this. Subclasses with their own mapping ignore it.
    double m_scaleFactor  = 1.0;
    double m_defaultValue = 0.0;

    // Velocity-based drag state
    bool       m_dragging       = false;
    bool       m_dragMoved      = false;
    int        m_dragStartInt   = 0;
    double     m_dragSliderPosF = 0.0;
    QPoint     m_dragLastPos;
    qulonglong m_dragLastTimeMs = 0;
};

#endif // PARAMSLIDER_H
