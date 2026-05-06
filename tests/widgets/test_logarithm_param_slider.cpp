#include <QTest>
#include <QSignalSpy>
#include <QSlider>
#include <QDoubleSpinBox>
#include "LogarithmParamSlider.h"

class TestLogarithmParamSlider : public QObject {
    Q_OBJECT

private slots:
    // Defaults at min — 0.0 isn't representable on a positive-log scale, so the
    // ctor anchors the handle at min instead.
    void defaultValueIsMin() {
        LogarithmParamSlider s("Radius", 0.5, 500.0);
        QCOMPARE(s.value(), 0.5);
    }

    void setValue_atMin() {
        LogarithmParamSlider s("Radius", 0.5, 500.0);
        s.setValue(0.5);
        QCOMPARE(s.value(), 0.5);
    }

    void setValue_atMax() {
        LogarithmParamSlider s("Radius", 0.5, 500.0, 2);
        s.setValue(500.0);
        QCOMPARE(s.value(), 500.0);
    }

    // Geometric midpoint of [1, 100] is 10. With resolution=1000 the slider lands
    // at int 500 → exactly 10.0 within the spinbox's 2-decimal precision.
    void geometricMidpoint() {
        LogarithmParamSlider s("Radius", 1.0, 100.0, 2, 1000);
        s.setValue(10.0);
        auto* slider = s.findChild<QSlider*>();
        QVERIFY(slider);
        QCOMPARE(slider->value(), 500);
        // round-trip back through the spinbox's 2-decimal precision
        QVERIFY(qAbs(s.value() - 10.0) < 0.01);
    }

    // Slider int 0 → min, resolution → max, half → geometric midpoint
    void sliderToValue_endpoints() {
        LogarithmParamSlider s("Radius", 1.0, 100.0, 2, 1000);
        auto* slider = s.findChild<QSlider*>();
        QVERIFY(slider);
        slider->setValue(0);
        QVERIFY(qAbs(s.value() - 1.0) < 0.01);
        slider->setValue(1000);
        QVERIFY(qAbs(s.value() - 100.0) < 0.5);
    }

    // setValue must not emit (matches ParamSlider contract)
    void setValue_doesNotEmit() {
        LogarithmParamSlider s("Radius", 0.5, 500.0);
        QSignalSpy spy(&s, &LogarithmParamSlider::valueChanged);
        s.setValue(50.0);
        QCOMPARE(spy.count(), 0);
    }

    // Slider drag emits valueChanged with the log-mapped value
    void sliderDrag_emitsLogMappedValue() {
        LogarithmParamSlider s("Radius", 1.0, 100.0, 2, 1000);
        QSignalSpy spy(&s, &LogarithmParamSlider::valueChanged);
        auto* slider = s.findChild<QSlider*>();
        QVERIFY(slider);
        slider->setValue(500);
        QCOMPARE(spy.count(), 1);
        QVERIFY(qAbs(spy.at(0).at(0).toDouble() - 10.0) < 0.01);
    }

    // Inherits double-click reset behavior, but reset clamps to min on log scale
    // (since 0 isn't representable). Verify it lands at min.
    void doubleClick_resetsToMin() {
        LogarithmParamSlider s("Radius", 0.5, 500.0);
        s.show();
        s.setValue(50.0);
        QSignalSpy spy(&s, &LogarithmParamSlider::valueChanged);
        auto* slider = s.findChild<QSlider*>();
        QVERIFY(slider);
        QTest::mouseDClick(slider, Qt::LeftButton);
        // Reset emits valueChanged(0.0) but valueToSlider(0) clamps to slider int 0,
        // which round-trips back to min.
        QCOMPARE(s.value(), 0.5);
    }
};

QTEST_MAIN(TestLogarithmParamSlider)
#include "test_logarithm_param_slider.moc"
