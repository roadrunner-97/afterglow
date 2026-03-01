#include <QTest>
#include <QSignalSpy>
#include "ParamSlider.h"

class TestParamSlider : public QObject {
    Q_OBJECT

private slots:
    void defaultValueIsZero() {
        ParamSlider s("Brightness", -100.0, 100.0);
        QCOMPARE(s.value(), 0.0);
    }

    void defaultValueIsZero_fractionalStep() {
        ParamSlider s("Saturation", -20.0, 20.0, 0.1, 1);
        QCOMPARE(s.value(), 0.0);
    }

    void setValue_roundTrip_integerStep() {
        ParamSlider s("Test", -100.0, 100.0);
        s.setValue(42.0);
        QCOMPARE(s.value(), 42.0);
    }

    void setValue_roundTrip_negativeValue() {
        ParamSlider s("Test", -100.0, 100.0);
        s.setValue(-37.0);
        QCOMPARE(s.value(), -37.0);
    }

    void setValue_roundTrip_atMin() {
        ParamSlider s("Test", -100.0, 100.0);
        s.setValue(-100.0);
        QCOMPARE(s.value(), -100.0);
    }

    void setValue_roundTrip_atMax() {
        ParamSlider s("Test", -100.0, 100.0);
        s.setValue(100.0);
        QCOMPARE(s.value(), 100.0);
    }

    // 3.5 is exactly representable in binary floating point
    void setValue_roundTrip_fractionalStep() {
        ParamSlider s("Test", -20.0, 20.0, 0.1, 1);
        s.setValue(3.5);
        QCOMPARE(s.value(), 3.5);
    }

    void setValue_roundTrip_negativeFractional() {
        ParamSlider s("Test", -20.0, 20.0, 0.1, 1);
        s.setValue(-7.5);
        QCOMPARE(s.value(), -7.5);
    }

    // setValue() must not emit — callers use it to restore state silently
    void setValue_doesNotEmitValueChanged() {
        ParamSlider s("Test", -100.0, 100.0);
        QSignalSpy spy(&s, &ParamSlider::valueChanged);
        s.setValue(50.0);
        QCOMPARE(spy.count(), 0);
    }

    void setValue_doesNotEmitEditingFinished() {
        ParamSlider s("Test", -100.0, 100.0);
        QSignalSpy spy(&s, &ParamSlider::editingFinished);
        s.setValue(50.0);
        QCOMPARE(spy.count(), 0);
    }

    void setValue_multipleTimesNoSignals() {
        ParamSlider s("Test", -100.0, 100.0);
        QSignalSpy spy(&s, &ParamSlider::valueChanged);
        s.setValue(10.0);
        s.setValue(-10.0);
        s.setValue(0.0);
        QCOMPARE(spy.count(), 0);
    }

    // Slider range: constructed with integer steps, verify min/max round-trip
    void integerSlider_minAndMax() {
        ParamSlider s("Radius", 0.0, 100.0);
        s.setValue(0.0);
        QCOMPARE(s.value(), 0.0);
        s.setValue(100.0);
        QCOMPARE(s.value(), 100.0);
    }
};

QTEST_MAIN(TestParamSlider)
#include "test_param_slider.moc"
