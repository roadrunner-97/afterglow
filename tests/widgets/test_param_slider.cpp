#include <QTest>
#include <QSignalSpy>
#include <QSlider>
#include <QDoubleSpinBox>
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

    // --- Constructor lambda coverage ---

    // QSlider::valueChanged lambda: syncs spinbox, updates label, emits valueChanged
    void sliderValueChanged_emitsValueChangedAndSyncsSpinbox() {
        ParamSlider s("Test", -100.0, 100.0);
        QSignalSpy spy(&s, &ParamSlider::valueChanged);
        auto* slider = s.findChild<QSlider*>();
        QVERIFY(slider);
        slider->setValue(50);
        QCOMPARE(spy.count(), 1);
        QCOMPARE(spy.at(0).at(0).toDouble(), 50.0);
        QCOMPARE(s.value(), 50.0);
    }

    // QSlider::valueChanged lambda with fractional scale factor
    void sliderValueChanged_fractionalStep_scaledCorrectly() {
        ParamSlider s("Test", -20.0, 20.0, 0.1, 1);
        QSignalSpy spy(&s, &ParamSlider::valueChanged);
        auto* slider = s.findChild<QSlider*>();
        QVERIFY(slider);
        // scaleFactor = 10; int value 35 → 3.5
        slider->setValue(35);
        QCOMPARE(spy.count(), 1);
        QCOMPARE(spy.at(0).at(0).toDouble(), 3.5);
        QCOMPARE(s.value(), 3.5);
    }

    // QSlider::sliderReleased lambda: emits editingFinished
    void sliderReleased_emitsEditingFinished() {
        ParamSlider s("Test", -100.0, 100.0);
        QSignalSpy spy(&s, &ParamSlider::editingFinished);
        auto* slider = s.findChild<QSlider*>();
        QVERIFY(slider);
        QMetaObject::invokeMethod(slider, "sliderReleased");
        QCOMPARE(spy.count(), 1);
    }

    // QDoubleSpinBox::valueChanged lambda: syncs slider, does NOT emit ParamSlider::valueChanged
    void spinboxValueChanged_syncsSliderAndNoSignal() {
        ParamSlider s("Test", -100.0, 100.0);
        QSignalSpy spyValue(&s, &ParamSlider::valueChanged);
        auto* spinbox = s.findChild<QDoubleSpinBox*>();
        auto* slider  = s.findChild<QSlider*>();
        QVERIFY(spinbox);
        QVERIFY(slider);
        spinbox->setValue(30.0);
        // The spinbox lambda syncs the slider but must NOT fire ParamSlider::valueChanged
        QCOMPARE(spyValue.count(), 0);
        QCOMPARE(slider->value(), 30);
        QCOMPARE(s.value(), 30.0);
    }

    // QDoubleSpinBox::editingFinished lambda: emits both valueChanged and editingFinished
    void spinboxEditingFinished_emitsBothSignals() {
        ParamSlider s("Test", -100.0, 100.0);
        QSignalSpy spyValue(&s, &ParamSlider::valueChanged);
        QSignalSpy spyEdit(&s, &ParamSlider::editingFinished);
        auto* spinbox = s.findChild<QDoubleSpinBox*>();
        QVERIFY(spinbox);
        spinbox->setValue(25.0);
        // Invoke editingFinished signal directly (simulates user pressing Enter)
        QMetaObject::invokeMethod(spinbox, "editingFinished");
        QCOMPARE(spyValue.count(), 1);
        QCOMPARE(spyValue.at(0).at(0).toDouble(), 25.0);
        QCOMPARE(spyEdit.count(), 1);
    }

    // --- eventFilter coverage ---

    // Left-button double-click on slider resets value to 0 and emits both signals
    void eventFilter_leftDoublClick_resetsToZeroAndEmits() {
        ParamSlider s("Test", -100.0, 100.0);
        s.show();
        s.setValue(75.0);
        QSignalSpy spyValue(&s, &ParamSlider::valueChanged);
        QSignalSpy spyEdit(&s, &ParamSlider::editingFinished);
        auto* slider = s.findChild<QSlider*>();
        QVERIFY(slider);
        QTest::mouseDClick(slider, Qt::LeftButton);
        QCOMPARE(s.value(), 0.0);
        QCOMPARE(spyValue.count(), 1);
        QCOMPARE(spyValue.at(0).at(0).toDouble(), 0.0);
        QCOMPARE(spyEdit.count(), 1);
    }

    // Right-button double-click is a no-op (falls through to base eventFilter)
    void eventFilter_rightDoubleClick_noOp() {
        ParamSlider s("Test", -100.0, 100.0);
        s.show();
        s.setValue(75.0);
        QSignalSpy spyEdit(&s, &ParamSlider::editingFinished);
        auto* slider = s.findChild<QSlider*>();
        QVERIFY(slider);
        QTest::mouseDClick(slider, Qt::RightButton);
        // Value unchanged, no editingFinished from eventFilter path
        QCOMPARE(s.value(), 75.0);
        QCOMPARE(spyEdit.count(), 0);
    }
};

QTEST_MAIN(TestParamSlider)
#include "test_param_slider.moc"
