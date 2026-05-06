#include <QTest>
#include <QApplication>
#include <QMouseEvent>
#include <QSignalSpy>
#include <QSlider>
#include <QDoubleSpinBox>
#include "ParamSlider.h"

namespace {

// Send a QMouseEvent with a fixed timestamp (the velocity-gain drag math
// divides by Δt, so we need deterministic values across runs).  Keeps the
// event on the stack — QMouseEvent's copy/move constructors are deleted.
void sendMouseEvent(QObject* target, QEvent::Type type, QPoint pos,
                    Qt::MouseButton button,
                    Qt::KeyboardModifiers mods,
                    qulonglong tsMs) {
    QMouseEvent ev(type, QPointF(pos), QPointF(pos),
                   button,
                   button == Qt::NoButton ? Qt::LeftButton : button,
                   mods);
    ev.setTimestamp(tsMs);
    QApplication::sendEvent(target, &ev);
}

} // namespace

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

    // --- Velocity-gain drag handlers ---

    // Right-button press is ignored (only left starts a drag).
    void drag_pressNonLeftButton_passesThrough() {
        ParamSlider s("Test", -100.0, 100.0);
        s.show();
        s.resize(200, 30);
        auto* slider = s.findChild<QSlider*>();
        QVERIFY(slider);
        QSignalSpy spyEdit(&s, &ParamSlider::editingFinished);
        sendMouseEvent(slider, QEvent::MouseButtonPress, QPoint(50, 10),
                       Qt::RightButton, Qt::NoModifier, 100);
        QCOMPARE(spyEdit.count(), 0);  // never started a drag
    }

    // MouseMove with no prior press is a no-op (handleDragMove early-out).
    void drag_moveBeforePress_noOp() {
        ParamSlider s("Test", -100.0, 100.0);
        s.show();
        s.resize(200, 30);
        auto* slider = s.findChild<QSlider*>();
        QVERIFY(slider);
        const int before = slider->value();
        sendMouseEvent(slider, QEvent::MouseMove, QPoint(80, 10),
                       Qt::NoButton, Qt::NoModifier, 100);
        QCOMPARE(slider->value(), before);  // untouched
    }

    // Press → fast move → release.  A wide horizontal drag at gain≥1 sweeps
    // the slider toward its maximum and emits editingFinished on release.
    void drag_pressMoveRelease_advancesAndEmitsEditingFinished() {
        ParamSlider s("Test", -100.0, 100.0);
        s.show();
        s.resize(200, 30);
        auto* slider = s.findChild<QSlider*>();
        QVERIFY(slider);
        slider->resize(180, 20);
        s.setValue(0.0);

        QSignalSpy spyValue(&s, &ParamSlider::valueChanged);
        QSignalSpy spyEdit (&s, &ParamSlider::editingFinished);

        sendMouseEvent(slider, QEvent::MouseButtonPress, QPoint(20, 10),
                       Qt::LeftButton, Qt::NoModifier, 100);
        // 60-px sweep over 100 ms → ~600 px/s → gain ≈ 1.25×.
        sendMouseEvent(slider, QEvent::MouseMove, QPoint(80, 10),
                       Qt::NoButton, Qt::NoModifier, 200);
        sendMouseEvent(slider, QEvent::MouseButtonRelease, QPoint(80, 10),
                       Qt::LeftButton, Qt::NoModifier, 210);

        QVERIFY(s.value() > 0.0);          // moved positively
        QVERIFY(spyValue.count() >= 1);     // at least one tick
        QCOMPARE(spyEdit.count(), 1);       // released after motion
    }

    // Same-pixel MouseMove (dx == 0) is the early-out branch — it must
    // still update the timestamp without changing the slider.
    void drag_moveZeroDelta_noChangeButTimestampAdvances() {
        ParamSlider s("Test", -100.0, 100.0);
        s.show();
        s.resize(200, 30);
        auto* slider = s.findChild<QSlider*>();
        slider->resize(180, 20);
        s.setValue(0.0);

        sendMouseEvent(slider, QEvent::MouseButtonPress, QPoint(50, 10),
                       Qt::LeftButton, Qt::NoModifier, 100);
        sendMouseEvent(slider, QEvent::MouseMove, QPoint(50, 10),
                       Qt::NoButton, Qt::NoModifier, 200);
        QCOMPARE(s.value(), 0.0);

        // Now a real dx — should advance from the updated timestamp baseline.
        sendMouseEvent(slider, QEvent::MouseMove, QPoint(70, 10),
                       Qt::NoButton, Qt::NoModifier, 250);
        QVERIFY(s.value() > 0.0);
    }

    // Shift modifier forces fine-grain (gain capped at 0.4) — same drag
    // produces a smaller slider movement than the unmodified case above.
    void drag_shiftModifier_capsGainForFineGrain() {
        auto runDrag = [](Qt::KeyboardModifiers mods) -> double {
            ParamSlider s("Test", -100.0, 100.0);
            s.show();
            s.resize(200, 30);
            auto* slider = s.findChild<QSlider*>();
            slider->resize(180, 20);
            s.setValue(0.0);
            sendMouseEvent(slider, QEvent::MouseButtonPress, QPoint(20, 10),
                           Qt::LeftButton, Qt::NoModifier, 100);
            sendMouseEvent(slider, QEvent::MouseMove, QPoint(80, 10),
                           Qt::NoButton, mods, 200);
            return s.value();
        };
        const double unshifted = runDrag(Qt::NoModifier);
        const double shifted   = runDrag(Qt::ShiftModifier);
        QVERIFY(unshifted > 0.0);
        QVERIFY(shifted   > 0.0);
        QVERIFY(shifted < unshifted);            // Shift compressed the move
        QVERIFY(shifted < 0.5 * unshifted);      // by a meaningful margin
    }

    // Release without prior motion (a "click that didn't drag") must NOT
    // emit editingFinished — only motion-then-release does.
    void drag_releaseWithoutMove_doesNotEmitEditingFinished() {
        ParamSlider s("Test", -100.0, 100.0);
        s.show();
        s.resize(200, 30);
        auto* slider = s.findChild<QSlider*>();
        slider->resize(180, 20);

        QSignalSpy spyEdit(&s, &ParamSlider::editingFinished);
        sendMouseEvent(slider, QEvent::MouseButtonPress, QPoint(50, 10),
                       Qt::LeftButton, Qt::NoModifier, 100);
        sendMouseEvent(slider, QEvent::MouseButtonRelease, QPoint(50, 10),
                       Qt::LeftButton, Qt::NoModifier, 110);
        QCOMPARE(spyEdit.count(), 0);
    }

    // Right-button release outside a drag returns false (passes through).
    void drag_releaseNonLeftButton_passesThrough() {
        ParamSlider s("Test", -100.0, 100.0);
        s.show();
        s.resize(200, 30);
        auto* slider = s.findChild<QSlider*>();
        QSignalSpy spyEdit(&s, &ParamSlider::editingFinished);
        sendMouseEvent(slider, QEvent::MouseButtonRelease, QPoint(50, 10),
                       Qt::RightButton, Qt::NoModifier, 100);
        QCOMPARE(spyEdit.count(), 0);
    }

    // A flurry of same-tick events (Δt < 4 ms) must not blow the velocity
    // gain up via division by zero — the move handler clamps Δt at 4 ms.
    void drag_sameTickEvents_clampsTimeDelta() {
        ParamSlider s("Test", -100.0, 100.0);
        s.show();
        s.resize(200, 30);
        auto* slider = s.findChild<QSlider*>();
        slider->resize(180, 20);
        s.setValue(0.0);

        sendMouseEvent(slider, QEvent::MouseButtonPress, QPoint(20, 10),
                       Qt::LeftButton, Qt::NoModifier, 100);
        sendMouseEvent(slider, QEvent::MouseMove, QPoint(40, 10),
                       Qt::NoButton, Qt::NoModifier, 100);  // Δt = 0
        // Should still produce a finite, positive value bounded by max gain.
        QVERIFY(s.value() > 0.0);
        QVERIFY(s.value() <= 100.0);
    }

    // Drag clamped to slider's [min, max] range even when the velocity-gain
    // math would push past it.
    void drag_clampsToSliderRange() {
        ParamSlider s("Test", -100.0, 100.0);
        s.show();
        s.resize(200, 30);
        auto* slider = s.findChild<QSlider*>();
        slider->resize(180, 20);
        s.setValue(80.0);

        sendMouseEvent(slider, QEvent::MouseButtonPress, QPoint(20, 10),
                       Qt::LeftButton, Qt::NoModifier, 100);
        // Huge fast drag → would exceed max without clamping.
        sendMouseEvent(slider, QEvent::MouseMove, QPoint(180, 10),
                       Qt::NoButton, Qt::NoModifier, 110);
        QCOMPARE(s.value(), 100.0);
    }

    // Unhandled event types fall through to the base QWidget::eventFilter,
    // exercising the default-branch return path in eventFilter().
    void eventFilter_otherEventType_fallsThrough() {
        ParamSlider s("Test", -100.0, 100.0);
        s.show();
        auto* slider = s.findChild<QSlider*>();
        QVERIFY(slider);
        QEvent ev(QEvent::FocusIn);
        QApplication::sendEvent(slider, &ev);  // shouldn't crash
        QVERIFY(true);
    }
};

QTEST_MAIN(TestParamSlider)
#include "test_param_slider.moc"
