#include <QTest>
#include <QSignalSpy>
#include "EffectManager.h"

// Minimal concrete PhotoEditorEffect for testing orchestration logic.
// Returns the input image unchanged — no GPU, no OpenCL required.
class MockEffect : public PhotoEditorEffect {
    Q_OBJECT
public:
    QString getName()        const override { return "Mock"; }
    QString getDescription() const override { return "Test effect"; }
    QString getVersion()     const override { return "1.0"; }
    bool    initialize()           override { return true; }

    QImage processImage(const QImage& img, const QMap<QString,QVariant>&) override {
        return img;
    }
};

class TestEffectManager : public QObject {
    Q_OBJECT

private slots:
    void defaultIsEmpty() {
        EffectManager mgr;
        QCOMPARE(mgr.entries().size(), 0);
    }

    void addEffect_appendsEntry() {
        EffectManager mgr;
        mgr.addEffect(new MockEffect());
        QCOMPARE(mgr.entries().size(), 1);
    }

    void addEffect_defaultEnabledTrue() {
        EffectManager mgr;
        mgr.addEffect(new MockEffect());
        QVERIFY(mgr.entries()[0].enabled);
    }

    void addEffect_disabledInitially() {
        EffectManager mgr;
        mgr.addEffect(new MockEffect(), /*enabled=*/false);
        QVERIFY(!mgr.entries()[0].enabled);
    }

    void setEnabled_togglesState() {
        EffectManager mgr;
        mgr.addEffect(new MockEffect()); // starts enabled
        mgr.setEnabled(0, false);
        QVERIFY(!mgr.entries()[0].enabled);
        mgr.setEnabled(0, true);
        QVERIFY(mgr.entries()[0].enabled);
    }

    void setEnabled_emitsSignal() {
        EffectManager mgr;
        mgr.addEffect(new MockEffect());
        QSignalSpy spy(&mgr, &EffectManager::effectToggled);

        mgr.setEnabled(0, false);

        QCOMPARE(spy.count(), 1);
        QCOMPARE(spy[0][0].toInt(),   0);     // index
        QCOMPARE(spy[0][1].toBool(),  false); // enabled
    }

    void setEnabled_emitsCorrectIndexForMultipleEffects() {
        EffectManager mgr;
        mgr.addEffect(new MockEffect());
        mgr.addEffect(new MockEffect());
        QSignalSpy spy(&mgr, &EffectManager::effectToggled);

        mgr.setEnabled(1, false);

        QCOMPARE(spy.count(), 1);
        QCOMPARE(spy[0][0].toInt(), 1);
    }

    void setEnabled_outOfRangeIsNoop() {
        EffectManager mgr;
        mgr.addEffect(new MockEffect());

        // Must not crash or alter any entry's state
        mgr.setEnabled(-1, false);
        mgr.setEnabled(99, false);

        QVERIFY(mgr.entries()[0].enabled);
    }

    void setEnabled_outOfRangeEmitsNoSignal() {
        EffectManager mgr;
        mgr.addEffect(new MockEffect());
        QSignalSpy spy(&mgr, &EffectManager::effectToggled);

        mgr.setEnabled(-1, false);
        mgr.setEnabled(99, false);

        QCOMPARE(spy.count(), 0);
    }

    void multipleEffects_orderedByInsertion() {
        EffectManager mgr;
        auto* a = new MockEffect();
        auto* b = new MockEffect();
        mgr.addEffect(a);
        mgr.addEffect(b);

        QCOMPARE(mgr.entries().size(), 2);
        QCOMPARE(mgr.entries()[0].effect, a);
        QCOMPARE(mgr.entries()[1].effect, b);
    }

    void setEnabled_doesNotAffectOtherEffects() {
        EffectManager mgr;
        mgr.addEffect(new MockEffect());
        mgr.addEffect(new MockEffect());

        mgr.setEnabled(0, false);

        QVERIFY(!mgr.entries()[0].enabled);
        QVERIFY( mgr.entries()[1].enabled); // untouched
    }
};

QTEST_GUILESS_MAIN(TestEffectManager)
#include "test_effect_manager.moc"
