#include <QTest>
#include <QSignalSpy>
#include <memory>
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
        mgr.addEffect(std::make_unique<MockEffect>());
        QCOMPARE(mgr.entries().size(), 1);
    }

    void addEffect_defaultEnabledTrue() {
        EffectManager mgr;
        mgr.addEffect(std::make_unique<MockEffect>());
        QVERIFY(mgr.entries()[0].enabled);
    }

    void addEffect_disabledInitially() {
        EffectManager mgr;
        mgr.addEffect(std::make_unique<MockEffect>(), /*enabled=*/false);
        QVERIFY(!mgr.entries()[0].enabled);
    }

    void setEnabled_togglesState() {
        EffectManager mgr;
        mgr.addEffect(std::make_unique<MockEffect>()); // starts enabled
        mgr.setEnabled(0, false);
        QVERIFY(!mgr.entries()[0].enabled);
        mgr.setEnabled(0, true);
        QVERIFY(mgr.entries()[0].enabled);
    }

    void setEnabled_emitsSignal() {
        EffectManager mgr;
        mgr.addEffect(std::make_unique<MockEffect>());
        QSignalSpy spy(&mgr, &EffectManager::effectToggled);

        mgr.setEnabled(0, false);

        QCOMPARE(spy.count(), 1);
        QCOMPARE(spy[0][0].toInt(),   0);     // index
        QCOMPARE(spy[0][1].toBool(),  false); // enabled
    }

    void setEnabled_emitsCorrectIndexForMultipleEffects() {
        EffectManager mgr;
        mgr.addEffect(std::make_unique<MockEffect>());
        mgr.addEffect(std::make_unique<MockEffect>());
        QSignalSpy spy(&mgr, &EffectManager::effectToggled);

        mgr.setEnabled(1, false);

        QCOMPARE(spy.count(), 1);
        QCOMPARE(spy[0][0].toInt(), 1);
    }

    void setEnabled_outOfRangeIsNoop() {
        EffectManager mgr;
        mgr.addEffect(std::make_unique<MockEffect>());

        // Must not crash or alter any entry's state
        mgr.setEnabled(-1, false);
        mgr.setEnabled(99, false);

        QVERIFY(mgr.entries()[0].enabled);
    }

    void setEnabled_outOfRangeEmitsNoSignal() {
        EffectManager mgr;
        mgr.addEffect(std::make_unique<MockEffect>());
        QSignalSpy spy(&mgr, &EffectManager::effectToggled);

        mgr.setEnabled(-1, false);
        mgr.setEnabled(99, false);

        QCOMPARE(spy.count(), 0);
    }

    void multipleEffects_orderedByInsertion() {
        EffectManager mgr;
        auto a = std::make_unique<MockEffect>();
        auto b = std::make_unique<MockEffect>();
        auto* aRaw = a.get();
        auto* bRaw = b.get();
        mgr.addEffect(std::move(a));
        mgr.addEffect(std::move(b));

        QCOMPARE(mgr.entries().size(), 2);
        QCOMPARE(mgr.entries()[0].effect, aRaw);
        QCOMPARE(mgr.entries()[1].effect, bRaw);
    }

    void activeEffects_emptyWhenNoneAdded() {
        EffectManager mgr;
        QVERIFY(mgr.activeEffects().isEmpty());
    }

    void activeEffects_returnsOnlyEnabled() {
        EffectManager mgr;
        auto a = std::make_unique<MockEffect>();
        auto b = std::make_unique<MockEffect>();
        auto c = std::make_unique<MockEffect>();
        auto* aRaw = a.get();
        auto* cRaw = c.get();
        mgr.addEffect(std::move(a), /*enabled=*/true);
        mgr.addEffect(std::move(b), /*enabled=*/false);
        mgr.addEffect(std::move(c), /*enabled=*/true);

        const auto active = mgr.activeEffects();
        QCOMPARE(active.size(), 2);
        QCOMPARE(active[0], aRaw);
        QCOMPARE(active[1], cRaw);
    }

    void activeEffects_reflectsRuntimeToggles() {
        EffectManager mgr;
        auto a = std::make_unique<MockEffect>();
        auto b = std::make_unique<MockEffect>();
        auto* bRaw = b.get();
        mgr.addEffect(std::move(a));
        mgr.addEffect(std::move(b));

        QCOMPARE(mgr.activeEffects().size(), 2);
        mgr.setEnabled(0, false);
        const auto active = mgr.activeEffects();
        QCOMPARE(active.size(), 1);
        QCOMPARE(active[0], bRaw);
    }

    void setEnabled_doesNotAffectOtherEffects() {
        EffectManager mgr;
        mgr.addEffect(std::make_unique<MockEffect>());
        mgr.addEffect(std::make_unique<MockEffect>());

        mgr.setEnabled(0, false);

        QVERIFY(!mgr.entries()[0].enabled);
        QVERIFY( mgr.entries()[1].enabled); // untouched
    }

    // ── PhotoEditorEffect base class default methods ──────────────────────────
    // MockEffect does NOT override createControlsWidget, getParameters,
    // onImageLoaded, or supportsGpuInPlace — so these calls exercise the
    // default implementations in PhotoEditorEffect.h.

    void baseClass_createControlsWidget_returnsNull() {
        MockEffect e;
        QVERIFY(e.createControlsWidget() == nullptr);
    }

    void baseClass_getParameters_returnsEmpty() {
        MockEffect e;
        QVERIFY(e.getParameters().isEmpty());
    }

    void baseClass_onImageLoaded_doesNotCrash() {
        MockEffect e;
        ImageMetadata meta;
        meta.colorTempK = 5500.0f;
        e.onImageLoaded(meta);  // default impl is a no-op
    }

    void baseClass_supportsGpuInPlace_returnsFalse() {
        MockEffect e;
        QVERIFY(!e.supportsGpuInPlace());
    }

    // Heap-allocate EffectManager so the unique_ptr-driven cleanup runs and
    // ASan/leak-sanitiser would catch a missed destruction.
    void destructor_heapAllocated_deletesEffects() {
        auto* mgr = new EffectManager();
        mgr->addEffect(std::make_unique<MockEffect>());
        mgr->addEffect(std::make_unique<MockEffect>());
        delete mgr;
    }
};

QTEST_GUILESS_MAIN(TestEffectManager)
#include "test_effect_manager.moc"
