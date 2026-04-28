#include <QTest>
#include <QSignalSpy>
#include <memory>
#include "EffectManager.h"
#include "ICropSource.h"

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

// MockEffect that also implements ICropSource — exercises the
// addEffect() interface-pointer caching and cropSource() lookups.
class MockCropEffect : public PhotoEditorEffect, public ICropSource {
    Q_OBJECT
public:
    QString getName()        const override { return "MockCrop"; }
    QString getDescription() const override { return ""; }
    QString getVersion()     const override { return "1.0"; }
    bool    initialize()           override { return true; }
    QImage processImage(const QImage& img, const QMap<QString,QVariant>&) override { return img; }
    QRectF userCropRect()    const override { return {0, 0, 1, 1}; }
    float  userCropAngle()   const override { return 0.0f; }
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

    // ── cropSource caching ────────────────────────────────────────────────────
    void cropSource_isNullWhenNoCropEffectAdded() {
        EffectManager mgr;
        mgr.addEffect(std::make_unique<MockEffect>());
        QVERIFY(mgr.cropSource() == nullptr);
        QVERIFY(mgr.activeCropSource() == nullptr);
    }

    void cropSource_returnsCachedPointerEvenWhenDisabled() {
        EffectManager mgr;
        auto crop = std::make_unique<MockCropEffect>();
        auto* cropRaw = crop.get();
        mgr.addEffect(std::move(crop), /*enabled=*/false);
        QCOMPARE(static_cast<ICropSource*>(cropRaw), mgr.cropSource());
        // activeCropSource respects the enabled bit
        QVERIFY(mgr.activeCropSource() == nullptr);
    }

    void activeCropSource_returnsPointerWhenEnabled() {
        EffectManager mgr;
        auto crop = std::make_unique<MockCropEffect>();
        auto* cropRaw = crop.get();
        mgr.addEffect(std::move(crop));
        QCOMPARE(static_cast<ICropSource*>(cropRaw), mgr.activeCropSource());
    }

    void activeCropSource_clearsWhenEffectDisabledAtRuntime() {
        EffectManager mgr;
        mgr.addEffect(std::make_unique<MockCropEffect>());
        QVERIFY(mgr.activeCropSource() != nullptr);
        mgr.setEnabled(0, false);
        QVERIFY(mgr.activeCropSource() == nullptr);
        QVERIFY(mgr.cropSource() != nullptr);  // cache survives toggle
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
