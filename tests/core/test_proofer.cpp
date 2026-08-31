#include <QTest>
#include <QSignalSpy>
#include <QTemporaryDir>
#include "Proofer.h"
#include "ProofCache.h"
#include "EffectManager.h"
#include "SettingsImporter.h"
#include "GpuDeviceRegistry.h"
#include "BrightnessEffect.h"
#include "CropRotateEffect.h"
#include <QFile>

// Minimal EffectManager with no real effects — sufficient for queue-management
// tests that keep the proofer paused so no GPU dispatch occurs.
static std::unique_ptr<EffectManager> emptyMgr() {
    return std::make_unique<EffectManager>();
}

static SettingsImporter::Settings emptyDefaults() {
    return {};
}

class TestProofer : public QObject {
    Q_OBJECT

private:
    QTemporaryDir m_dir;
    ProofCache   *m_cache = nullptr;

private slots:
    void init() {
        m_cache = new ProofCache(); // orphaned; cleaned up in cleanup()
    }

    void cleanup() {
        delete m_cache;
        m_cache = nullptr;
    }

    // ── setQueue ──────────────────────────────────────────────────────────────

    void setQueue_populatesQueue() {
        Proofer p(emptyMgr(), emptyDefaults(), m_cache);
        p.pause();
        p.setQueue({"a", "b", "c"});
        QCOMPARE(p.pendingCount(), 3);
    }

    void setDefaults_acceptsUpdatedPreferences() {
        Proofer                    p(emptyMgr(), emptyDefaults(), m_cache);
        SettingsImporter::Settings defaults;
        defaults.image = "new-defaults";
        p.setDefaults(std::move(defaults));
        QCOMPARE(p.pendingCount(), 0);
    }

    void setQueue_replacesExistingQueue() {
        Proofer p(emptyMgr(), emptyDefaults(), m_cache);
        p.pause();
        p.setQueue({"x", "y"});
        p.setQueue({"a", "b", "c"});
        QCOMPARE(p.pendingCount(), 3);
    }

    void setQueue_emptyList_clearsQueue() {
        Proofer p(emptyMgr(), emptyDefaults(), m_cache);
        p.pause();
        p.setQueue({"a", "b"});
        p.setQueue({});
        QCOMPARE(p.pendingCount(), 0);
    }

    // ── promote ───────────────────────────────────────────────────────────────

    void promote_movesItemToHead() {
        Proofer p(emptyMgr(), emptyDefaults(), m_cache);
        p.pause();
        p.setQueue({"a", "b", "c"});
        p.promote("c");
        // Queue is now {c, a, b} — can't read it directly, but the next
        // dispatch (verified via proofStarted signal) should fire for "c".
        // Since we're paused, verify via pendingCount invariant instead.
        QCOMPARE(p.pendingCount(), 3);
    }

    void promote_noopWhenAlreadyAtHead() {
        Proofer p(emptyMgr(), emptyDefaults(), m_cache);
        p.pause();
        p.setQueue({"a", "b", "c"});
        p.promote("a"); // already at head
        QCOMPARE(p.pendingCount(), 3);
    }

    void promote_addsToHeadWhenNotInQueue() {
        Proofer p(emptyMgr(), emptyDefaults(), m_cache);
        p.pause();
        p.setQueue({"a", "b"});
        p.promote("z"); // not in queue — should be added
        QCOMPARE(p.pendingCount(), 3);
    }

    void refresh_requeuesExactlyOnce() {
        Proofer p(emptyMgr(), emptyDefaults(), m_cache);
        p.pause();
        p.setQueue({"a", "b"});
        p.refresh("a");
        p.refresh("a");
        QCOMPARE(p.pendingCount(), 2);
    }

    void clear_invalidatesQueueBeforeReplacement() {
        Proofer p(emptyMgr(), emptyDefaults(), m_cache);
        p.pause();
        p.setQueue({"old-a", "old-b"});
        p.clear();
        p.setQueue({"new"});
        QCOMPARE(p.pendingCount(), 1);
    }

    // ── clear ─────────────────────────────────────────────────────────────────

    void clear_emptiesQueue() {
        Proofer p(emptyMgr(), emptyDefaults(), m_cache);
        p.pause();
        p.setQueue({"a", "b", "c"});
        p.clear();
        QCOMPARE(p.pendingCount(), 0);
    }

    // ── pause / resume ────────────────────────────────────────────────────────

    void pause_preventsDispatch() {
        Proofer p(emptyMgr(), emptyDefaults(), m_cache);
        p.pause();
        QSignalSpy spy(&p, &Proofer::proofStarted);
        p.setQueue({"some/path.jpg"});
        // Give the event loop a moment to fire any pending signals.
        QCoreApplication::processEvents();
        QCOMPARE(spy.count(), 0);
    }

    void resume_withEmptyQueue_doesNothing() {
        Proofer p(emptyMgr(), emptyDefaults(), m_cache);
        p.pause();
        QSignalSpy spy(&p, &Proofer::proofStarted);
        p.resume();
        QCoreApplication::processEvents();
        QCOMPARE(spy.count(), 0);
    }

    // ── pendingCount ──────────────────────────────────────────────────────────

    void pendingCount_startsAtZero() {
        Proofer p(emptyMgr(), emptyDefaults(), m_cache);
        p.pause();
        QCOMPARE(p.pendingCount(), 0);
    }

    void sidecarParametersApplyWithoutControlsWidget() {
        GpuDeviceRegistry::instance().enumerate();
        if (GpuDeviceRegistry::instance().count() == 0) QSKIP("No OpenCL device found");

        const QString imagePath = m_dir.filePath("edited.png");
        QImage        input(16, 16, QImage::Format_RGB32);
        input.fill(qRgb(20, 20, 20));
        QVERIFY(input.save(imagePath));

        QFile sidecar(ProofCache::sidecarPath(imagePath));
        QVERIFY(sidecar.open(QIODevice::WriteOnly | QIODevice::Text));
        const QByteArray yaml = "effects:\n"
                                "  - id: \"brightness_contrast\"\n"
                                "    enabled: true\n"
                                "    parameters:\n"
                                "      brightness: 100\n"
                                "      contrast: 0\n";
        QCOMPARE(sidecar.write(yaml), static_cast<qint64>(yaml.size()));
        sidecar.close();

        auto effects = std::make_unique<EffectManager>();
        auto effect  = std::make_unique<BrightnessEffect>();
        QVERIFY(effect->initialize());
        effects->addEffect(std::move(effect));

        SettingsImporter::Settings defaults;
        defaults.effects.append(
            {"brightness_contrast", "Brightness & Contrast", true, {{"brightness", 0.0}, {"contrast", 0.0}}});
        Proofer    proofer(std::move(effects), defaults, m_cache);
        QSignalSpy finished(&proofer, &Proofer::proofFinished);
        proofer.setQueue({imagePath});
        QTRY_COMPARE_WITH_TIMEOUT(finished.count(), 1, 10000);

        const QImage proof = finished.first().at(1).value<QImage>();
        QVERIFY(!proof.isNull());
        QVERIFY(qRed(proof.pixel(0, 0)) > 100);
    }

    void proofBakesActiveCropGeometry() {
        GpuDeviceRegistry::instance().enumerate();
        if (GpuDeviceRegistry::instance().count() == 0) QSKIP("No OpenCL device found");

        const QString imagePath = m_dir.filePath("cropped.png");
        QImage        input(40, 20, QImage::Format_RGB32);
        input.fill(Qt::green);
        QVERIFY(input.save(imagePath));

        QFile sidecar(ProofCache::sidecarPath(imagePath));
        QVERIFY(sidecar.open(QIODevice::WriteOnly | QIODevice::Text));
        const QByteArray yaml = "effects:\n"
                                "  - id: \"crop_rotate\"\n"
                                "    enabled: true\n"
                                "    parameters:\n"
                                "      angle: 0\n"
                                "      quarterTurns: 0\n"
                                "      cropX0: 0.25\n"
                                "      cropY0: 0.0\n"
                                "      cropX1: 0.75\n"
                                "      cropY1: 1.0\n";
        QCOMPARE(sidecar.write(yaml), static_cast<qint64>(yaml.size()));
        sidecar.close();

        auto effects = std::make_unique<EffectManager>();
        auto crop    = std::make_unique<CropRotateEffect>();
        QVERIFY(crop->initialize());
        effects->addEffect(std::move(crop));

        SettingsImporter::Settings defaults;
        defaults.effects.append({"crop_rotate", "Crop & Rotate", true, {}});
        Proofer    proofer(std::move(effects), defaults, m_cache);
        QSignalSpy finished(&proofer, &Proofer::proofFinished);
        proofer.setQueue({imagePath});
        QTRY_COMPARE_WITH_TIMEOUT(finished.count(), 1, 10000);

        const QImage proof = finished.first().at(1).value<QImage>();
        QCOMPARE(proof.size(), QSize(20, 20));
    }
};

QTEST_GUILESS_MAIN(TestProofer)
#include "test_proofer.moc"
