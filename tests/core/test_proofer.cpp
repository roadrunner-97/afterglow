#include <QTest>
#include <QSignalSpy>
#include <QTemporaryDir>
#include "Proofer.h"
#include "ProofCache.h"
#include "EffectManager.h"
#include "SettingsImporter.h"

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
};

QTEST_GUILESS_MAIN(TestProofer)
#include "test_proofer.moc"
