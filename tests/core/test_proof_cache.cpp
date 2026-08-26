#include <QTest>
#include <QDir>
#include <QFile>
#include <QFileDevice>
#include <QFileInfo>
#include <QImage>
#include <QTemporaryDir>
#include <memory>
#include "ProofCache.h"

static bool writeJpeg(const QString &path, QRgb colour = qRgb(100, 150, 200)) {
    QDir().mkpath(QFileInfo(path).absolutePath());
    QImage img(8, 8, QImage::Format_RGB32);
    img.fill(colour);
    return img.save(path, "JPEG", 90);
}

// Create an empty file at path (creates parent dirs if needed).
static void touchEmpty(const QString &path) {
    QDir().mkpath(QFileInfo(path).absolutePath());
    QFile f(path);
    if (!f.open(QIODevice::WriteOnly | QIODevice::Append)) qFatal("touchEmpty: cannot open %s", qPrintable(path));
    f.close();
}

class TestProofCache : public QObject {
    Q_OBJECT

private:
    std::unique_ptr<QTemporaryDir> m_dir;
    QString                        m_img;

private slots:
    void init() {
        m_dir = std::make_unique<QTemporaryDir>();
        QVERIFY(m_dir->isValid());
        m_img = m_dir->filePath("IMG_0001.CR2");
        // Placeholder so the directory is valid for QFileInfo.
        touchEmpty(m_img);
    }

    void cleanup() {
        m_dir.reset();
    }

    // ── proofPath / sidecarPath ───────────────────────────────────────────────

    void proofPath_containsAfterglow() {
        QVERIFY(ProofCache::proofPath(m_img).contains(".afterglow/proofs/"));
    }

    void proofPath_endsWithJpg() {
        QVERIFY(ProofCache::proofPath(m_img).endsWith(".jpg"));
    }

    void proofPath_disambiguatesExtension() {
        const QString cr2 = m_dir->filePath("IMG_0001.CR2");
        const QString nef = m_dir->filePath("IMG_0001.NEF");
        QVERIFY(ProofCache::proofPath(cr2) != ProofCache::proofPath(nef));
    }

    void sidecarPath_isYml() {
        QVERIFY(ProofCache::sidecarPath(m_img).endsWith(".yml"));
    }

    // ── isProofed ─────────────────────────────────────────────────────────────

    void isProofed_falseWhenNoProofFile() {
        ProofCache cache;
        QVERIFY(!cache.isProofed(m_img));
    }

    void isProofed_trueWhenProofExistsAndNoSidecar() {
        QVERIFY(writeJpeg(ProofCache::proofPath(m_img)));
        ProofCache cache;
        QVERIFY(cache.isProofed(m_img));
    }

    void isProofed_trueWhenProofNewerThanSidecar() {
        const QString sidecar = ProofCache::sidecarPath(m_img);
        const QString proof   = ProofCache::proofPath(m_img);

        touchEmpty(sidecar);
        QVERIFY(writeJpeg(proof));

        // Push the proof's mtime 1 second ahead of the sidecar.
        const QDateTime sidecarMtime = QFileInfo(sidecar).lastModified();
        const QDateTime proofMtime   = sidecarMtime.addSecs(1);
        QFile           pf(proof);
        QVERIFY(pf.open(QIODevice::ReadWrite));
        QVERIFY(pf.setFileTime(proofMtime, QFileDevice::FileModificationTime));
        pf.close();

        ProofCache cache;
        QVERIFY(cache.isProofed(m_img));
    }

    void isProofed_falseWhenSidecarNewerThanProof() {
        const QString sidecar = ProofCache::sidecarPath(m_img);
        const QString proof   = ProofCache::proofPath(m_img);

        QVERIFY(writeJpeg(proof));
        touchEmpty(sidecar);

        // Push the sidecar's mtime 1 second ahead of the proof.
        const QDateTime proofMtime   = QFileInfo(proof).lastModified();
        const QDateTime sidecarMtime = proofMtime.addSecs(1);
        QFile           sc(sidecar);
        QVERIFY(sc.open(QIODevice::ReadWrite));
        QVERIFY(sc.setFileTime(sidecarMtime, QFileDevice::FileModificationTime));
        sc.close();

        ProofCache cache;
        QVERIFY(!cache.isProofed(m_img));
    }

    // ── store / proof ─────────────────────────────────────────────────────────

    void store_writesFileToDisk() {
        QImage img(16, 16, QImage::Format_RGB32);
        img.fill(qRgb(255, 0, 0));

        ProofCache cache;
        cache.store(m_img, img);

        QVERIFY(QFileInfo::exists(ProofCache::proofPath(m_img)));
    }

    void proof_returnsNullWhenNotProofed() {
        ProofCache cache;
        QVERIFY(cache.proof(m_img).isNull());
    }

    void proof_returnsValidImageAfterStore() {
        QImage img(16, 16, QImage::Format_RGB32);
        img.fill(qRgb(0, 255, 0));

        ProofCache cache;
        cache.store(m_img, img);

        QVERIFY(!cache.proof(m_img).isNull());
    }

    void proof_coldCache_loadsFromDisk() {
        // Store → clear LRU → proof() must fall through to disk.
        QImage img(8, 8, QImage::Format_RGB32);
        img.fill(qRgb(77, 77, 77));

        ProofCache cache;
        cache.store(m_img, img);
        cache.clear(); // evict in-memory; disk file survives

        QVERIFY(!cache.proof(m_img).isNull());
    }

    void store_updatesExistingLruEntry() {
        // Calling store() twice for the same path exercises lruInsert's
        // "key already in cache — promote and replace" branch.
        QImage img1(8, 8, QImage::Format_RGB32);
        img1.fill(Qt::red);
        QImage img2(8, 8, QImage::Format_RGB32);
        img2.fill(Qt::blue);

        ProofCache cache;
        cache.store(m_img, img1);
        cache.store(m_img, img2); // update branch

        QVERIFY(!cache.proof(m_img).isNull());
    }

    void proof_hotCacheHit_isRejectedWhenDiskFileDisappears() {
        QImage img(8, 8, QImage::Format_RGB32);
        img.fill(qRgb(42, 42, 42));

        ProofCache cache;
        cache.store(m_img, img);
        QFile::remove(ProofCache::proofPath(m_img));

        QVERIFY(cache.proof(m_img).isNull());
    }

    void isProofed_falseWhenSourceIsNewerThanProof() {
        const QString proof = ProofCache::proofPath(m_img);
        QVERIFY(writeJpeg(proof));

        const QDateTime proofMtime  = QFileInfo(proof).lastModified();
        const QDateTime sourceMtime = proofMtime.addSecs(1);
        QFile           source(m_img);
        QVERIFY(source.open(QIODevice::ReadWrite));
        QVERIFY(source.setFileTime(sourceMtime, QFileDevice::FileModificationTime));
        source.close();

        ProofCache cache;
        QVERIFY(!cache.isProofed(m_img));
    }

    void inputFingerprint_changesWithSidecarContents() {
        const QByteArray before = ProofCache::inputFingerprint(m_img);
        QFile            sidecar(ProofCache::sidecarPath(m_img));
        QVERIFY(sidecar.open(QIODevice::WriteOnly));
        QCOMPARE(sidecar.write("effects: changed\n"), qint64(17));
        sidecar.close();
        QVERIFY(ProofCache::inputFingerprint(m_img) != before);
    }

    // ── invalidate ────────────────────────────────────────────────────────────

    void invalidate_deletesDiskFile() {
        QImage img(8, 8, QImage::Format_RGB32);
        img.fill(Qt::blue);

        ProofCache cache;
        cache.store(m_img, img);
        QVERIFY(QFileInfo::exists(ProofCache::proofPath(m_img)));

        cache.invalidate(m_img);
        QVERIFY(!QFileInfo::exists(ProofCache::proofPath(m_img)));
    }

    void invalidate_removesFromLru() {
        QImage img(8, 8, QImage::Format_RGB32);
        img.fill(Qt::yellow);

        ProofCache cache;
        cache.store(m_img, img);
        cache.invalidate(m_img);

        QVERIFY(cache.proof(m_img).isNull());
    }

    void invalidate_noopWhenNotProofed() {
        ProofCache cache;
        // Must not crash when the proof file doesn't exist.
        cache.invalidate(m_img);
    }

    // ── LRU eviction ─────────────────────────────────────────────────────────

    void lru_evictsOldestWhenFull() {
        ProofCache cache;
        const int  OVER = 9; // MAX_LRU is 8; inserting 9 evicts the first
        QString    img0;

        for (int i = 0; i < OVER; ++i) {
            const QString src = m_dir->filePath(QString("lru%1.jpg").arg(i));
            touchEmpty(src);

            QImage img(4, 4, QImage::Format_RGB32);
            img.fill(qRgb(i * 20, i * 20, i * 20));
            cache.store(src, img);
            if (i == 0) img0 = src;
        }

        // img0 was evicted from LRU.  Remove its disk file to force a miss.
        QFile::remove(ProofCache::proofPath(img0));
        QVERIFY(cache.proof(img0).isNull());
    }

    // ── clear ─────────────────────────────────────────────────────────────────

    void clear_emptiesInMemoryCache() {
        QImage img(8, 8, QImage::Format_RGB32);
        img.fill(Qt::cyan);

        ProofCache cache;
        cache.store(m_img, img);
        cache.clear();

        // Remove disk file so proof() can't fall back to it.
        QFile::remove(ProofCache::proofPath(m_img));
        QVERIFY(cache.proof(m_img).isNull());
    }
};

QTEST_GUILESS_MAIN(TestProofCache)
#include "test_proof_cache.moc"
