#include "StackFrameCache.h"

#include <QFile>
#include <QFileInfo>
#include <QTemporaryDir>
#include <QTest>

class TestStackFrameCache : public QObject {
    Q_OBJECT

private slots:
    void storesAndLoadsOnlyMatchingLosslessRender() {
        QTemporaryDir dir;
        QVERIFY(dir.isValid());
        const QString sourcePath = dir.filePath("frame.raw");
        QFile         source(sourcePath);
        QVERIFY(source.open(QIODevice::WriteOnly));
        QCOMPARE(source.write("source"), 6);
        source.close();

        const QByteArray key = StackFrameCache::fingerprint(sourcePath, "settings-a");
        QVERIFY(StackFrameCache::load(sourcePath, key).isNull());
        QVERIFY(!StackFrameCache::store(sourcePath, key, {}));

        QImage render(2, 1, QImage::Format_RGB32);
        auto  *row = reinterpret_cast<QRgb *>(render.scanLine(0));
        row[0]     = qRgb(12, 34, 56);
        row[1]     = qRgb(210, 180, 140);
        QVERIFY(StackFrameCache::store(sourcePath, key, render));
        QVERIFY(QFileInfo::exists(StackFrameCache::renderPath(sourcePath)));
        QVERIFY(QFileInfo::exists(StackFrameCache::fingerprintPath(sourcePath)));
        QCOMPARE(StackFrameCache::load(sourcePath, key), render);
        QVERIFY(StackFrameCache::load(sourcePath, QByteArray("wrong")).isNull());
    }

    void fingerprintChangesWithSettingsAndSourceIdentity() {
        QTemporaryDir dir;
        QVERIFY(dir.isValid());
        const QString sourcePath = dir.filePath("frame.raw");
        QFile         source(sourcePath);
        QVERIFY(source.open(QIODevice::WriteOnly));
        QCOMPARE(source.write("one"), 3);
        source.close();

        const QByteArray original = StackFrameCache::fingerprint(sourcePath, "settings-a");
        QVERIFY(original != StackFrameCache::fingerprint(sourcePath, "settings-b"));

        QVERIFY(source.open(QIODevice::Append));
        QCOMPARE(source.write("-changed"), 8);
        source.close();
        QVERIFY(original != StackFrameCache::fingerprint(sourcePath, "settings-a"));
    }

    void storesFloatBlockCompositesAndTracksDecisions() {
        QTemporaryDir dir;
        QVERIFY(dir.isValid());
        const QString first  = dir.filePath("first.raw");
        const QString second = dir.filePath("second.raw");
        for (const QString &path : {first, second}) {
            QFile file(path);
            QVERIFY(file.open(QIODevice::WriteOnly));
            QVERIFY(file.write("raw") == 3);
        }
        QVector<StackFrame> frames{{first, StackFrameDecision::Include}, {second, StackFrameDecision::Exclude}};
        const QByteArray    key = StackFrameCache::blockFingerprint(frames, "settings", "maximum-v1");
        QImage              image(1, 1, QImage::Format_RGBA32FPx4);
        auto               *pixel = reinterpret_cast<float *>(image.scanLine(0));
        pixel[0]                  = 1.5f;
        pixel[1]                  = 0.4f;
        pixel[2]                  = 0.2f;
        pixel[3]                  = 1.0f;
        QVERIFY(StackFrameCache::storeBlock(dir.path(), "per-channel-maximum", 0, key, image));
        QVERIFY(QFileInfo::exists(StackFrameCache::blockRenderPath(dir.path(), "per-channel-maximum", 0)));
        const QImage loaded = StackFrameCache::loadBlock(dir.path(), "per-channel-maximum", 0, key);
        QVERIFY(!loaded.isNull());
        QCOMPARE(reinterpret_cast<const float *>(loaded.constScanLine(0))[0], 1.5f);
        QVERIFY(StackFrameCache::loadBlock(dir.path(), "per-channel-maximum", 0, "wrong").isNull());

        frames[1].decision = StackFrameDecision::Include;
        QVERIFY(key != StackFrameCache::blockFingerprint(frames, "settings", "maximum-v1"));
        QVERIFY(key != StackFrameCache::blockFingerprint(frames, "different", "maximum-v1"));
    }
};

QTEST_GUILESS_MAIN(TestStackFrameCache)
#include "test_stack_frame_cache.moc"
