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
};

QTEST_GUILESS_MAIN(TestStackFrameCache)
#include "test_stack_frame_cache.moc"
