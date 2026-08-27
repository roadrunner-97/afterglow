#include <QDir>
#include <QFile>
#include <QFileInfo>
#include <QTemporaryDir>
#include <QTest>

#include "CachePurger.h"

static void writeFile(const QString &path, const QByteArray &contents = "data") {
    QDir().mkpath(QFileInfo(path).absolutePath());
    QFile file(path);
    if (!file.open(QIODevice::WriteOnly) || file.write(contents) != contents.size())
        qFatal("Could not write test file: %s", qPrintable(path));
}

class TestCachePurger : public QObject {
    Q_OBJECT

private slots:
    void removesOnlyGeneratedPhotoCaches() {
        QTemporaryDir dir;
        QVERIFY(dir.isValid());
        writeFile(dir.filePath(".afterglow-thumbs/a.jpg"));
        writeFile(dir.filePath(".afterglow/proofs/a.raw.jpg"));
        writeFile(dir.filePath("photo.yml"), "settings");
        writeFile(dir.filePath("photo.history.yml"), "history");
        writeFile(dir.filePath(".afterglow-catalog.json"), "catalog");
        writeFile(dir.filePath(".afterglow/keep.txt"), "future-data");
        writeFile(dir.filePath("photo.raw"), "source");

        const CachePurger::Result result = CachePurger::purgePhotoCaches(dir.path());

        QVERIFY(result.success);
        QCOMPARE(result.filesRemoved, 2);
        QVERIFY(!QFileInfo::exists(dir.filePath(".afterglow-thumbs")));
        QVERIFY(!QFileInfo::exists(dir.filePath(".afterglow/proofs")));
        QVERIFY(QFileInfo::exists(dir.filePath("photo.yml")));
        QVERIFY(QFileInfo::exists(dir.filePath("photo.history.yml")));
        QVERIFY(QFileInfo::exists(dir.filePath(".afterglow-catalog.json")));
        QVERIFY(QFileInfo::exists(dir.filePath(".afterglow/keep.txt")));
        QVERIFY(QFileInfo::exists(dir.filePath("photo.raw")));
    }

    void succeedsWhenCachesDoNotExist() {
        QTemporaryDir dir;
        QVERIFY(dir.isValid());
        const CachePurger::Result result = CachePurger::purgePhotoCaches(dir.path());
        QVERIFY(result.success);
        QCOMPARE(result.filesRemoved, 0);
    }

    void rejectsMissingFolder() {
        const CachePurger::Result result = CachePurger::purgePhotoCaches("/path/that/does/not/exist");
        QVERIFY(!result.success);
        QVERIFY(!result.error.isEmpty());
    }

    void removesCacheSymlinkWithoutFollowingIt() {
        QTemporaryDir dir;
        QTemporaryDir external;
        QVERIFY(dir.isValid());
        QVERIFY(external.isValid());
        const QString externalFile = external.filePath("must-survive.jpg");
        writeFile(externalFile);
        QVERIFY(QFile::link(external.path(), dir.filePath(".afterglow-thumbs")));

        const CachePurger::Result result = CachePurger::purgePhotoCaches(dir.path());
        QVERIFY(result.success);
        QCOMPARE(result.filesRemoved, 1);
        QVERIFY(!QFileInfo::exists(dir.filePath(".afterglow-thumbs")));
        QVERIFY(QFileInfo::exists(externalFile));
    }
};

QTEST_GUILESS_MAIN(TestCachePurger)
#include "test_cache_purger.moc"
