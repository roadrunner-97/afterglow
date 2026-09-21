#include "StackProjectStore.h"

#include <QFile>
#include <QFileInfo>
#include <QTemporaryDir>
#include <QTest>

class TestStackProjectStore : public QObject {
    Q_OBJECT

private slots:
    void roundTripsWorkingProject() {
        QTemporaryDir folder;
        QTemporaryDir external;
        QVERIFY(folder.isValid());
        QVERIFY(external.isValid());
        const QString localPath = folder.filePath("one.raw");
        const QString externalPath = external.filePath("two.raw");
        for (const QString &path : {localPath, externalPath}) {
            QFile file(path);
            QVERIFY(file.open(QIODevice::WriteOnly));
            QVERIFY(file.write("raw") == 3);
        }

        StackProject source;
        source.frames = {{localPath, StackFrameDecision::Include}, {externalPath, StackFrameDecision::Exclude}};
        source.referencePath = localPath;
        source.aggregation.methodId = "per-channel-maximum";
        source.aggregation.parameters.insert("future", 90);
        QString error;
        QVERIFY2(StackProjectStore::save(folder.path(), source, &error), qPrintable(error));
        QVERIFY(QFileInfo::exists(StackProjectStore::manifestPath(folder.path())));
        QVERIFY(StackProjectStore::masterPath(folder.path()).endsWith("/.afterglow/stacks/working/master.exr"));

        StackProject loaded;
        QVERIFY2(StackProjectStore::load(folder.path(), &loaded, &error), qPrintable(error));
        QCOMPARE(loaded.frames.size(), 2);
        QCOMPARE(loaded.frames[0].path, localPath);
        QCOMPARE(loaded.frames[0].decision, StackFrameDecision::Include);
        QCOMPARE(loaded.frames[1].path, externalPath);
        QCOMPARE(loaded.frames[1].decision, StackFrameDecision::Exclude);
        QCOMPARE(loaded.referencePath, localPath);
        QCOMPARE(loaded.aggregation.parameters.value("future").toInt(), 90);
    }

    void handlesMissingAndMalformedProjects() {
        QTemporaryDir folder;
        QVERIFY(folder.isValid());
        StackProject project;
        QString      error;
        QVERIFY(!StackProjectStore::load(folder.path(), &project, &error));
        QVERIFY(error.isEmpty());
        QVERIFY(!StackProjectStore::load(folder.path(), nullptr, &error));
        QVERIFY(!error.isEmpty());
        QVERIFY(!StackProjectStore::save(folder.filePath("missing"), project, &error));
        QVERIFY(!StackProjectStore::save("/proc", project, &error));

        QDir().mkpath(StackProjectStore::projectDirectory(folder.path()));
        QFile file(StackProjectStore::manifestPath(folder.path()));
        QVERIFY(file.open(QIODevice::WriteOnly));
        QVERIFY(file.write("not json") > 0);
        file.close();
        QVERIFY(!StackProjectStore::load(folder.path(), &project, &error));
        QVERIFY(!error.isEmpty());

        QVERIFY(file.open(QIODevice::WriteOnly | QIODevice::Truncate));
        QVERIFY(file.write("{\"schema\": 99}") > 0);
        file.close();
        QVERIFY(!StackProjectStore::load(folder.path(), &project, &error));
        QVERIFY(error.contains("unsupported"));
    }

    void reportsManifestOpenFailures() {
        QTemporaryDir folder;
        QVERIFY(folder.isValid());
        QDir().mkpath(StackProjectStore::manifestPath(folder.path()));
        StackProject project;
        QString      error;
        QVERIFY(!StackProjectStore::load(folder.path(), &project, &error));
        QVERIFY(!error.isEmpty());

        QTemporaryDir second;
        QVERIFY(second.isValid());
        QDir().mkpath(StackProjectStore::manifestPath(second.path()));
        QVERIFY(!StackProjectStore::save(second.path(), project, &error));
        QVERIFY(!error.isEmpty());
    }
};

QTEST_GUILESS_MAIN(TestStackProjectStore)
#include "test_stack_project_store.moc"
