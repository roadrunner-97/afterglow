#include <QTest>
#include <QDir>
#include <QFile>
#include <QSet>
#include <QTemporaryDir>
#include "ExportOptions.h"
#include "ExportPath.h"

class TestExportPath : public QObject {
    Q_OBJECT

private slots:
    // ── resolvePattern ──────────────────────────────────────────────────────

    void resolvePattern_substitutesNameToken() {
        QCOMPARE(ExportPath::resolvePattern("{name}", "/a/b/photo.cr2", 1),
                 QString("photo"));
    }

    void resolvePattern_padsBatchIndexToThreeDigits() {
        QCOMPARE(ExportPath::resolvePattern("{n}",  "/x.jpg", 7),   QString("007"));
        QCOMPARE(ExportPath::resolvePattern("{n}",  "/x.jpg", 42),  QString("042"));
        // Past 999 the field grows naturally — pad is a *minimum*, not a cap.
        QCOMPARE(ExportPath::resolvePattern("{n}",  "/x.jpg", 1234), QString("1234"));
    }

    void resolvePattern_substitutesDateToken() {
        const QDate d(2026, 4, 29);
        QCOMPARE(ExportPath::resolvePattern("{date}", "/x.jpg", 1, d),
                 QString("2026-04-29"));
    }

    void resolvePattern_combinesTokens() {
        const QDate d(2026, 1, 2);
        QCOMPARE(ExportPath::resolvePattern("{date}_{name}_{n}", "/sun.dng", 12, d),
                 QString("2026-01-02_sun_012"));
    }

    void resolvePattern_unknownTokensPassThroughVerbatim() {
        QCOMPARE(ExportPath::resolvePattern("export-{name}-{whatever}", "/x.jpg", 1),
                 QString("export-x-{whatever}"));
    }

    void resolvePattern_preservesMultiDotBaseName() {
        // completeBaseName() drops only the last dot suffix.
        QCOMPARE(ExportPath::resolvePattern("{name}", "/a/b/IMG_1234.HDR.dng", 1),
                 QString("IMG_1234.HDR"));
    }

    void resolvePattern_handlesEmptySource() {
        QCOMPARE(ExportPath::resolvePattern("{name}-{n}", "", 3),
                 QString("-003"));
    }

    void resolvePattern_literalPatternIsPassedThrough() {
        QCOMPARE(ExportPath::resolvePattern("plain-name", "/x.jpg", 1),
                 QString("plain-name"));
    }

    // ── chooseDestination ──────────────────────────────────────────────────

    void chooseDestination_overwriteReturnsCandidateEvenIfExists() {
        ExportOptions::Options opts;
        opts.destinationDir   = "/out";
        opts.filenamePattern  = "{name}";
        opts.format           = ExportOptions::Format::JPEG;
        opts.onConflict       = ExportOptions::OverwritePolicy::Overwrite;
        QCOMPARE(ExportPath::chooseDestination(
            opts, "/x/photo.cr2", 1,
            [](const QString&) { return true; }),
            QString("/out/photo.jpg"));
    }

    void chooseDestination_skipReturnsEmptyOnConflict() {
        ExportOptions::Options opts;
        opts.destinationDir   = "/out";
        opts.format           = ExportOptions::Format::PNG;
        opts.onConflict       = ExportOptions::OverwritePolicy::Skip;
        QVERIFY(ExportPath::chooseDestination(
            opts, "/x/photo.cr2", 1,
            [](const QString&) { return true; }).isEmpty());
    }

    void chooseDestination_skipReturnsCandidateWhenFree() {
        ExportOptions::Options opts;
        opts.destinationDir   = "/out";
        opts.format           = ExportOptions::Format::PNG;
        opts.onConflict       = ExportOptions::OverwritePolicy::Skip;
        QCOMPARE(ExportPath::chooseDestination(
            opts, "/x/photo.cr2", 1,
            [](const QString&) { return false; }),
            QString("/out/photo.png"));
    }

    void chooseDestination_appendSuffixReturnsCandidateWhenFree() {
        ExportOptions::Options opts;
        opts.destinationDir   = "/out";
        opts.format           = ExportOptions::Format::JPEG;
        opts.onConflict       = ExportOptions::OverwritePolicy::AppendSuffix;
        QCOMPARE(ExportPath::chooseDestination(
            opts, "/x/photo.cr2", 1,
            [](const QString&) { return false; }),
            QString("/out/photo.jpg"));
    }

    void chooseDestination_appendSuffixWalksUntilFree() {
        const QSet<QString> taken {
            "/out/photo.tif", "/out/photo_1.tif", "/out/photo_2.tif",
        };
        ExportOptions::Options opts;
        opts.destinationDir   = "/out";
        opts.format           = ExportOptions::Format::TIFF;
        opts.onConflict       = ExportOptions::OverwritePolicy::AppendSuffix;
        QCOMPARE(ExportPath::chooseDestination(
            opts, "/x/photo.cr2", 1,
            [&](const QString& q) { return taken.contains(q); }),
            QString("/out/photo_3.tif"));
    }

    void chooseDestination_appendSuffixGivesUpAfter9999() {
        // Predicate that swallows everything → no free slot exists.
        ExportOptions::Options opts;
        opts.destinationDir   = "/out";
        opts.format           = ExportOptions::Format::JPEG;
        opts.onConflict       = ExportOptions::OverwritePolicy::AppendSuffix;
        QVERIFY(ExportPath::chooseDestination(
            opts, "/x/photo.cr2", 1,
            [](const QString&) { return true; }).isEmpty());
    }

    void chooseDestination_defaultPredicateUsesRealFilesystem() {
        // Exercises the QFile::exists default branch — no stub injected.
        QTemporaryDir tmp;
        QVERIFY(tmp.isValid());
        ExportOptions::Options opts;
        opts.destinationDir   = tmp.path();
        opts.filenamePattern  = "{name}";
        opts.format           = ExportOptions::Format::JPEG;
        opts.onConflict       = ExportOptions::OverwritePolicy::AppendSuffix;

        // First call: no collision → bare name.
        const QString p1 = ExportPath::chooseDestination(opts, "/x/photo.cr2", 1);
        QCOMPARE(p1, QDir(tmp.path()).filePath("photo.jpg"));

        // Touch the file, then call again: must walk to _1.
        QFile f(p1);
        QVERIFY(f.open(QIODevice::WriteOnly));
        f.close();
        const QString p2 = ExportPath::chooseDestination(opts, "/x/photo.cr2", 1);
        QCOMPARE(p2, QDir(tmp.path()).filePath("photo_1.jpg"));
    }

    void chooseDestination_honoursDestinationDir() {
        ExportOptions::Options opts;
        opts.destinationDir   = "/some/where/else";
        opts.filenamePattern  = "out_{n}";
        opts.format           = ExportOptions::Format::PNG;
        opts.onConflict       = ExportOptions::OverwritePolicy::Overwrite;
        QCOMPARE(ExportPath::chooseDestination(
            opts, "/src/img.dng", 5,
            [](const QString&) { return false; }),
            QString("/some/where/else/out_005.png"));
    }

    // ── ExportOptions helpers ──────────────────────────────────────────────

    void extensionFor_coversAllFormats() {
        QCOMPARE(ExportOptions::extensionFor(ExportOptions::Format::JPEG), QString("jpg"));
        QCOMPARE(ExportOptions::extensionFor(ExportOptions::Format::PNG),  QString("png"));
        QCOMPARE(ExportOptions::extensionFor(ExportOptions::Format::TIFF), QString("tif"));
    }

    void qImageFormatHint_coversAllFormats() {
        QCOMPARE(QString(ExportOptions::qImageFormatHint(ExportOptions::Format::JPEG)),
                 QString("JPEG"));
        QCOMPARE(QString(ExportOptions::qImageFormatHint(ExportOptions::Format::PNG)),
                 QString("PNG"));
        QCOMPARE(QString(ExportOptions::qImageFormatHint(ExportOptions::Format::TIFF)),
                 QString("TIFF"));
    }

    void qualityFor_onlyAppliesToJpeg() {
        ExportOptions::Options opts;
        opts.jpegQuality = 70;
        opts.format = ExportOptions::Format::JPEG;
        QCOMPARE(ExportOptions::qualityFor(opts), 70);
        opts.format = ExportOptions::Format::PNG;
        QCOMPARE(ExportOptions::qualityFor(opts), -1);
        opts.format = ExportOptions::Format::TIFF;
        QCOMPARE(ExportOptions::qualityFor(opts), -1);
    }
};

QTEST_GUILESS_MAIN(TestExportPath)
#include "test_export_path.moc"
