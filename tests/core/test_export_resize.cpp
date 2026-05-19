#include <QImage>
#include <QTest>

#include "ExportResize.h"

class TestExportResize : public QObject {
    Q_OBJECT

private slots:
    // ── targetSize: pure size logic ────────────────────────────────────────

    void targetSize_modeNone_isIdentity() {
        ExportResize::Params p;
        p.mode = ExportResize::Mode::None;
        QCOMPARE(ExportResize::targetSize(QSize(800, 600), p), QSize(800, 600));
    }

    void targetSize_emptySource_passesThrough() {
        ExportResize::Params p;
        p.mode   = ExportResize::Mode::LongEdge;
        p.pixels = 1024;
        QCOMPARE(ExportResize::targetSize(QSize(0, 0), p), QSize(0, 0));
    }

    void targetSize_longEdge_landscape() {
        ExportResize::Params p;
        p.mode        = ExportResize::Mode::LongEdge;
        p.pixels      = 1000;
        p.dontEnlarge = true;
        // 4000x3000 → long edge 1000 → 1000x750
        QCOMPARE(ExportResize::targetSize(QSize(4000, 3000), p), QSize(1000, 750));
    }

    void targetSize_longEdge_portrait() {
        ExportResize::Params p;
        p.mode   = ExportResize::Mode::LongEdge;
        p.pixels = 1000;
        // 3000x4000 → long edge 1000 → 750x1000
        QCOMPARE(ExportResize::targetSize(QSize(3000, 4000), p), QSize(750, 1000));
    }

    void targetSize_shortEdge() {
        ExportResize::Params p;
        p.mode   = ExportResize::Mode::ShortEdge;
        p.pixels = 600;
        // 4000x3000 → short edge 600 → 800x600
        QCOMPARE(ExportResize::targetSize(QSize(4000, 3000), p), QSize(800, 600));
    }

    void targetSize_width() {
        ExportResize::Params p;
        p.mode   = ExportResize::Mode::Width;
        p.pixels = 1600;
        QCOMPARE(ExportResize::targetSize(QSize(3200, 2000), p), QSize(1600, 1000));
    }

    void targetSize_height() {
        ExportResize::Params p;
        p.mode   = ExportResize::Mode::Height;
        p.pixels = 1000;
        QCOMPARE(ExportResize::targetSize(QSize(3200, 2000), p), QSize(1600, 1000));
    }

    void targetSize_percentage_downscale() {
        ExportResize::Params p;
        p.mode    = ExportResize::Mode::Percentage;
        p.percent = 25;
        QCOMPARE(ExportResize::targetSize(QSize(800, 600), p), QSize(200, 150));
    }

    void targetSize_dontEnlarge_clampsToSource() {
        ExportResize::Params p;
        p.mode        = ExportResize::Mode::LongEdge;
        p.pixels      = 8000;
        p.dontEnlarge = true;
        QCOMPARE(ExportResize::targetSize(QSize(800, 600), p), QSize(800, 600));
    }

    void targetSize_allowEnlarge_upscales() {
        ExportResize::Params p;
        p.mode        = ExportResize::Mode::LongEdge;
        p.pixels      = 1600;
        p.dontEnlarge = false;
        QCOMPARE(ExportResize::targetSize(QSize(800, 600), p), QSize(1600, 1200));
    }

    void targetSize_percentage_dontEnlarge_clamps() {
        ExportResize::Params p;
        p.mode        = ExportResize::Mode::Percentage;
        p.percent     = 200;
        p.dontEnlarge = true;
        QCOMPARE(ExportResize::targetSize(QSize(800, 600), p), QSize(800, 600));
    }

    void targetSize_neverShrinksBelow1px() {
        ExportResize::Params p;
        p.mode        = ExportResize::Mode::Percentage;
        p.percent     = 1;
        p.dontEnlarge = false;
        // 10x4 at 1% would round to 0 — clamp to 1 in each dim.
        const QSize r = ExportResize::targetSize(QSize(10, 4), p);
        QVERIFY(r.width() >= 1);
        QVERIFY(r.height() >= 1);
    }

    // ── apply: end-to-end on a real QImage ────────────────────────────────

    void apply_modeNone_returnsSameSize() {
        QImage src(120, 80, QImage::Format_RGB32);
        src.fill(Qt::red);
        ExportResize::Params p;
        const QImage         dst = ExportResize::apply(src, p);
        QCOMPARE(dst.size(), src.size());
    }

    void apply_longEdge_resizes() {
        QImage src(800, 600, QImage::Format_RGB32);
        src.fill(Qt::blue);
        ExportResize::Params p;
        p.mode           = ExportResize::Mode::LongEdge;
        p.pixels         = 400;
        const QImage dst = ExportResize::apply(src, p);
        QCOMPARE(dst.size(), QSize(400, 300));
    }
};

QTEST_GUILESS_MAIN(TestExportResize)
#include "test_export_resize.moc"
