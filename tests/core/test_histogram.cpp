#include <QTest>
#include <QImage>
#include <cstdint>
#include <numeric>
#include "Histogram.h"

class TestHistogram : public QObject {
    Q_OBJECT

private slots:
    void nullImage_returnsEmpty() {
        QCOMPARE(computeLuminanceHistogram(QImage()).size(), size_t(0));
    }

    void size_is256() {
        QImage img(4, 4, QImage::Format_RGB32);
        img.fill(qRgb(0, 0, 0));
        QCOMPARE(computeLuminanceHistogram(img).size(), size_t(256));
    }

    // Every pixel pure black → every count falls into bin 0.
    void allBlack_fillsBinZero() {
        QImage img(32, 32, QImage::Format_RGB32);
        img.fill(qRgb(0, 0, 0));
        auto bins = computeLuminanceHistogram(img);
        QCOMPARE(bins[0], uint32_t(32 * 32));
        for (int i = 1; i < 256; ++i) QCOMPARE(bins[i], uint32_t(0));
    }

    // Every pixel pure white → falls into the last bin.
    void allWhite_fillsLastBin() {
        QImage img(32, 32, QImage::Format_RGB32);
        img.fill(qRgb(255, 255, 255));
        auto bins = computeLuminanceHistogram(img);
        QCOMPARE(bins[255], uint32_t(32 * 32));
        uint32_t others = 0;
        for (int i = 0; i < 255; ++i) others += bins[i];
        QCOMPARE(others, uint32_t(0));
    }

    // Total bin count equals pixel count regardless of content.
    void totalCount_matchesPixelCount() {
        QImage img(100, 50, QImage::Format_RGB32);
        for (int y = 0; y < img.height(); ++y) {
            auto* row = reinterpret_cast<QRgb*>(img.scanLine(y));
            for (int x = 0; x < img.width(); ++x) {
                int v = (x + y) * 255 / (img.width() + img.height() - 2);
                row[x] = qRgb(v, v, v);
            }
        }
        auto bins = computeLuminanceHistogram(img);
        uint64_t total = std::accumulate(bins.begin(), bins.end(), uint64_t(0));
        QCOMPARE(total, uint64_t(img.width() * img.height()));
    }

    // Mid-grey RGB(128,128,128) has perceptual L ≈ 128/255 ≈ 0.502 →
    // bin ≈ 128.  Allow a small window (±2) for rounding around the gamma
    // inverse round-trip.
    void midGrey_lainsMidBin() {
        QImage img(16, 16, QImage::Format_RGB32);
        img.fill(qRgb(128, 128, 128));
        auto bins = computeLuminanceHistogram(img);

        int maxBin = 0;
        for (int i = 0; i < 256; ++i)
            if (bins[i] > bins[maxBin]) maxBin = i;

        QVERIFY2(maxBin >= 126 && maxBin <= 130,
                 qPrintable(QString("maxBin=%1 expected ~128").arg(maxBin)));
        QCOMPARE(bins[maxBin], uint32_t(16 * 16));
    }

    // 16-bit path: all-black RGBX64 → bin 0.
    void sixteenBit_allBlack_fillsBinZero() {
        QImage img(16, 16, QImage::Format_RGBX64);
        img.fill(QColor(0, 0, 0));
        auto bins = computeLuminanceHistogram(img);
        QCOMPARE(bins[0], uint32_t(16 * 16));
    }

    // 16-bit path: all-white RGBX64 → last bin.
    void sixteenBit_allWhite_fillsLastBin() {
        QImage img(16, 16, QImage::Format_RGBX64);
        img.fill(QColor(255, 255, 255));
        auto bins = computeLuminanceHistogram(img);
        QCOMPARE(bins[255], uint32_t(16 * 16));
    }

    // Unsupported format: ARGB32_Premultiplied should still produce a valid
    // histogram via internal format conversion.
    void unsupportedFormat_convertsAndWorks() {
        QImage img(8, 8, QImage::Format_ARGB32_Premultiplied);
        img.fill(qRgb(0, 0, 0));
        auto bins = computeLuminanceHistogram(img);
        QCOMPARE(bins.size(), size_t(256));
        QCOMPARE(bins[0], uint32_t(8 * 8));
    }
};

QTEST_APPLESS_MAIN(TestHistogram)
#include "test_histogram.moc"
