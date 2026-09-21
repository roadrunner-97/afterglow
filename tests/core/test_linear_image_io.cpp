#include "LinearImageIO.h"

#include <OpenEXR/ImfChannelList.h>
#include <OpenEXR/ImfFrameBuffer.h>
#include <OpenEXR/ImfHeader.h>
#include <OpenEXR/ImfOutputFile.h>

#include <QFile>
#include <QFileInfo>
#include <QTemporaryDir>
#include <QTest>
#include <cmath>

class TestLinearImageIO : public QObject {
    Q_OBJECT

private slots:
    void identifiesExrPathsCaseInsensitively() {
        QVERIFY(LinearImageIO::isExrPath("master.exr"));
        QVERIFY(LinearImageIO::isExrPath("MASTER.EXR"));
        QVERIFY(!LinearImageIO::isExrPath("master.png"));
    }

    void roundTripsFloatRgbWithoutClipping() {
        QTemporaryDir dir;
        QVERIFY(dir.isValid());
        QImage      image(2, 1, QImage::Format_RGBA32FPx4);
        auto       *pixels   = reinterpret_cast<float *>(image.scanLine(0));
        const float values[] = {1.75f, 0.25f, -0.1f, 1.0f, 0.5f, 2.25f, 0.75f, 1.0f};
        std::copy(std::begin(values), std::end(values), pixels);
        image.setText("color_space", "linear");

        const QString path = dir.filePath("master.exr");
        QString       error;
        QVERIFY2(LinearImageIO::writeExr(path, image, &error), qPrintable(error));
        QVERIFY(QFileInfo::exists(path));
        const QImage loaded = LinearImageIO::readExr(path, &error);
        QVERIFY2(!loaded.isNull(), qPrintable(error));
        QCOMPARE(loaded.format(), QImage::Format_RGBA32FPx4);
        QCOMPARE(loaded.text("color_space"), QString("linear"));
        const auto *actual = reinterpret_cast<const float *>(loaded.constScanLine(0));
        for (size_t i = 0; i < std::size(values); ++i) QVERIFY(std::abs(actual[i] - values[i]) < 1e-6f);
    }

    void reportsInvalidInputs() {
        QTemporaryDir dir;
        QVERIFY(dir.isValid());
        QString error;
        QVERIFY(!LinearImageIO::writeExr(dir.filePath("empty.exr"), {}, &error));
        QVERIFY(!error.isEmpty());
        error.clear();
        QVERIFY(LinearImageIO::readExr(dir.filePath("missing.exr"), &error).isNull());
        QVERIFY(!error.isEmpty());
        error.clear();
        QImage image(1, 1, QImage::Format_RGBA32FPx4);
        QVERIFY(!LinearImageIO::writeExr("/proc/afterglow-test/master.exr", image, &error));
        QVERIFY(!error.isEmpty());
        error.clear();
        QVERIFY(!LinearImageIO::writeExr(dir.path(), image, &error));
        QVERIFY(!error.isEmpty());
    }

    void rejectsExrWithoutRgbChannels() {
        QTemporaryDir dir;
        QVERIFY(dir.isValid());
        const QString path = dir.filePath("luminance.exr");
        {
            Imf::Header header(1, 1);
            header.channels().insert("Y", Imf::Channel(Imf::FLOAT));
            Imf::OutputFile  file(QFile::encodeName(path).constData(), header);
            float            value = 0.5f;
            Imf::FrameBuffer frameBuffer;
            frameBuffer.insert("Y",
                               Imf::Slice(Imf::FLOAT, reinterpret_cast<char *>(&value), sizeof(float), sizeof(float)));
            file.setFrameBuffer(frameBuffer);
            file.writePixels(1);
        }
        QString error;
        QVERIFY(LinearImageIO::readExr(path, &error).isNull());
        QVERIFY(error.contains("RGB"));
    }
};

QTEST_GUILESS_MAIN(TestLinearImageIO)
#include "test_linear_image_io.moc"
