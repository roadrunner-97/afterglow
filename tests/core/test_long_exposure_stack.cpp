#include "LongExposureStack.h"

#include <QTest>

class TestLongExposureStack : public QObject {
    Q_OBJECT

private slots:
    void firstFrameSeedsAccumulator() {
        QImage frame(2, 1, QImage::Format_RGB32);
        auto  *row = reinterpret_cast<QRgb *>(frame.scanLine(0));
        row[0]     = qRgb(10, 20, 30);
        row[1]     = qRgb(40, 50, 60);

        QImage accumulator;
        QVERIFY(LongExposureStack::blendLighten(&accumulator, frame));
        QCOMPARE(accumulator, frame);
    }

    void keepsBrightestValuePerChannel() {
        QImage accumulator(1, 1, QImage::Format_RGB32);
        QImage frame(1, 1, QImage::Format_RGB32);
        reinterpret_cast<QRgb *>(accumulator.scanLine(0))[0] = qRgb(220, 30, 100);
        reinterpret_cast<QRgb *>(frame.scanLine(0))[0]       = qRgb(20, 240, 90);

        QVERIFY(LongExposureStack::blendLighten(&accumulator, frame));
        const QRgb pixel = reinterpret_cast<const QRgb *>(accumulator.constScanLine(0))[0];
        QCOMPARE(qRed(pixel), 220);
        QCOMPARE(qGreen(pixel), 240);
        QCOMPARE(qBlue(pixel), 100);
    }

    void rejectsMismatchedDimensionsWithoutChangingResult() {
        QImage accumulator(2, 2, QImage::Format_RGB32);
        accumulator.fill(Qt::red);
        const QImage before = accumulator.copy();
        QImage       frame(3, 2, QImage::Format_RGB32);
        frame.fill(Qt::blue);
        QString error;

        QVERIFY(!LongExposureStack::blendLighten(&accumulator, frame, &error));
        QVERIFY(error.contains("3 × 2"));
        QCOMPARE(accumulator, before);
    }

    void rejectsNullFrames() {
        QImage  accumulator;
        QString error;
        QVERIFY(!LongExposureStack::blendLighten(&accumulator, {}, &error));
        QVERIFY(!error.isEmpty());
    }

    void rejectsMissingAccumulator() {
        QImage frame(1, 1, QImage::Format_RGB32);
        frame.fill(Qt::white);
        QString error;
        QVERIFY(!LongExposureStack::blendLighten(nullptr, frame, &error));
        QVERIFY(!error.isEmpty());
    }

    void normalizesAnExistingAccumulatorToRgb32() {
        QImage accumulator(1, 1, QImage::Format_RGB888);
        accumulator.fill(qRgb(5, 10, 15));
        QImage frame(1, 1, QImage::Format_RGB32);
        frame.fill(qRgb(20, 5, 10));

        QVERIFY(LongExposureStack::blendLighten(&accumulator, frame));
        QCOMPARE(accumulator.format(), QImage::Format_RGB32);
        const QRgb pixel = reinterpret_cast<const QRgb *>(accumulator.constScanLine(0))[0];
        QCOMPARE(pixel, qRgb(20, 10, 15));
    }
};

QTEST_APPLESS_MAIN(TestLongExposureStack)
#include "test_long_exposure_stack.moc"
