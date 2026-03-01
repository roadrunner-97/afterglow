#include <QTest>
#include "UnsharpEffect.h"
#include "GpuDeviceRegistry.h"
#include "ImageHelpers.h"

class TestUnsharp : public QObject {
    Q_OBJECT

private:
    bool m_hasGpu = false;

private slots:
    void initTestCase() {
        GpuDeviceRegistry::instance().enumerate();
        if (GpuDeviceRegistry::instance().count() == 0)
            QSKIP("No OpenCL device found — skipping GPU effect tests");
        GpuDeviceRegistry::instance().setDevice(0);
        m_hasGpu = true;
    }

    void nullImage_passThrough() {
        UnsharpEffect e;
        QVERIFY(e.processImage(QImage(), {}).isNull());
    }

    // radius=0 → early return (code: if(amount==0||radius==0) return image).
    void zeroRadius_isIdentity() {
        UnsharpEffect e;
        QImage input = makeSolid(32, 32, 128, 100, 80);
        QMap<QString, QVariant> params;
        params["amount"]    = 2.0;
        params["radius"]    = 0;
        params["threshold"] = 3;
        QImage out = e.processImage(input, params);
        QVERIFY(!out.isNull());
        QVERIFY(allPixels(out, [](QRgb px) {
            return qRed(px) == 128 && qGreen(px) == 100 && qBlue(px) == 80;
        }));
    }

    // amount=0 → early return.
    void zeroAmount_isIdentity() {
        UnsharpEffect e;
        QImage input = makeSolid(32, 32, 128, 100, 80);
        QMap<QString, QVariant> params;
        params["amount"]    = 0.0;
        params["radius"]    = 5;
        params["threshold"] = 3;
        QImage out = e.processImage(input, params);
        QVERIFY(!out.isNull());
        QVERIFY(allPixels(out, [](QRgb px) {
            return qRed(px) == 128 && qGreen(px) == 100 && qBlue(px) == 80;
        }));
    }

    // Solid colour: blurred == original → all channel diffs = 0 < threshold →
    // the unsharpCombine kernel passes through the original unchanged.
    void solidColour_unsharpUnchanged() {
        if (!m_hasGpu) QSKIP("No GPU");
        UnsharpEffect e;
        QImage input = makeSolid(64, 64, 128, 128, 128);
        QMap<QString, QVariant> params;
        params["amount"]    = 3.0;
        params["radius"]    = 10;
        params["threshold"] = 3;
        QImage out = e.processImage(input, params);
        QVERIFY(!out.isNull());
        QVERIFY(allPixels(out, [](QRgb px) { return qRed(px) == 128; }));
    }

    // Left/right split image: unsharp masking increases local contrast at edges.
    // Left half: 100 (dark side). Right half: 200 (bright side).
    // After sharpening with amount=2, radius=5:
    //   - Pixel near the edge on the dark side should get DARKER (< 100):
    //     original=100, blurred>100 (bright side bleeds in) → diff negative →
    //     output = 100 + 2*(negative) < 100.
    //   - Pixel near the edge on the bright side should get BRIGHTER (> 200).
    //
    // We test columns that are close to the split (a few pixels in from each side)
    // to ensure the blur kernel extends across the boundary.
    void edgeSplit_darkSideGetsDarker() {
        if (!m_hasGpu) QSKIP("No GPU");
        UnsharpEffect e;
        QImage input = makeSplit(128, 64, 100, 100, 100, 200, 200, 200);
        QMap<QString, QVariant> params;
        params["amount"]    = 2.0;
        params["radius"]    = 8;
        params["threshold"] = 3;
        QImage out = e.processImage(input, params);
        QVERIFY(!out.isNull());
        // Check a pixel 2 columns left of the boundary (x=62, y=32).
        // It lies on the dark side and should be darkened.
        QVERIFY(pixelR(out, 62, 32) < 100);
    }

    void edgeSplit_brightSideGetsBrighter() {
        if (!m_hasGpu) QSKIP("No GPU");
        UnsharpEffect e;
        QImage input = makeSplit(128, 64, 100, 100, 100, 200, 200, 200);
        QMap<QString, QVariant> params;
        params["amount"]    = 2.0;
        params["radius"]    = 8;
        params["threshold"] = 3;
        QImage out = e.processImage(input, params);
        QVERIFY(!out.isNull());
        // 2 columns right of the boundary (x=66, y=32) → bright side, gets brighter.
        QVERIFY(pixelR(out, 66, 32) > 200);
    }

    // Pixels far from the edge should be unchanged (blurred ≈ original there).
    void edgeSplit_farFromEdge_unchanged() {
        if (!m_hasGpu) QSKIP("No GPU");
        UnsharpEffect e;
        QImage input = makeSplit(128, 64, 100, 100, 100, 200, 200, 200);
        QMap<QString, QVariant> params;
        params["amount"]    = 2.0;
        params["radius"]    = 8;
        params["threshold"] = 3;
        QImage out = e.processImage(input, params);
        QVERIFY(!out.isNull());
        // Deep in the dark half (x=10) and deep in the bright half (x=118) are
        // far enough from the edge that blur ≈ original and diff < threshold.
        QCOMPARE(pixelR(out, 10,  32), 100);
        QCOMPARE(pixelR(out, 118, 32), 200);
    }
};

QTEST_GUILESS_MAIN(TestUnsharp)
#include "test_unsharp.moc"
