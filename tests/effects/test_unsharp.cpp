#include <QTest>
#include <QSignalSpy>
#include <QSlider>
#include <QWidget>
#include "UnsharpEffect.h"
#include "GpuDeviceRegistry.h"
#include "ImageHelpers.h"
#include "ParamSlider.h"

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

    // Non-square (tall) image: unsharp mask chains a H then V blur with
    // differing extents.  On a 64x128 top/bottom split, the dark side near
    // the horizontal edge should be darkened by sharpening, the bright side
    // brightened.  Output dimensions must match input.
    void nonSquare_edgeSplit_sharpensAcrossHorizontalEdge() {
        if (!m_hasGpu) QSKIP("No GPU");
        UnsharpEffect e;
        // Top half dark (100), bottom half bright (200).
        QImage input(64, 128, QImage::Format_RGB32);
        for (int y = 0; y < 128; ++y) {
            auto* row = reinterpret_cast<QRgb*>(input.scanLine(y));
            for (int x = 0; x < 64; ++x)
                row[x] = (y < 64) ? qRgb(100, 100, 100) : qRgb(200, 200, 200);
        }
        QMap<QString, QVariant> params;
        params["amount"]    = 2.0;
        params["radius"]    = 8;
        params["threshold"] = 3;
        QImage out = e.processImage(input, params);
        QVERIFY(!out.isNull());
        QCOMPARE(out.width(),  64);
        QCOMPARE(out.height(), 128);
        // 2 px above the edge (y=62) → dark side gets darker.
        QVERIFY(pixelR(out, 32, 62) < 100);
        // 2 px below the edge (y=66) → bright side gets brighter.
        QVERIFY(pixelR(out, 32, 66) > 200);
    }

    void meta_nonEmpty() {
        UnsharpEffect e;
        QVERIFY(!e.getName().isEmpty());
        QVERIFY(!e.getDescription().isEmpty());
        QVERIFY(!e.getVersion().isEmpty());
        QVERIFY(e.initialize());
    }

    void defaultParameters_keys() {
        UnsharpEffect e;
        auto params = e.getParameters();
        QVERIFY(params.contains("amount"));
        QVERIFY(params.contains("radius"));
        QVERIFY(params.contains("threshold"));
        QCOMPARE(params["amount"].toDouble(),    1.0);
        QCOMPARE(params["radius"].toInt(),       2);
        QCOMPARE(params["threshold"].toInt(),    3);
    }

    // Solid colour 16-bit: blurred == original, so diff < threshold → unchanged.
    void solidColour_16bit() {
        if (!m_hasGpu) QSKIP("No GPU");
        UnsharpEffect e;
        QImage input = makeSolid16bit(64, 64, 128, 128, 128);
        QMap<QString, QVariant> params;
        params["amount"]    = 1.0;
        params["radius"]    = 2;
        params["threshold"] = 3;
        QImage out = e.processImage(input, params);
        QVERIFY(!out.isNull());
    }

    void createControlsWidget_constructsAndCaches() {
        UnsharpEffect e;
        QWidget* w = e.createControlsWidget();
        QVERIFY(w != nullptr);
        QVERIFY(e.createControlsWidget() == w);
    }

    // Fire signal lambdas for all three ParamSliders (amount, radius, threshold).
    void connectSlider_signals_coverLambdaBodies() {
        UnsharpEffect e;
        QWidget* w = e.createControlsWidget();
        QVERIFY(w);

        QSignalSpy spyChanged(&e, &PhotoEditorEffect::parametersChanged);
        QSignalSpy spyLive(&e, &PhotoEditorEffect::liveParametersChanged);

        auto sliders = w->findChildren<ParamSlider*>();
        QVERIFY(sliders.size() >= 3);

        for (auto* ps : sliders) {
            auto* qs = ps->findChild<QSlider*>();
            QVERIFY(qs);
            qs->setValue(qs->value() + 1);
            QMetaObject::invokeMethod(qs, "sliderReleased");
        }

        QVERIFY(spyChanged.count() >= 3);
        QVERIFY(spyLive.count() >= 3);
    }

    void destructor_heapAllocated_doesNotCrash() {
        auto* e = new UnsharpEffect();
        e->createControlsWidget();
        delete e;
    }
};

QTEST_MAIN(TestUnsharp)
#include "test_unsharp.moc"
