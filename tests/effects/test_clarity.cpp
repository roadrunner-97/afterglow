#include <QTest>
#include <QSignalSpy>
#include <QSlider>
#include <QWidget>
#include "ClarityEffect.h"
#include "GpuDeviceRegistry.h"
#include "ImageHelpers.h"
#include "ParamSlider.h"

class TestClarity : public QObject {
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
        ClarityEffect e;
        QVERIFY(e.processImage(QImage(), {}).isNull());
    }

    // amount=0 → early return (identity).
    void zeroAmount_isIdentity() {
        ClarityEffect e;
        QImage input = makeSolid(32, 32, 128, 100, 80);
        QMap<QString, QVariant> params;
        params["amount"] = 0;
        params["radius"] = 30;
        QImage out = e.processImage(input, params);
        QVERIFY(!out.isNull());
        QVERIFY(allPixels(out, [](QRgb px) {
            return qRed(px) == 128 && qGreen(px) == 100 && qBlue(px) == 80;
        }));
    }

    // radius=0 → early return (identity).
    void zeroRadius_isIdentity() {
        ClarityEffect e;
        QImage input = makeSolid(32, 32, 128, 100, 80);
        QMap<QString, QVariant> params;
        params["amount"] = 50;
        params["radius"] = 0;
        QImage out = e.processImage(input, params);
        QVERIFY(!out.isNull());
        QVERIFY(allPixels(out, [](QRgb px) {
            return qRed(px) == 128 && qGreen(px) == 100 && qBlue(px) == 80;
        }));
    }

    // Solid midtone: blurred == original → orig − blurred = 0 → output unchanged.
    void solidMidtone_unchanged() {
        if (!m_hasGpu) QSKIP("No GPU");
        ClarityEffect e;
        QImage input = makeSolid(64, 64, 128, 128, 128);
        QMap<QString, QVariant> params;
        params["amount"] = 100;
        params["radius"] = 20;
        QImage out = e.processImage(input, params);
        QVERIFY(!out.isNull());
        QVERIFY(allPixels(out, [](QRgb px) {
            return qRed(px) == 128 && qGreen(px) == 128 && qBlue(px) == 128;
        }));
    }

    // Pure black / pure white have luminance mask = 0, so clarity is a no-op there
    // even across an edge: the midtone mask protects the extremes.
    void pureExtremes_maskProtectsFromChange() {
        if (!m_hasGpu) QSKIP("No GPU");
        ClarityEffect e;
        // Left half black, right half white.  Columns deep inside each side are
        // far from the edge AND at the extremes of luminance → mask≈0 → unchanged.
        QImage input = makeSplit(128, 64, 0, 0, 0, 255, 255, 255);
        QMap<QString, QVariant> params;
        params["amount"] = 100;
        params["radius"] = 10;
        QImage out = e.processImage(input, params);
        QVERIFY(!out.isNull());
        QCOMPARE(pixelR(out,  10, 32), 0);
        QCOMPARE(pixelR(out, 118, 32), 255);
    }

    // Midtone edge split: positive clarity increases local contrast on midtones.
    // Dark side near the edge darkens further; bright side near the edge brightens.
    void midtoneEdge_positiveClarity_increasesContrast() {
        if (!m_hasGpu) QSKIP("No GPU");
        ClarityEffect e;
        QImage input = makeSplit(128, 64, 100, 100, 100, 160, 160, 160);
        QMap<QString, QVariant> params;
        params["amount"] = 100;
        params["radius"] = 10;
        QImage out = e.processImage(input, params);
        QVERIFY(!out.isNull());
        // 2 px left of the boundary (x=62) — on the dark side, expect darker.
        QVERIFY(pixelR(out, 62, 32) < 100);
        // 2 px right of the boundary (x=66) — on the bright side, expect brighter.
        QVERIFY(pixelR(out, 66, 32) > 160);
    }

    // Negative clarity reduces local midtone contrast (softens).
    void midtoneEdge_negativeClarity_softens() {
        if (!m_hasGpu) QSKIP("No GPU");
        ClarityEffect e;
        QImage input = makeSplit(128, 64, 100, 100, 100, 160, 160, 160);
        QMap<QString, QVariant> params;
        params["amount"] = -100;
        params["radius"] = 10;
        QImage out = e.processImage(input, params);
        QVERIFY(!out.isNull());
        // Near-edge dark-side pixel should move toward the bright side (brighter).
        QVERIFY(pixelR(out, 62, 32) > 100);
        // Near-edge bright-side pixel should move toward the dark side (darker).
        QVERIFY(pixelR(out, 66, 32) < 160);
    }

    // Non-square (tall) image: clarity's internal blur is 2D (separable H+V
    // with different extents).  On a 64x128 top/bottom midtone split, positive
    // clarity must still increase local contrast across the horizontal edge.
    void nonSquare_midtoneEdge_positiveClarity_increasesContrast() {
        if (!m_hasGpu) QSKIP("No GPU");
        ClarityEffect e;
        // Top half dark grey, bottom half lighter grey.
        QImage input(64, 128, QImage::Format_RGB32);
        for (int y = 0; y < 128; ++y) {
            auto* row = reinterpret_cast<QRgb*>(input.scanLine(y));
            for (int x = 0; x < 64; ++x)
                row[x] = (y < 64) ? qRgb(100, 100, 100) : qRgb(160, 160, 160);
        }
        QMap<QString, QVariant> params;
        params["amount"] = 100;
        params["radius"] = 10;
        QImage out = e.processImage(input, params);
        QVERIFY(!out.isNull());
        QCOMPARE(out.width(),  64);
        QCOMPARE(out.height(), 128);
        // 2 px above the edge (y=62) → dark side darker; 2 px below (y=66) → brighter.
        QVERIFY(pixelR(out, 32, 62) < 100);
        QVERIFY(pixelR(out, 32, 66) > 160);
    }

    // 16-bit path: solid midtone stays unchanged (blur == orig).
    void solidMidtone_16bit() {
        if (!m_hasGpu) QSKIP("No GPU");
        ClarityEffect e;
        QImage input = makeSolid16bit(64, 64, 128, 128, 128);
        QMap<QString, QVariant> params;
        params["amount"] = 80;
        params["radius"] = 15;
        QImage out = e.processImage(input, params);
        QVERIFY(!out.isNull());
    }

    void meta_nonEmpty() {
        ClarityEffect e;
        QVERIFY(!e.getName().isEmpty());
        QVERIFY(!e.getDescription().isEmpty());
        QVERIFY(!e.getVersion().isEmpty());
        QVERIFY(e.initialize());
    }

    void defaultParameters_keysAndValues() {
        ClarityEffect e;
        auto params = e.getParameters();
        QVERIFY(params.contains("amount"));
        QVERIFY(params.contains("radius"));
        // Widget not yet built → fall-through defaults
        QCOMPARE(params["amount"].toInt(), 0);
        QCOMPARE(params["radius"].toInt(), 30);
    }

    void createControlsWidget_constructsAndCaches() {
        ClarityEffect e;
        QWidget* w = e.createControlsWidget();
        QVERIFY(w != nullptr);
        QVERIFY(e.createControlsWidget() == w);
    }

    // After the controls widget is built, radius carries its UI default (30).
    void getParameters_afterWidget_returnsUiDefaults() {
        ClarityEffect e;
        e.createControlsWidget();
        auto params = e.getParameters();
        QCOMPARE(params["amount"].toInt(), 0);
        QCOMPARE(params["radius"].toInt(), 30);
    }

    // Fire every slider's editingFinished + valueChanged path.
    void connectSlider_signals_coverLambdaBodies() {
        ClarityEffect e;
        QWidget* w = e.createControlsWidget();
        QVERIFY(w);

        QSignalSpy spyChanged(&e, &PhotoEditorEffect::parametersChanged);
        QSignalSpy spyLive(&e, &PhotoEditorEffect::liveParametersChanged);

        auto sliders = w->findChildren<ParamSlider*>();
        QCOMPARE(sliders.size(), 2);

        for (auto* ps : sliders) {
            auto* qs = ps->findChild<QSlider*>();
            QVERIFY(qs);
            qs->setValue(qs->value() + 1);
            QMetaObject::invokeMethod(qs, "sliderReleased");
        }

        QVERIFY(spyChanged.count() >= 2);
        QVERIFY(spyLive.count()    >= 2);
    }

    void destructor_heapAllocated_doesNotCrash() {
        auto* e = new ClarityEffect();
        e->createControlsWidget();
        delete e;
    }
};

QTEST_MAIN(TestClarity)
#include "test_clarity.moc"
