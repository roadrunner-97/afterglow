#include <QTest>
#include <QSignalSpy>
#include <QSlider>
#include <QWidget>
#include "SplitToningEffect.h"
#include "GpuDeviceRegistry.h"
#include "ImageHelpers.h"
#include "ParamSlider.h"

class TestSplitToning : public QObject {
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
        SplitToningEffect e;
        QVERIFY(e.processImage(QImage(), {}).isNull());
    }

    // Both saturations at 0 → identity, regardless of the other sliders.
    void zeroSaturations_isIdentity() {
        SplitToningEffect e;
        QImage input = makeSolid(64, 64, 100, 120, 80);
        QMap<QString, QVariant> params;
        params["shadowHue"]    = 240;  // blue
        params["shadowSat"]    = 0;
        params["highlightHue"] = 60;   // yellow
        params["highlightSat"] = 0;
        params["balance"]      = 0;
        QImage out = e.processImage(input, params);
        QVERIFY(!out.isNull());
        QVERIFY(allPixels(out, [](QRgb px) {
            return qRed(px) == 100 && qGreen(px) == 120 && qBlue(px) == 80;
        }));
    }

    // Shadow tint at hue=240 (blue) on a dark gray pushes the pixel's blue
    // channel above its red/green channels.
    void shadowTint_biasesDarkPixelsTowardHue() {
        if (!m_hasGpu) QSKIP("No GPU");
        SplitToningEffect e;
        QImage input = makeSolid(64, 64, 60, 60, 60);  // dark neutral gray
        QMap<QString, QVariant> params;
        params["shadowHue"]    = 240;  // blue
        params["shadowSat"]    = 100;
        params["highlightHue"] = 0;
        params["highlightSat"] = 0;
        params["balance"]      = 0;
        QImage out = e.processImage(input, params);
        QVERIFY(!out.isNull());

        int r = pixelR(out, 32, 32);
        int g = pixelG(out, 32, 32);
        int b = pixelB(out, 32, 32);
        QVERIFY2(b > r && b > g,
                 qPrintable(QString("r=%1 g=%2 b=%3").arg(r).arg(g).arg(b)));
    }

    // Highlight tint at hue=0 (red) on a bright gray raises the red channel
    // above green/blue.
    void highlightTint_biasesBrightPixelsTowardHue() {
        if (!m_hasGpu) QSKIP("No GPU");
        SplitToningEffect e;
        QImage input = makeSolid(64, 64, 200, 200, 200);  // bright neutral
        QMap<QString, QVariant> params;
        params["shadowHue"]    = 0;
        params["shadowSat"]    = 0;
        params["highlightHue"] = 0;    // red
        params["highlightSat"] = 100;
        params["balance"]      = 0;
        QImage out = e.processImage(input, params);
        QVERIFY(!out.isNull());

        int r = pixelR(out, 32, 32);
        int g = pixelG(out, 32, 32);
        int b = pixelB(out, 32, 32);
        QVERIFY2(r > g && r > b,
                 qPrintable(QString("r=%1 g=%2 b=%3").arg(r).arg(g).arg(b)));
    }

    // Shadow tint should leave a bright neutral pixel essentially unchanged
    // (highlightMask≈1, shadowMask≈0) when only the shadow slider is set.
    void shadowTint_leavesBrightPixelsAlone() {
        if (!m_hasGpu) QSKIP("No GPU");
        SplitToningEffect e;
        QImage input = makeSolid(64, 64, 250, 250, 250);
        QMap<QString, QVariant> params;
        params["shadowHue"]    = 240;
        params["shadowSat"]    = 100;
        params["highlightHue"] = 0;
        params["highlightSat"] = 0;
        params["balance"]      = 0;
        QImage out = e.processImage(input, params);
        QVERIFY(!out.isNull());

        int r = pixelR(out, 32, 32);
        int g = pixelG(out, 32, 32);
        int b = pixelB(out, 32, 32);
        // Luminance is ~0.98 so shadowMask ≈ 0.02 → only a tiny shift.
        QVERIFY2(std::abs(r - 250) < 15 && std::abs(g - 250) < 15 && std::abs(b - 250) < 15,
                 qPrintable(QString("r=%1 g=%2 b=%3").arg(r).arg(g).arg(b)));
    }

    // Balance=+100 pushes shadowMask toward 0 everywhere (highlight-dominated):
    // a midtone grey with only a shadow tint enabled should be close to identity.
    void balance_shiftsCrossover() {
        if (!m_hasGpu) QSKIP("No GPU");
        SplitToningEffect e;
        QImage input = makeSolid(64, 64, 128, 128, 128);
        QMap<QString, QVariant> p;
        p["shadowHue"]    = 240;
        p["shadowSat"]    = 100;
        p["highlightHue"] = 0;
        p["highlightSat"] = 0;

        p["balance"] = -100;
        QImage shadowHeavy = e.processImage(input, p);
        p["balance"] =  100;
        QImage highlightHeavy = e.processImage(input, p);
        QVERIFY(!shadowHeavy.isNull() && !highlightHeavy.isNull());

        // Shadow tint is pure blue: a shadow-heavy balance pulls R/G toward zero;
        // a highlight-heavy balance leaves the gray largely intact (the highlight
        // tint is disabled).
        int rShadow    = pixelR(shadowHeavy,    32, 32);
        int rHighlight = pixelR(highlightHeavy, 32, 32);
        QVERIFY2(rShadow < rHighlight - 30,
                 qPrintable(QString("rShadow=%1 rHighlight=%2").arg(rShadow).arg(rHighlight)));
    }

    // 16-bit path: non-null output when a tint is active.
    void tint_16bit() {
        if (!m_hasGpu) QSKIP("No GPU");
        SplitToningEffect e;
        QImage input = makeSolid16bit(64, 64, 128, 128, 128);
        QMap<QString, QVariant> p;
        p["shadowHue"]    = 240;
        p["shadowSat"]    = 50;
        p["highlightHue"] = 60;
        p["highlightSat"] = 50;
        p["balance"]      = 0;
        QImage out = e.processImage(input, p);
        QVERIFY(!out.isNull());
    }

    void meta_nonEmpty() {
        SplitToningEffect e;
        QVERIFY(!e.getName().isEmpty());
        QVERIFY(!e.getDescription().isEmpty());
        QVERIFY(!e.getVersion().isEmpty());
        QVERIFY(e.initialize());
    }

    void defaultParameters_keysAndValues() {
        SplitToningEffect e;
        auto params = e.getParameters();
        QVERIFY(params.contains("shadowHue"));
        QVERIFY(params.contains("shadowSat"));
        QVERIFY(params.contains("highlightHue"));
        QVERIFY(params.contains("highlightSat"));
        QVERIFY(params.contains("balance"));
        QCOMPARE(params["shadowHue"].toInt(),    0);
        QCOMPARE(params["shadowSat"].toInt(),    0);
        QCOMPARE(params["highlightHue"].toInt(), 0);
        QCOMPARE(params["highlightSat"].toInt(), 0);
        QCOMPARE(params["balance"].toInt(),      0);
    }

    void createControlsWidget_constructsAndCaches() {
        SplitToningEffect e;
        QWidget* w = e.createControlsWidget();
        QVERIFY(w != nullptr);
        QVERIFY(e.createControlsWidget() == w);
    }

    void supportsGpuInPlace_returnsTrue() {
        SplitToningEffect e;
        QVERIFY(e.supportsGpuInPlace());
    }

    // Fire every slider's editingFinished + valueChanged path.
    void connectSlider_signals_coverLambdaBodies() {
        SplitToningEffect e;
        QWidget* w = e.createControlsWidget();
        QVERIFY(w);

        QSignalSpy spyChanged(&e, &PhotoEditorEffect::parametersChanged);
        QSignalSpy spyLive(&e, &PhotoEditorEffect::liveParametersChanged);

        auto sliders = w->findChildren<ParamSlider*>();
        QCOMPARE(sliders.size(), 5);

        for (auto* ps : sliders) {
            auto* qs = ps->findChild<QSlider*>();
            QVERIFY(qs);
            qs->setValue(qs->value() + 1);
            QMetaObject::invokeMethod(qs, "sliderReleased");
        }

        QVERIFY(spyChanged.count() >= 5);
        QVERIFY(spyLive.count()    >= 5);
    }

    void destructor_heapAllocated_doesNotCrash() {
        auto* e = new SplitToningEffect();
        e->createControlsWidget();
        delete e;
    }
};

QTEST_MAIN(TestSplitToning)
#include "test_splittoning.moc"
