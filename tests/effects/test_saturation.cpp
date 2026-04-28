#include <QTest>
#include <QSignalSpy>
#include <QSlider>
#include <QWidget>
#include "SaturationEffect.h"
#include "GpuDeviceRegistry.h"
#include "ImageHelpers.h"
#include "ParamSlider.h"

class TestSaturation : public QObject {
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
        SaturationEffect e;
        QVERIFY(e.processImage(QImage(), {}).isNull());
    }

    // Solid grey with sat=0, vib=0: the kernel does RGB→HSV→RGB but with
    // delta=0 the hue/saturation reconstruction is exact → pixel unchanged.
    void grey_identityWithZeroParams() {
        if (!m_hasGpu) QSKIP("No GPU");
        SaturationEffect e;
        QImage input = makeSolid(32, 32, 128, 128, 128);
        QMap<QString, QVariant> params;
        params["saturation"] = 0.0;
        params["vibrancy"]   = 0.0;
        QImage out = e.processImage(input, params);
        QVERIFY(!out.isNull());
        QVERIFY(allPixels(out, [](QRgb px) {
            return qRed(px) == 128 && qGreen(px) == 128 && qBlue(px) == 128;
        }));
    }

    // sat=-20 on a pixel with HSV saturation 0.5 (R=200, G=100, B=100, h=0°):
    // s_new = 0.5 − 0.2 = 0.3 → chroma gap R−G shrinks from 100 to ~60.
    void desaturate_reducesChromaGap() {
        if (!m_hasGpu) QSKIP("No GPU");
        SaturationEffect e;
        QImage input = makeSolid(32, 32, 200, 100, 100);
        QMap<QString, QVariant> params;
        params["saturation"] = -20.0;
        params["vibrancy"]   = 0.0;
        QImage out = e.processImage(input, params);
        QVERIFY(!out.isNull());
        int gapIn  = 200 - 100;
        int gapOut = pixelR(out, 0, 0) - pixelG(out, 0, 0);
        QVERIFY(gapOut < gapIn);
    }

    // sat=+20 on the same pixel: s_new = 0.5 + 0.2 = 0.7 → gap grows to ~140.
    void saturate_increasesChromaGap() {
        if (!m_hasGpu) QSKIP("No GPU");
        SaturationEffect e;
        QImage input = makeSolid(32, 32, 200, 100, 100);
        QMap<QString, QVariant> params;
        params["saturation"] = 20.0;
        params["vibrancy"]   = 0.0;
        QImage out = e.processImage(input, params);
        QVERIFY(!out.isNull());
        int gapIn  = 200 - 100;
        int gapOut = pixelR(out, 0, 0) - pixelG(out, 0, 0);
        QVERIFY(gapOut > gapIn);
    }

    // Fully desaturate a pixel whose HSV saturation matches the slider range.
    // Pixel (200, 148, 148): s ≈ (200-148)/200 = 0.26; after sat=-20: s = 0.06.
    // Not fully grey but gap must shrink substantially.
    void desaturate_lowSatPixel_nearGrey() {
        if (!m_hasGpu) QSKIP("No GPU");
        SaturationEffect e;
        QImage input = makeSolid(32, 32, 200, 148, 148);
        QMap<QString, QVariant> params;
        params["saturation"] = -20.0;
        params["vibrancy"]   = 0.0;
        QImage out = e.processImage(input, params);
        // After desaturation, R−G gap should be smaller than before (52 → ~12).
        int gapOut = pixelR(out, 0, 0) - pixelG(out, 0, 0);
        QVERIFY(gapOut < 52);
    }

    // Vibrancy boosts dull (low-saturation) colours.
    // Pixel (200, 180, 180) has low saturation (s≈0.1).
    // With vib=+20, s increases → gap between R and G should grow.
    void vibrancy_boostsDullColour() {
        if (!m_hasGpu) QSKIP("No GPU");
        SaturationEffect e;
        QImage input = makeSolid(32, 32, 200, 180, 180);
        QMap<QString, QVariant> params;
        params["saturation"] = 0.0;
        params["vibrancy"]   = 20.0;
        QImage out = e.processImage(input, params);
        int gapIn  = 200 - 180;
        int gapOut = pixelR(out, 0, 0) - pixelG(out, 0, 0);
        QVERIFY(gapOut > gapIn);
    }

    // Non-square (wide) image: HSV saturation adjustment is per-pixel.
    // Output dimensions must match input and a positive saturation bump
    // must widen the R−G chroma gap.
    void nonSquare_saturate_increasesChromaGap() {
        if (!m_hasGpu) QSKIP("No GPU");
        SaturationEffect e;
        QImage input = makeSolid(128, 64, 200, 100, 100);
        QMap<QString, QVariant> params;
        params["saturation"] = 20.0;
        params["vibrancy"]   = 0.0;
        QImage out = e.processImage(input, params);
        QVERIFY(!out.isNull());
        QCOMPARE(out.width(),  128);
        QCOMPARE(out.height(), 64);
        int gapIn  = 200 - 100;
        int gapOut = pixelR(out, 0, 0) - pixelG(out, 0, 0);
        QVERIFY(gapOut > gapIn);
    }

    void meta_nonEmpty() {
        SaturationEffect e;
        QVERIFY(!e.getName().isEmpty());
        QVERIFY(!e.getDescription().isEmpty());
        QVERIFY(!e.getVersion().isEmpty());
        QVERIFY(e.initialize());
    }

    void defaultParameters_keys() {
        SaturationEffect e;
        auto params = e.getParameters();
        QVERIFY(params.contains("saturation"));
        QVERIFY(params.contains("vibrancy"));
        QCOMPARE(params["saturation"].toDouble(), 0.0);
        QCOMPARE(params["vibrancy"].toDouble(),   0.0);
    }

    void identity_16bit() {
        if (!m_hasGpu) QSKIP("No GPU");
        SaturationEffect e;
        QImage input = makeSolid16bit(32, 32, 128, 128, 128);
        QMap<QString, QVariant> params;
        params["saturation"] = 0.0;
        params["vibrancy"]   = 0.0;
        QImage out = e.processImage(input, params);
        QVERIFY(!out.isNull());
    }

    void createControlsWidget_constructsAndCaches() {
        SaturationEffect e;
        QWidget* w = e.createControlsWidget();
        QVERIFY(w != nullptr);
        QVERIFY(e.createControlsWidget() == w);
    }

    // 16-bit with non-zero saturation — exercises processImageGPU16.
    void saturation_16bit_nonZeroParams() {
        if (!m_hasGpu) QSKIP("No GPU");
        SaturationEffect e;
        QImage input = makeSolid16bit(32, 32, 200, 100, 100);
        QMap<QString, QVariant> params;
        params["saturation"] = 10.0;
        params["vibrancy"]   = 0.0;
        QImage out = e.processImage(input, params);
        QVERIFY(!out.isNull());
    }

    // 16-bit with non-zero vibrancy — also exercises processImageGPU16.
    void vibrancy_16bit_nonZeroParams() {
        if (!m_hasGpu) QSKIP("No GPU");
        SaturationEffect e;
        QImage input = makeSolid16bit(32, 32, 200, 180, 180);
        QMap<QString, QVariant> params;
        params["saturation"] = 0.0;
        params["vibrancy"]   = 10.0;
        QImage out = e.processImage(input, params);
        QVERIFY(!out.isNull());
    }

    // Fire signal lambdas in createControlsWidget (covers both slider signal lambda bodies).
    void connectSlider_signals_coverLambdaBodies() {
        SaturationEffect e;
        QWidget* w = e.createControlsWidget();
        QVERIFY(w);

        QSignalSpy spyChanged(&e, &PhotoEditorEffect::parametersChanged);
        QSignalSpy spyLive(&e, &PhotoEditorEffect::liveParametersChanged);

        auto sliders = w->findChildren<ParamSlider*>();
        QVERIFY(sliders.size() >= 2);

        for (auto* ps : sliders) {
            auto* qs = ps->findChild<QSlider*>();
            QVERIFY(qs);
            qs->setValue(qs->value() + 1);
            QMetaObject::invokeMethod(qs, "sliderReleased");
        }

        QVERIFY(spyChanged.count() >= 2);
        QVERIFY(spyLive.count() >= 2);
    }

    void destructor_heapAllocated_doesNotCrash() {
        auto* e = new SaturationEffect();
        e->createControlsWidget();
        delete e;
    }
};

QTEST_MAIN(TestSaturation)
#include "test_saturation.moc"
