#include <QTest>
#include <QWidget>
#include "BrightnessEffect.h"
#include "GpuDeviceRegistry.h"
#include "ImageHelpers.h"

class TestBrightness : public QObject {
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

    // processImage(null, {}) must return a null image, not crash.
    void nullImage_passThrough() {
        BrightnessEffect e;
        QImage result = e.processImage(QImage(), {});
        QVERIFY(result.isNull());
    }

    // brightness=0, contrast=0 → identity: no pixel should change.
    void identity_zeroParams() {
        if (!m_hasGpu) QSKIP("No GPU");
        BrightnessEffect e;
        QImage input = makeSolid(32, 32, 128, 90, 60);
        QMap<QString, QVariant> params;
        params["brightness"] = 0;
        params["contrast"]   = 0;
        QImage out = e.processImage(input, params);
        QVERIFY(!out.isNull());
        QVERIFY(allPixels(out, [](QRgb px) {
            return qRed(px) == 128 && qGreen(px) == 90 && qBlue(px) == 60;
        }));
    }

    // brightness=+50 on mid-grey → all pixels must get brighter.
    void brightnessUp_brightensPixels() {
        if (!m_hasGpu) QSKIP("No GPU");
        BrightnessEffect e;
        QImage input = makeSolid(32, 32, 128, 128, 128);
        QMap<QString, QVariant> params;
        params["brightness"] = 50;
        params["contrast"]   = 0;
        QImage out = e.processImage(input, params);
        QVERIFY(!out.isNull());
        QVERIFY(allPixels(out, [](QRgb px) { return qRed(px) > 128; }));
    }

    // brightness=-50 on mid-grey → all pixels must get darker.
    void brightnessDown_darkensPixels() {
        if (!m_hasGpu) QSKIP("No GPU");
        BrightnessEffect e;
        QImage input = makeSolid(32, 32, 128, 128, 128);
        QMap<QString, QVariant> params;
        params["brightness"] = -50;
        params["contrast"]   = 0;
        QImage out = e.processImage(input, params);
        QVERIFY(!out.isNull());
        QVERIFY(allPixels(out, [](QRgb px) { return qRed(px) < 128; }));
    }

    // White + max brightness → must not overflow: still 255.
    void white_clampsAt255() {
        if (!m_hasGpu) QSKIP("No GPU");
        BrightnessEffect e;
        QImage input = makeSolid(32, 32, 255, 255, 255);
        QMap<QString, QVariant> params;
        params["brightness"] = 100;
        params["contrast"]   = 0;
        QImage out = e.processImage(input, params);
        QVERIFY(allPixels(out, [](QRgb px) {
            return qRed(px) == 255 && qGreen(px) == 255 && qBlue(px) == 255;
        }));
    }

    // Black + max negative brightness → must not underflow: still 0.
    void black_clampsAt0() {
        if (!m_hasGpu) QSKIP("No GPU");
        BrightnessEffect e;
        QImage input = makeSolid(32, 32, 0, 0, 0);
        QMap<QString, QVariant> params;
        params["brightness"] = -100;
        params["contrast"]   = 0;
        QImage out = e.processImage(input, params);
        QVERIFY(allPixels(out, [](QRgb px) {
            return qRed(px) == 0 && qGreen(px) == 0 && qBlue(px) == 0;
        }));
    }

    // contrast=+50: dark pixels should get darker, bright pixels get brighter.
    // contrastFactor = (50+100)/100 = 1.5; pivot is at 0.5 (≈128).
    void contrastUp_spreadsPixels() {
        if (!m_hasGpu) QSKIP("No GPU");
        BrightnessEffect e;
        QMap<QString, QVariant> params;
        params["brightness"] = 0;
        params["contrast"]   = 50;

        QImage dark = makeSolid(32, 32, 50, 50, 50);
        QImage darkOut = e.processImage(dark, params);
        QVERIFY(pixelR(darkOut, 0, 0) < 50);

        QImage bright = makeSolid(32, 32, 200, 200, 200);
        QImage brightOut = e.processImage(bright, params);
        QVERIFY(pixelR(brightOut, 0, 0) > 200);
    }

    // contrast=-50: dark and bright pixels should converge toward mid-grey.
    // contrastFactor = (-50+100)/100 = 0.5.
    void contrastDown_compressesPixels() {
        if (!m_hasGpu) QSKIP("No GPU");
        BrightnessEffect e;
        QMap<QString, QVariant> params;
        params["brightness"] = 0;
        params["contrast"]   = -50;

        QImage dark = makeSolid(32, 32, 50, 50, 50);
        QImage darkOut = e.processImage(dark, params);
        QVERIFY(pixelR(darkOut, 0, 0) > 50);

        QImage bright = makeSolid(32, 32, 200, 200, 200);
        QImage brightOut = e.processImage(bright, params);
        QVERIFY(pixelR(brightOut, 0, 0) < 200);
    }

    void meta_nonEmpty() {
        BrightnessEffect e;
        QVERIFY(!e.getName().isEmpty());
        QVERIFY(!e.getDescription().isEmpty());
        QVERIFY(!e.getVersion().isEmpty());
        QVERIFY(e.initialize());
    }

    void defaultParameters_keys() {
        BrightnessEffect e;
        auto params = e.getParameters();
        QVERIFY(params.contains("brightness"));
        QVERIFY(params.contains("contrast"));
        QCOMPARE(params["brightness"].toInt(), 0);
        QCOMPARE(params["contrast"].toInt(), 0);
    }

    void identity_16bit() {
        if (!m_hasGpu) QSKIP("No GPU");
        BrightnessEffect e;
        QImage input = makeSolid16bit(32, 32, 128, 90, 60);
        QMap<QString, QVariant> params;
        params["brightness"] = 0;
        params["contrast"]   = 0;
        QImage out = e.processImage(input, params);
        QVERIFY(!out.isNull());
    }

    void createControlsWidget_constructsAndCaches() {
        BrightnessEffect e;
        QWidget* w = e.createControlsWidget();
        QVERIFY(w != nullptr);
        QVERIFY(e.createControlsWidget() == w);
    }
};

QTEST_MAIN(TestBrightness)
#include "test_brightness.moc"
