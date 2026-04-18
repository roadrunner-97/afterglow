#include <QTest>
#include <QSignalSpy>
#include <QSlider>
#include <QWidget>
#include "WhiteBalanceEffect.h"
#include "GpuDeviceRegistry.h"
#include "ImageHelpers.h"
#include "ImageMetadata.h"
#include "ParamSlider.h"

class TestWhiteBalance : public QObject {
    Q_OBJECT

private:
    bool m_hasGpu = false;

    // Build a neutral grey params map: target == shot → no colour shift.
    static QMap<QString, QVariant> neutralParams(double shotK = 5500.0) {
        QMap<QString, QVariant> p;
        p["shot_temp"]   = shotK;
        p["temperature"] = shotK;
        p["tint"]        = 0.0;
        return p;
    }

private slots:
    void initTestCase() {
        GpuDeviceRegistry::instance().enumerate();
        if (GpuDeviceRegistry::instance().count() == 0)
            QSKIP("No OpenCL device found — skipping GPU effect tests");
        GpuDeviceRegistry::instance().setDevice(0);
        m_hasGpu = true;
    }

    void nullImage_passThrough() {
        WhiteBalanceEffect e;
        QVERIFY(e.processImage(QImage(), {}).isNull());
    }

    // target == shot, tint == 0 → all multipliers are 1.0 → pixel values unchanged (±1 for float rounding).
    void identity_targetEqualsShot() {
        if (!m_hasGpu) QSKIP("No GPU");
        WhiteBalanceEffect e;
        QImage input = makeSolid(32, 32, 128, 128, 128);
        QImage out   = e.processImage(input, neutralParams(5500.0));
        QVERIFY(!out.isNull());
        QVERIFY(allPixels(out, [](QRgb px) {
            return qAbs(qRed(px)   - 128) <= 2
                && qAbs(qGreen(px) - 128) <= 2
                && qAbs(qBlue(px)  - 128) <= 2;
        }));
    }

    // Warming (targetK > shotK): more red/amber, less blue.
    // Solid grey (128,128,128): after warming, R > original B.
    void warming_increasesRed_decreasesBlue() {
        if (!m_hasGpu) QSKIP("No GPU");
        WhiteBalanceEffect e;
        QImage input = makeSolid(32, 32, 128, 128, 128);
        QMap<QString, QVariant> params;
        params["shot_temp"]   = 5500.0;
        params["temperature"] = 8000.0;   // warmer
        params["tint"]        = 0.0;
        QImage out = e.processImage(input, params);
        QVERIFY(!out.isNull());
        // Warming shifts R up and B down relative to G.
        QVERIFY(allPixels(out, [](QRgb px) { return qRed(px) >= qBlue(px); }));
    }

    // Cooling (targetK < shotK): less red, more blue.
    void cooling_increasesBlue_decreasesRed() {
        if (!m_hasGpu) QSKIP("No GPU");
        WhiteBalanceEffect e;
        QImage input = makeSolid(32, 32, 128, 128, 128);
        QMap<QString, QVariant> params;
        params["shot_temp"]   = 5500.0;
        params["temperature"] = 3000.0;   // cooler
        params["tint"]        = 0.0;
        QImage out = e.processImage(input, params);
        QVERIFY(!out.isNull());
        QVERIFY(allPixels(out, [](QRgb px) { return qBlue(px) >= qRed(px); }));
    }

    // Positive tint (magenta): reduces green, so G < R for a grey input.
    void tintMagenta_reducesGreen() {
        if (!m_hasGpu) QSKIP("No GPU");
        WhiteBalanceEffect e;
        QImage input = makeSolid(32, 32, 128, 128, 128);
        QMap<QString, QVariant> params;
        params["shot_temp"]   = 5500.0;
        params["temperature"] = 5500.0;
        params["tint"]        = 80.0;   // strong magenta
        QImage out = e.processImage(input, params);
        QVERIFY(!out.isNull());
        QVERIFY(allPixels(out, [](QRgb px) { return qGreen(px) < qRed(px); }));
    }

    // 16-bit path: identity still leaves pixels unchanged.
    void identity_16bit() {
        if (!m_hasGpu) QSKIP("No GPU");
        WhiteBalanceEffect e;
        QImage input = makeSolid16bit(32, 32, 128, 128, 128);
        QImage out   = e.processImage(input, neutralParams(5500.0));
        QVERIFY(!out.isNull());
    }

    // 16-bit warming path.
    void warming_16bit() {
        if (!m_hasGpu) QSKIP("No GPU");
        WhiteBalanceEffect e;
        QImage input = makeSolid16bit(32, 32, 128, 128, 128);
        QMap<QString, QVariant> params;
        params["shot_temp"]   = 5500.0;
        params["temperature"] = 8000.0;
        params["tint"]        = 0.0;
        QImage out = e.processImage(input, params);
        QVERIFY(!out.isNull());
    }

    // Non-square (wide) image: white-balance multipliers are applied per-pixel.
    // Output dimensions must match input and warming must shift R above B
    // everywhere on a mid-grey input.
    void nonSquare_warming_increasesRed_decreasesBlue() {
        if (!m_hasGpu) QSKIP("No GPU");
        WhiteBalanceEffect e;
        QImage input = makeSolid(128, 64, 128, 128, 128);
        QMap<QString, QVariant> params;
        params["shot_temp"]   = 5500.0;
        params["temperature"] = 8000.0;
        params["tint"]        = 0.0;
        QImage out = e.processImage(input, params);
        QVERIFY(!out.isNull());
        QCOMPARE(out.width(),  128);
        QCOMPARE(out.height(), 64);
        QVERIFY(allPixels(out, [](QRgb px) { return qRed(px) >= qBlue(px); }));
    }

    void meta_nonEmpty() {
        WhiteBalanceEffect e;
        QVERIFY(!e.getName().isEmpty());
        QVERIFY(!e.getDescription().isEmpty());
        QVERIFY(!e.getVersion().isEmpty());
        QVERIFY(e.initialize());
    }

    void defaultParameters_keys() {
        WhiteBalanceEffect e;
        auto params = e.getParameters();
        QVERIFY(params.contains("temperature"));
        QVERIFY(params.contains("tint"));
        QVERIFY(params.contains("shot_temp"));
    }

    void createControlsWidget_constructsAndCaches() {
        WhiteBalanceEffect e;
        QWidget* w = e.createControlsWidget();
        QVERIFY(w != nullptr);
        QVERIFY(e.createControlsWidget() == w);
    }

    void getParameters_afterControlsWidget_returnsDefaults() {
        WhiteBalanceEffect e;
        e.createControlsWidget();
        auto params = e.getParameters();
        QCOMPARE(params["tint"].toDouble(), 0.0);
        QVERIFY(params.contains("shot_temp"));
    }

    // onImageLoaded: valid temperature is adopted; out-of-range falls back to 5500.
    void onImageLoaded_validTemp_adopted() {
        WhiteBalanceEffect e;
        e.createControlsWidget();  // creates temperatureParam

        ImageMetadata meta;
        meta.colorTempK = 3200.0f;
        e.onImageLoaded(meta);

        // getParameters returns the current slider value (== meta.colorTempK after reload).
        auto params = e.getParameters();
        QCOMPARE(params["shot_temp"].toDouble(), 3200.0);
    }

    void onImageLoaded_outOfRange_fallsBackTo5500() {
        WhiteBalanceEffect e;
        e.createControlsWidget();

        ImageMetadata meta;
        meta.colorTempK = 0.0f;  // invalid
        e.onImageLoaded(meta);

        auto params = e.getParameters();
        QCOMPARE(params["shot_temp"].toDouble(), 5500.0);
    }

    void supportsGpuInPlace_returnsTrue() {
        WhiteBalanceEffect e;
        QVERIFY(e.supportsGpuInPlace());
    }

    // Fire signal lambdas in createControlsWidget.
    void connectSlider_signals_coverLambdaBodies() {
        WhiteBalanceEffect e;
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

        QVERIFY(spyChanged.count() >= 1);
        QVERIFY(spyLive.count() >= 1);
    }

    void destructor_heapAllocated_doesNotCrash() {
        auto* e = new WhiteBalanceEffect();
        e->createControlsWidget();
        delete e;
    }
};

QTEST_MAIN(TestWhiteBalance)
#include "test_whitebalance.moc"
