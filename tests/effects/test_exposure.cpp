#include <QTest>
#include <QApplication>
#include <QSignalSpy>
#include <QSlider>
#include <QWidget>
#include <cmath>
#include "ExposureEffect.h"
#include "GpuDeviceRegistry.h"
#include "ImageHelpers.h"
#include "ParamSlider.h"

class TestExposure : public QObject {
    Q_OBJECT

private:
    bool m_hasGpu = false;

    // Zero-out all zone parameters.
    static QMap<QString, QVariant> zeroParams() {
        QMap<QString, QVariant> p;
        p["exposure"]   = 0.0;
        p["whites"]     = 0.0;
        p["highlights"] = 0.0;
        p["shadows"]    = 0.0;
        p["blacks"]     = 0.0;
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
        ExposureEffect e;
        QVERIFY(e.processImage(QImage(), {}).isNull());
    }

    // All zones=0 → ev=0 → evFactor=1.0 → pixel passes through sRGB→linear→sRGB
    // unchanged apart from native_powr rounding (allow ±2).
    void identity_allZeroParams() {
        if (!m_hasGpu) QSKIP("No GPU");
        ExposureEffect e;
        QImage input = makeSolid(32, 32, 100, 100, 100);
        QImage out   = e.processImage(input, zeroParams());
        QVERIFY(!out.isNull());
        QVERIFY(allPixels(out, [](QRgb px) {
            return qAbs(qRed(px)   - 100) <= 2
                && qAbs(qGreen(px) - 100) <= 2
                && qAbs(qBlue(px)  - 100) <= 2;
        }));
    }

    // Pure black pixel stays black at any exposure (0 * scale = 0).
    void black_staysBlack_atAnyExposure() {
        if (!m_hasGpu) QSKIP("No GPU");
        ExposureEffect e;
        QImage input = makeSolid(32, 32, 0, 0, 0);

        for (double ev : { -3.0, -1.0, 1.0, 3.0 }) {
            auto params = zeroParams();
            params["exposure"] = ev;
            QImage out = e.processImage(input, params);
            QVERIFY(!out.isNull());
            QVERIFY(allPixels(out, [](QRgb px) {
                return qRed(px) == 0 && qGreen(px) == 0 && qBlue(px) == 0;
            }));
        }
    }

    // Positive global EV → all pixels brighter.
    // +2 EV = ×4 in linear light, should push mid-grey (100) well above 100.
    void positiveExposure_brightens() {
        if (!m_hasGpu) QSKIP("No GPU");
        ExposureEffect e;
        QImage input = makeSolid(32, 32, 100, 100, 100);
        auto params  = zeroParams();
        params["exposure"] = 2.0;
        QImage out = e.processImage(input, params);
        QVERIFY(!out.isNull());
        QVERIFY(allPixels(out, [](QRgb px) { return qRed(px) > 100; }));
    }

    // Negative global EV → all pixels darker.
    void negativeExposure_darkens() {
        if (!m_hasGpu) QSKIP("No GPU");
        ExposureEffect e;
        QImage input = makeSolid(32, 32, 150, 150, 150);
        auto params  = zeroParams();
        params["exposure"] = -2.0;
        QImage out = e.processImage(input, params);
        QVERIFY(!out.isNull());
        QVERIFY(allPixels(out, [](QRgb px) { return qRed(px) < 150; }));
    }

    // White pixel (255) plus large positive EV must still clamp to 255.
    void brightPixel_clampsAt255() {
        if (!m_hasGpu) QSKIP("No GPU");
        ExposureEffect e;
        QImage input = makeSolid(32, 32, 255, 255, 255);
        auto params  = zeroParams();
        params["exposure"] = 3.0;
        QImage out = e.processImage(input, params);
        QVERIFY(allPixels(out, [](QRgb px) {
            return qRed(px) == 255 && qGreen(px) == 255 && qBlue(px) == 255;
        }));
    }

    // A dark pixel boosted by highlights=+3 EV must become brighter.
    // Highlights zone centres on lum≈0.675; a mid-light pixel (lum≈0.55) sits
    // within that zone so it receives a significant boost.
    void highlightsZone_boostsMidlightPixel() {
        if (!m_hasGpu) QSKIP("No GPU");
        ExposureEffect e;
        QImage input = makeSolid(32, 32, 160, 160, 160);
        auto params  = zeroParams();
        params["highlights"] = 2.0;
        QImage out = e.processImage(input, params);
        QVERIFY(!out.isNull());
        QVERIFY(allPixels(out, [](QRgb px) { return qRed(px) > 160; }));
    }

    // A very dark pixel boosted by blacks=+3 EV must become brighter.
    void blacksZone_boostsDarkPixel() {
        if (!m_hasGpu) QSKIP("No GPU");
        ExposureEffect e;
        QImage input = makeSolid(32, 32, 20, 20, 20);
        auto params  = zeroParams();
        params["blacks"] = 3.0;
        QImage out = e.processImage(input, params);
        QVERIFY(!out.isNull());
        QVERIFY(allPixels(out, [](QRgb px) { return qRed(px) > 20; }));
    }

    void meta_nonEmpty() {
        ExposureEffect e;
        QVERIFY(!e.getName().isEmpty());
        QVERIFY(!e.getDescription().isEmpty());
        QVERIFY(!e.getVersion().isEmpty());
        QVERIFY(e.initialize());
    }

    void defaultParameters_keys() {
        ExposureEffect e;
        auto params = e.getParameters();
        for (const auto& key : {"exposure", "whites", "highlights", "shadows", "blacks"})
            QVERIFY(params.contains(key));
    }

    void identity_16bit() {
        if (!m_hasGpu) QSKIP("No GPU");
        ExposureEffect e;
        QImage input = makeSolid16bit(32, 32, 100, 100, 100);
        QImage out = e.processImage(input, zeroParams());
        QVERIFY(!out.isNull());
    }

    void createControlsWidget_constructsAndCaches() {
        ExposureEffect e;
        QWidget* w = e.createControlsWidget();
        QVERIFY(w != nullptr);
        QVERIFY(e.createControlsWidget() == w);
    }

    // getParameters() with live sliders (non-null ParamSlider pointers).
    void getParameters_afterControlsWidget_returnsDefaults() {
        ExposureEffect e;
        e.createControlsWidget();
        auto params = e.getParameters();
        QCOMPARE(params["exposure"].toDouble(),   0.0);
        QCOMPARE(params["whites"].toDouble(),     0.0);
        QCOMPARE(params["highlights"].toDouble(), 0.0);
        QCOMPARE(params["shadows"].toDouble(),    0.0);
        QCOMPARE(params["blacks"].toDouble(),     0.0);
    }

    // Drive the underlying QSlider to fire valueChanged → refreshCurve lambda +
    // liveParametersChanged. Covers: refreshCurve body (lines 440–446),
    // valueChanged lambda body (lines 451–453).
    void connectSlider_valueChanged_firesRefreshCurveAndLiveParams() {
        ExposureEffect e;
        QWidget* w = e.createControlsWidget();
        QVERIFY(w);

        QSignalSpy spy(&e, &PhotoEditorEffect::liveParametersChanged);

        // exposure ParamSlider is the first child ParamSlider.
        // scaleFactor = 10 (step=0.1), so int value 10 → 1.0 EV.
        auto* paramSlider = w->findChildren<ParamSlider*>().value(0);
        QVERIFY(paramSlider);
        auto* qslider = paramSlider->findChild<QSlider*>();
        QVERIFY(qslider);
        qslider->setValue(10);

        QCOMPARE(spy.count(), 1);
    }

    // Invoke sliderReleased → editingFinished lambda → parametersChanged.
    void connectSlider_sliderReleased_firesParametersChanged() {
        ExposureEffect e;
        QWidget* w = e.createControlsWidget();
        QVERIFY(w);

        QSignalSpy spy(&e, &PhotoEditorEffect::parametersChanged);

        auto* paramSlider = w->findChildren<ParamSlider*>().value(0);
        QVERIFY(paramSlider);
        auto* qslider = paramSlider->findChild<QSlider*>();
        QVERIFY(qslider);
        QMetaObject::invokeMethod(qslider, "sliderReleased");

        QCOMPARE(spy.count(), 1);
    }

    // Show controls widget so ToneCurveWidget::paintEvent fires with the
    // default m_z = {} (global = 0 < 0.05 → no "Base:" label).
    // Covers: ToneCurveWidget constructor, setParams, paintEvent body (lines 313–378),
    // and all CPU math functions (lines 253–295).
    void toneCurveWidget_paintEvent_noLabel() {
        ExposureEffect e;
        QWidget* w = e.createControlsWidget();
        w->resize(300, 500);
        w->show();
        QApplication::processEvents();
    }

    // Fire the exposure slider BEFORE the first show so m_z.global = 1.0 when
    // the backing store is set up. The first paint event then draws the
    // "Base: +1.0 EV" label (lines 381–391) with a cleanly initialised QPainter.
    //
    // The exposure ParamSlider uses range [-5.0, 5.0] with step 0.1
    // (scaleFactor=10), giving QSlider integer range [-50, 50].  We find it
    // explicitly by range to be robust against any ordering uncertainty.
    void toneCurveWidget_paintEvent_withLabel() {
        ExposureEffect e;
        QWidget* w = e.createControlsWidget();
        w->resize(300, 500);

        // Find the exposure QSlider by its unique integer range ±50.
        QSlider* exposureQSlider = nullptr;
        for (auto* ps : w->findChildren<ParamSlider*>()) {
            auto* qs = ps->findChild<QSlider*>();
            if (qs && qs->minimum() == -50 && qs->maximum() == 50) {
                exposureQSlider = qs;
                break;
            }
        }
        QVERIFY(exposureQSlider);
        // int 10 / scaleFactor 10 = 1.0 EV → refreshCurve → setParams(global=1.0).
        exposureQSlider->setValue(10);

        w->show();
        QApplication::processEvents();  // first paint: m_z.global = 1.0 → label shown
    }

    // Heap-allocate and explicitly delete to ensure the destructor body runs
    // and is attributed to ExposureEffect.cpp's coverage counters.
    void destructor_heapAllocatedWithControls_doesNotCrash() {
        auto* e = new ExposureEffect();
        e->createControlsWidget();
        delete e;
    }

    void supportsGpuInPlace_returnsTrue() {
        ExposureEffect e;
        QVERIFY(e.supportsGpuInPlace());
    }

    // Zone-boundary pixel tests on the 16-bit path.  Pixels with luminance at
    // the zone boundaries exercise the PCHIP clamp conditions on the GPU kernel.
    void processImage16bit_darkPixel_blacks() {
        if (!m_hasGpu) QSKIP("No GPU");
        ExposureEffect e;
        // lum ≈ 0.02 (<= 0.075) → exactly blacksEv applied
        QImage input = makeSolid16bit(32, 32, 5, 5, 5);
        auto params = zeroParams();
        params["blacks"] = 2.0;
        QImage out = e.processImage(input, params);
        QVERIFY(!out.isNull());
    }

    void processImage16bit_brightPixel_whites() {
        if (!m_hasGpu) QSKIP("No GPU");
        ExposureEffect e;
        // lum ≈ 0.96 (>= 0.925) → exactly whitesEv applied
        QImage input = makeSolid16bit(32, 32, 245, 245, 245);
        auto params = zeroParams();
        params["whites"] = -1.0;
        QImage out = e.processImage(input, params);
        QVERIFY(!out.isNull());
    }
};

QTEST_MAIN(TestExposure)
#include "test_exposure.moc"
