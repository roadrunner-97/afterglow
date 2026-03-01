#include <QTest>
#include <QWidget>
#include <cmath>
#include "ExposureEffect.h"
#include "GpuDeviceRegistry.h"
#include "ImageHelpers.h"

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
};

QTEST_MAIN(TestExposure)
#include "test_exposure.moc"
