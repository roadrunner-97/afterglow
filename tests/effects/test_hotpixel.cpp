#include <QTest>
#include <QSignalSpy>
#include <QSlider>
#include <QWidget>
#include "HotPixelEffect.h"
#include "GpuDeviceRegistry.h"
#include "ImageHelpers.h"
#include "ParamSlider.h"

class TestHotPixel : public QObject {
    Q_OBJECT

private:
    bool m_hasGpu = false;

    // Build a uniform image with a single isolated bright spike at (cx, cy).
    static QImage makeWithSpike(int w, int h, int bgVal, int cx, int cy) {
        QImage img = makeSolid(w, h, bgVal, bgVal, bgVal);
        QRgb* row  = reinterpret_cast<QRgb*>(img.scanLine(cy));
        row[cx]    = qRgb(255, 255, 255);
        return img;
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
        HotPixelEffect e;
        QVERIFY(e.processImage(QImage(), {}).isNull());
    }

    // threshold=0 → early return, image unchanged (code: if(threshold==0) return image).
    void zeroThreshold_isIdentity() {
        HotPixelEffect e;
        QImage input = makeWithSpike(32, 32, 64, 16, 16);
        QMap<QString, QVariant> params;
        params["threshold"] = 0;
        QImage out = e.processImage(input, params);
        QVERIFY(!out.isNull());
        // Spike must be untouched.
        QCOMPARE(pixelR(out, 16, 16), 255);
    }

    // Uniform image: every pixel equals its neighbours → deviation=0 → no pixel replaced.
    void uniformImage_unchanged() {
        if (!m_hasGpu) QSKIP("No GPU");
        HotPixelEffect e;
        QImage input = makeSolid(32, 32, 80, 80, 80);
        QMap<QString, QVariant> params;
        params["threshold"] = 30;
        QImage out = e.processImage(input, params);
        QVERIFY(!out.isNull());
        QVERIFY(allPixels(out, [](QRgb px) { return qRed(px) == 80; }));
    }

    // Isolated bright spike in a uniform grey field must be replaced by the
    // neighbourhood average (≈bgVal) after correction.
    //
    // Image: 32×32, bgVal=64, spike at (16,16)=255.
    // threshold=30 → channel threshold = 30*2.55 = 76.5
    // spike deviation = |255 - 64| = 191 > 76.5 → replaced with avg ≈ 64.
    void hotSpike_isReplaced() {
        if (!m_hasGpu) QSKIP("No GPU");
        HotPixelEffect e;
        const int bgVal = 64;
        QImage input = makeWithSpike(32, 32, bgVal, 16, 16);
        QMap<QString, QVariant> params;
        params["threshold"] = 30;
        QImage out = e.processImage(input, params);
        QVERIFY(!out.isNull());
        // Corrected pixel should be close to the background value.
        int corrected = pixelR(out, 16, 16);
        QVERIFY(corrected < 200);          // definitely not still the spike
        QVERIFY(qAbs(corrected - bgVal) <= 5);  // close to neighbourhood average
    }

    // Pixels adjacent to the hot spike must NOT be replaced.
    // Their deviation from the neighbourhood average is low enough
    // because the spike's contribution to the avg is small (1/8 of 8 neighbours).
    //
    // Neighbour avg for pixel at (15,16):
    //   7 neighbours = bgVal(64), 1 neighbour = spike(255 in original buffer)
    //   avg = (7*64 + 255) / 8 = 703/8 ≈ 87.9
    //   deviation = |64 - 87.9| = 23.9 < 76.5 → NOT replaced.
    void normalPixel_nearSpike_unchanged() {
        if (!m_hasGpu) QSKIP("No GPU");
        HotPixelEffect e;
        const int bgVal = 64;
        QImage input = makeWithSpike(32, 32, bgVal, 16, 16);
        QMap<QString, QVariant> params;
        params["threshold"] = 30;
        QImage out = e.processImage(input, params);
        QVERIFY(!out.isNull());
        // Pixel to the left of the spike must stay at bgVal.
        QCOMPARE(pixelR(out, 15, 16), bgVal);
    }

    // High threshold (100) → threshold_channel = 255 → almost nothing replaced.
    void highThreshold_preservesSpike() {
        if (!m_hasGpu) QSKIP("No GPU");
        HotPixelEffect e;
        QImage input = makeWithSpike(32, 32, 200, 16, 16);
        QMap<QString, QVariant> params;
        params["threshold"] = 100;   // channel threshold = 255
        QImage out = e.processImage(input, params);
        QVERIFY(!out.isNull());
        // Spike deviation = |255-200|=55 < 255 → NOT replaced → still 255.
        QCOMPARE(pixelR(out, 16, 16), 255);
    }

    // Non-square (wide) image with a spike at a non-centred location.  The
    // neighbourhood kernel must index correctly into a non-square buffer:
    // the spike should be replaced and a far-from-spike pixel must remain
    // at the background value.  Output dimensions must match input.
    void nonSquare_hotSpike_isReplaced() {
        if (!m_hasGpu) QSKIP("No GPU");
        HotPixelEffect e;
        const int bgVal = 64;
        // Spike placed at (96, 32) — near the right half, far from centre.
        QImage input = makeWithSpike(128, 64, bgVal, 96, 32);
        QMap<QString, QVariant> params;
        params["threshold"] = 30;
        QImage out = e.processImage(input, params);
        QVERIFY(!out.isNull());
        QCOMPARE(out.width(),  128);
        QCOMPARE(out.height(), 64);
        int corrected = pixelR(out, 96, 32);
        QVERIFY(corrected < 200);
        QVERIFY(qAbs(corrected - bgVal) <= 5);
        // Pixel in the left half, far from the spike, must stay at bgVal.
        QCOMPARE(pixelR(out, 20, 32), bgVal);
    }

    void meta_nonEmpty() {
        HotPixelEffect e;
        QVERIFY(!e.getName().isEmpty());
        QVERIFY(!e.getDescription().isEmpty());
        QVERIFY(!e.getVersion().isEmpty());
        QVERIFY(e.initialize());
    }

    void defaultParameters_keys() {
        HotPixelEffect e;
        auto params = e.getParameters();
        QVERIFY(params.contains("threshold"));
        QCOMPARE(params["threshold"].toInt(), 30);
    }

    // Uniform 16-bit image: no pixel deviates from its neighbours → unchanged.
    void uniformImage_16bit() {
        if (!m_hasGpu) QSKIP("No GPU");
        HotPixelEffect e;
        QImage input = makeSolid16bit(32, 32, 80, 80, 80);
        QMap<QString, QVariant> params;
        params["threshold"] = 30;
        QImage out = e.processImage(input, params);
        QVERIFY(!out.isNull());
    }

    void createControlsWidget_constructsAndCaches() {
        HotPixelEffect e;
        QWidget* w = e.createControlsWidget();
        QVERIFY(w != nullptr);
        QVERIFY(e.createControlsWidget() == w);
    }

    // Fire signal lambdas for the threshold ParamSlider.
    void connectSlider_signals_coverLambdaBodies() {
        HotPixelEffect e;
        QWidget* w = e.createControlsWidget();
        QVERIFY(w);

        QSignalSpy spyChanged(&e, &PhotoEditorEffect::parametersChanged);
        QSignalSpy spyLive(&e, &PhotoEditorEffect::liveParametersChanged);

        auto sliders = w->findChildren<ParamSlider*>();
        QVERIFY(!sliders.isEmpty());

        for (auto* ps : sliders) {
            auto* qs = ps->findChild<QSlider*>();
            QVERIFY(qs);
            qs->setValue(qs->value() + 1);
            QMetaObject::invokeMethod(qs, "sliderReleased");
        }

        QCOMPARE(spyChanged.count(), 1);
        QCOMPARE(spyLive.count(), 1);
    }

    void supportsGpuInPlace_returnsTrue() {
        HotPixelEffect e;
        QVERIFY(e.supportsGpuInPlace());
    }

    void destructor_heapAllocated_doesNotCrash() {
        auto* e = new HotPixelEffect();
        e->createControlsWidget();
        delete e;
    }
};

QTEST_MAIN(TestHotPixel)
#include "test_hotpixel.moc"
