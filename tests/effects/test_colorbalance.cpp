#include <QTest>
#include <QSignalSpy>
#include <QSlider>
#include <QWidget>
#include "ColorBalanceEffect.h"
#include "GpuDeviceRegistry.h"
#include "ImageHelpers.h"
#include "ParamSlider.h"

class TestColorBalance : public QObject {
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
        ColorBalanceEffect e;
        QVERIFY(e.processImage(QImage(), {}).isNull());
    }

    // All sliders at 0 → identity.
    void allZero_isIdentity() {
        ColorBalanceEffect e;
        QImage input = makeSolid(64, 64, 100, 120, 80);
        QMap<QString, QVariant> params;
        params["shadowR"] = 0;  params["shadowG"] = 0;  params["shadowB"] = 0;
        params["midtoneR"] = 0; params["midtoneG"] = 0; params["midtoneB"] = 0;
        params["highlightR"] = 0; params["highlightG"] = 0; params["highlightB"] = 0;
        QImage out = e.processImage(input, params);
        QVERIFY(!out.isNull());
        QVERIFY(allPixels(out, [](QRgb px) {
            return qRed(px) == 100 && qGreen(px) == 120 && qBlue(px) == 80;
        }));
    }

    // Shadow-red offset on a near-black pixel (L≈0 → shadowW=1) lifts the red
    // channel noticeably above green/blue.
    void shadowOffset_biasesDarkPixels() {
        if (!m_hasGpu) QSKIP("No GPU");
        ColorBalanceEffect e;
        QImage input = makeSolid(64, 64, 20, 20, 20);  // very dark grey
        QMap<QString, QVariant> p;
        p["shadowR"] = 100;  // full +R on shadows
        QImage out = e.processImage(input, p);
        QVERIFY(!out.isNull());

        int r = pixelR(out, 32, 32);
        int g = pixelG(out, 32, 32);
        int b = pixelB(out, 32, 32);
        // +0.25 on R ≈ +64 on an 8-bit channel; G/B untouched.
        QVERIFY2(r > g + 40 && r > b + 40,
                 qPrintable(QString("r=%1 g=%2 b=%3").arg(r).arg(g).arg(b)));
        QCOMPARE(g, 20);
        QCOMPARE(b, 20);
    }

    // Highlight-blue offset on a near-white pixel (L≈1 → highlightW=1) does
    // not raise blue (already 255) but leaves R/G untouched.  Use a mid-
    // brighter pixel where the offset has room to act.
    void highlightOffset_biasesBrightPixels() {
        if (!m_hasGpu) QSKIP("No GPU");
        ColorBalanceEffect e;
        QImage input = makeSolid(64, 64, 180, 180, 180);
        QMap<QString, QVariant> p;
        p["highlightB"] = 100;
        QImage out = e.processImage(input, p);
        QVERIFY(!out.isNull());

        int r = pixelR(out, 32, 32);
        int g = pixelG(out, 32, 32);
        int b = pixelB(out, 32, 32);
        // L = 180/255 ≈ 0.706 → highlightW ≈ 0.41 → +0.25*0.41*255 ≈ +26
        QVERIFY2(b > r + 15 && b > g + 15,
                 qPrintable(QString("r=%1 g=%2 b=%3").arg(r).arg(g).arg(b)));
    }

    // Shadow offset should barely touch pure white pixels (shadowW=0).
    void shadowOffset_leavesBrightPixelsAlone() {
        if (!m_hasGpu) QSKIP("No GPU");
        ColorBalanceEffect e;
        QImage input = makeSolid(64, 64, 250, 250, 250);
        QMap<QString, QVariant> p;
        p["shadowR"] = 100;
        QImage out = e.processImage(input, p);
        QVERIFY(!out.isNull());

        int r = pixelR(out, 32, 32);
        int g = pixelG(out, 32, 32);
        int b = pixelB(out, 32, 32);
        // L ≈ 0.98 → shadowW ≈ 0 (saturated at 0 by max()).
        QVERIFY2(std::abs(r - 250) < 5 && g == 250 && b == 250,
                 qPrintable(QString("r=%1 g=%2 b=%3").arg(r).arg(g).arg(b)));
    }

    // Midtone offset should leave pure-black pixels unchanged (midtoneW=0).
    void midtoneOffset_leavesBlackAlone() {
        if (!m_hasGpu) QSKIP("No GPU");
        ColorBalanceEffect e;
        QImage input = makeSolid(64, 64, 0, 0, 0);
        QMap<QString, QVariant> p;
        p["midtoneG"] = 100;
        QImage out = e.processImage(input, p);
        QVERIFY(!out.isNull());
        QCOMPARE(pixelR(out, 32, 32), 0);
        QCOMPARE(pixelG(out, 32, 32), 0);
        QCOMPARE(pixelB(out, 32, 32), 0);
    }

    // Midtone +G on a midtone grey (L=0.5, midtoneW=1) visibly lifts green
    // above red/blue.
    void midtoneOffset_biasesMidtones() {
        if (!m_hasGpu) QSKIP("No GPU");
        ColorBalanceEffect e;
        QImage input = makeSolid(64, 64, 128, 128, 128);
        QMap<QString, QVariant> p;
        p["midtoneG"] = 100;
        QImage out = e.processImage(input, p);
        QVERIFY(!out.isNull());

        int r = pixelR(out, 32, 32);
        int g = pixelG(out, 32, 32);
        int b = pixelB(out, 32, 32);
        QVERIFY2(g > r + 40 && g > b + 40,
                 qPrintable(QString("r=%1 g=%2 b=%3").arg(r).arg(g).arg(b)));
    }

    // Negative offset decreases the channel.
    void negativeOffset_lowersChannel() {
        if (!m_hasGpu) QSKIP("No GPU");
        ColorBalanceEffect e;
        QImage input = makeSolid(64, 64, 128, 128, 128);
        QMap<QString, QVariant> p;
        p["midtoneR"] = -100;
        QImage out = e.processImage(input, p);
        QVERIFY(!out.isNull());

        int r = pixelR(out, 32, 32);
        int g = pixelG(out, 32, 32);
        int b = pixelB(out, 32, 32);
        QVERIFY2(r < g - 40 && r < b - 40,
                 qPrintable(QString("r=%1 g=%2 b=%3").arg(r).arg(g).arg(b)));
    }

    // Non-square (tall) image: shadow offset stays luminance-masked per-pixel.
    void nonSquare_shadowOffset_biasesDarkPixels() {
        if (!m_hasGpu) QSKIP("No GPU");
        ColorBalanceEffect e;
        QImage input = makeSolid(64, 128, 20, 20, 20);
        QMap<QString, QVariant> p;
        p["shadowR"] = 100;
        QImage out = e.processImage(input, p);
        QVERIFY(!out.isNull());
        QCOMPARE(out.width(),  64);
        QCOMPARE(out.height(), 128);
        int r = pixelR(out, 32, 64);
        int g = pixelG(out, 32, 64);
        int b = pixelB(out, 32, 64);
        QVERIFY2(r > g + 40 && r > b + 40,
                 qPrintable(QString("r=%1 g=%2 b=%3").arg(r).arg(g).arg(b)));
    }

    // 16-bit path: non-null output when offsets are non-zero.
    void offset_16bit() {
        if (!m_hasGpu) QSKIP("No GPU");
        ColorBalanceEffect e;
        QImage input = makeSolid16bit(64, 64, 128, 128, 128);
        QMap<QString, QVariant> p;
        p["midtoneR"] = 50;
        p["midtoneB"] = -50;
        QImage out = e.processImage(input, p);
        QVERIFY(!out.isNull());
    }

    void meta_nonEmpty() {
        ColorBalanceEffect e;
        QVERIFY(!e.getName().isEmpty());
        QVERIFY(!e.getDescription().isEmpty());
        QVERIFY(!e.getVersion().isEmpty());
        QVERIFY(e.initialize());
    }

    void defaultParameters_keysAndValues() {
        ColorBalanceEffect e;
        auto params = e.getParameters();
        for (const char* k : {"shadowR", "shadowG", "shadowB",
                               "midtoneR", "midtoneG", "midtoneB",
                               "highlightR", "highlightG", "highlightB"}) {
            QVERIFY(params.contains(k));
            QCOMPARE(params[k].toInt(), 0);
        }
    }

    void createControlsWidget_constructsAndCaches() {
        ColorBalanceEffect e;
        QWidget* w = e.createControlsWidget();
        QVERIFY(w != nullptr);
        QVERIFY(e.createControlsWidget() == w);
    }

    void supportsGpuInPlace_returnsTrue() {
        ColorBalanceEffect e;
        QVERIFY(e.supportsGpuInPlace());
    }

    // Fire every slider's editingFinished + valueChanged path.
    void connectSlider_signals_coverLambdaBodies() {
        ColorBalanceEffect e;
        QWidget* w = e.createControlsWidget();
        QVERIFY(w);

        QSignalSpy spyChanged(&e, &PhotoEditorEffect::parametersChanged);
        QSignalSpy spyLive(&e, &PhotoEditorEffect::liveParametersChanged);

        auto sliders = w->findChildren<ParamSlider*>();
        QCOMPARE(sliders.size(), 9);  // 3 bands × 3 channels

        for (auto* ps : sliders) {
            auto* qs = ps->findChild<QSlider*>();
            QVERIFY(qs);
            qs->setValue(qs->value() + 1);
            QMetaObject::invokeMethod(qs, "sliderReleased");
        }

        QVERIFY(spyChanged.count() >= 9);
        QVERIFY(spyLive.count()    >= 9);
    }

    // All-zero params should short-circuit before GPU dispatch — ensure it
    // still returns the original unchanged.
    void enqueueGpu_allZero_shortCircuits() {
        ColorBalanceEffect e;
        QImage input = makeSolid(32, 32, 42, 99, 200);
        QImage out = e.processImage(input, {});
        QVERIFY(!out.isNull());
        QCOMPARE(pixelR(out, 0, 0), 42);
        QCOMPARE(pixelG(out, 0, 0), 99);
        QCOMPARE(pixelB(out, 0, 0), 200);
    }

    void destructor_heapAllocated_doesNotCrash() {
        auto* e = new ColorBalanceEffect();
        e->createControlsWidget();
        delete e;
    }
};

QTEST_MAIN(TestColorBalance)
#include "test_colorbalance.moc"
