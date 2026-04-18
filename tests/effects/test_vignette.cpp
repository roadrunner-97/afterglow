#include <QTest>
#include <QSignalSpy>
#include <QSlider>
#include <QWidget>
#include "VignetteEffect.h"
#include "GpuDeviceRegistry.h"
#include "ImageHelpers.h"
#include "ParamSlider.h"

class TestVignette : public QObject {
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
        VignetteEffect e;
        QVERIFY(e.processImage(QImage(), {}).isNull());
    }

    // amount=0 → identity: every pixel unchanged regardless of other params.
    void zeroAmount_isIdentity() {
        VignetteEffect e;
        QImage input = makeSolid(64, 64, 100, 120, 80);
        QMap<QString, QVariant> params;
        params["amount"]    = 0;
        params["midpoint"]  = 50;
        params["feather"]   = 50;
        params["roundness"] = 0;
        QImage out = e.processImage(input, params);
        QVERIFY(!out.isNull());
        QVERIFY(allPixels(out, [](QRgb px) {
            return qRed(px) == 100 && qGreen(px) == 120 && qBlue(px) == 80;
        }));
    }

    // Negative amount darkens corners relative to the centre on a solid image.
    void darkening_cornerDarkerThanCenter() {
        if (!m_hasGpu) QSKIP("No GPU");
        VignetteEffect e;
        QImage input = makeSolid(64, 64, 200, 200, 200);
        QMap<QString, QVariant> params;
        params["amount"]    = -100;
        params["midpoint"]  = 30;
        params["feather"]   = 40;
        params["roundness"] = 0;
        QImage out = e.processImage(input, params);
        QVERIFY(!out.isNull());

        int center = pixelR(out, 32, 32);
        int corner = pixelR(out,  0,  0);
        QVERIFY2(center == 200, qPrintable(QString("centre=%1").arg(center)));
        QVERIFY2(corner <  50,  qPrintable(QString("corner=%1").arg(corner)));
    }

    // Positive amount lightens corners relative to the centre (within clamp).
    void lightening_cornerBrighterThanCenter() {
        if (!m_hasGpu) QSKIP("No GPU");
        VignetteEffect e;
        QImage input = makeSolid(64, 64, 100, 100, 100);
        QMap<QString, QVariant> params;
        params["amount"]    = 100;
        params["midpoint"]  = 30;
        params["feather"]   = 40;
        params["roundness"] = 0;
        QImage out = e.processImage(input, params);
        QVERIFY(!out.isNull());

        int center = pixelR(out, 32, 32);
        int corner = pixelR(out,  0,  0);
        QVERIFY(corner > center);
    }

    // feather=0 gives a near-hard edge: pixels near centre unchanged, pixels past
    // midpoint strongly affected.  midpoint=50 puts the edge halfway to corners.
    void hardEdge_featherZero() {
        if (!m_hasGpu) QSKIP("No GPU");
        VignetteEffect e;
        QImage input = makeSolid(64, 64, 180, 180, 180);
        QMap<QString, QVariant> params;
        params["amount"]    = -100;
        params["midpoint"]  = 50;
        params["feather"]   = 0;
        params["roundness"] = 0;
        QImage out = e.processImage(input, params);
        QVERIFY(!out.isNull());

        // centre untouched
        QCOMPARE(pixelR(out, 32, 32), 180);
        // corners fully darkened
        QVERIFY(pixelR(out, 0, 0) < 10);
    }

    // roundness=-100 pulls the falloff to the frame rectangle: side midpoints
    // darken as much as the corners.  roundness=+100 concentrates darkening in
    // the corners, so the side midpoints stay nearly untouched.
    void roundness_shapesFalloff() {
        if (!m_hasGpu) QSKIP("No GPU");
        VignetteEffect e;
        QMap<QString, QVariant> params;
        params["amount"]    = -100;
        params["midpoint"]  = 20;
        params["feather"]   = 20;

        QImage input = makeSolid(64, 64, 200, 200, 200);
        params["roundness"] = -100;
        QImage rect = e.processImage(input, params);
        params["roundness"] =  100;
        QImage round = e.processImage(input, params);
        QVERIFY(!rect.isNull() && !round.isNull());

        // (0, 32) is the middle of the left edge — on a 64x64 image its
        // normalised coordinate is (|nx|=1, |ny|=0).  Under the rectangle
        // metric this point is at d_norm≈1 (fully darkened); under the round
        // metric it is well inside the falloff (much less darkening).
        int rectSide  = pixelR(rect,  0, 32);
        int roundSide = pixelR(round, 0, 32);
        QVERIFY2(rectSide < roundSide - 20,
                 qPrintable(QString("rectSide=%1 roundSide=%2").arg(rectSide).arg(roundSide)));
    }

    // On a non-square image at roundness=0, the falloff is a circle through
    // the corners (radius = half-diagonal) — NOT an ellipse inscribed in the
    // frame.  Consequence on a 128x64 image: the long-edge midpoint sits at
    // ~0.894 of the way to the corner, while the short-edge midpoint sits at
    // only ~0.447.  With midpoint=80, feather=20 (falloff window 0.7..0.9),
    // the long-edge midpoint is deep inside the falloff (fully darkened) and
    // the short-edge midpoint is below edge0 (entirely untouched).  An
    // inscribed-ellipse metric would put both at ~0.707 and darken them
    // identically.
    void nonSquare_roundnessZero_isCircularThroughCorners() {
        if (!m_hasGpu) QSKIP("No GPU");
        VignetteEffect e;
        QImage input = makeSolid(128, 64, 200, 200, 200);
        QMap<QString, QVariant> params;
        params["amount"]    = -100;
        params["midpoint"]  = 80;
        params["feather"]   = 20;
        params["roundness"] = 0;
        QImage out = e.processImage(input, params);
        QVERIFY(!out.isNull());

        int longEdgeMid  = pixelR(out,  0, 32);   // middle of left edge
        int shortEdgeMid = pixelR(out, 64,  0);   // middle of top edge

        QVERIFY2(longEdgeMid  < 20,  qPrintable(QString("longEdgeMid=%1").arg(longEdgeMid)));
        QVERIFY2(shortEdgeMid > 190, qPrintable(QString("shortEdgeMid=%1").arg(shortEdgeMid)));
    }

    // On a non-square image at roundness=0, darkening depends only on the
    // pixel-space distance from centre: two points at equal radius but on
    // perpendicular axes (30 px left of centre, 30 px above centre) must
    // receive matching darkening.  Under the previous aspect-normalised
    // metric these points mapped to very different d values (0.332 vs 0.663)
    // and darkened unequally.
    void nonSquare_roundnessZero_isIsotropic() {
        if (!m_hasGpu) QSKIP("No GPU");
        VignetteEffect e;
        QImage input = makeSolid(128, 64, 200, 200, 200);
        QMap<QString, QVariant> params;
        params["amount"]    = -100;
        params["midpoint"]  = 30;
        params["feather"]   = 30;
        params["roundness"] = 0;
        QImage out = e.processImage(input, params);
        QVERIFY(!out.isNull());

        int alongLong  = pixelR(out, 34, 32);  // 30 px left of centre (64,32)
        int alongShort = pixelR(out, 64,  2);  // 30 px above centre
        int diff = std::abs(alongLong - alongShort);
        QVERIFY2(diff < 5,
                 qPrintable(QString("alongLong=%1 alongShort=%2 diff=%3")
                            .arg(alongLong).arg(alongShort).arg(diff)));
    }

    // 16-bit path: no crash and output is non-null when amount is non-zero.
    void darkening_16bit() {
        if (!m_hasGpu) QSKIP("No GPU");
        VignetteEffect e;
        QImage input = makeSolid16bit(64, 64, 200, 200, 200);
        QMap<QString, QVariant> params;
        params["amount"]    = -50;
        params["midpoint"]  = 40;
        params["feather"]   = 40;
        params["roundness"] = 0;
        QImage out = e.processImage(input, params);
        QVERIFY(!out.isNull());
    }

    void meta_nonEmpty() {
        VignetteEffect e;
        QVERIFY(!e.getName().isEmpty());
        QVERIFY(!e.getDescription().isEmpty());
        QVERIFY(!e.getVersion().isEmpty());
        QVERIFY(e.initialize());
    }

    void defaultParameters_keysAndValues() {
        VignetteEffect e;
        auto params = e.getParameters();
        QVERIFY(params.contains("amount"));
        QVERIFY(params.contains("midpoint"));
        QVERIFY(params.contains("feather"));
        QVERIFY(params.contains("roundness"));
        // Widget not yet built → fall-through defaults
        QCOMPARE(params["amount"].toInt(),    0);
        QCOMPARE(params["midpoint"].toInt(),  50);
        QCOMPARE(params["feather"].toInt(),   50);
        QCOMPARE(params["roundness"].toInt(), 0);
    }

    void createControlsWidget_constructsAndCaches() {
        VignetteEffect e;
        QWidget* w = e.createControlsWidget();
        QVERIFY(w != nullptr);
        QVERIFY(e.createControlsWidget() == w);
    }

    // After the controls widget is built, midpoint/feather carry their UI defaults.
    void getParameters_afterWidget_returnsUiDefaults() {
        VignetteEffect e;
        e.createControlsWidget();
        auto params = e.getParameters();
        QCOMPARE(params["amount"].toInt(),    0);
        QCOMPARE(params["midpoint"].toInt(),  50);
        QCOMPARE(params["feather"].toInt(),   50);
        QCOMPARE(params["roundness"].toInt(), 0);
    }

    void supportsGpuInPlace_returnsTrue() {
        VignetteEffect e;
        QVERIFY(e.supportsGpuInPlace());
    }

    // Fire every slider's editingFinished + valueChanged path.
    void connectSlider_signals_coverLambdaBodies() {
        VignetteEffect e;
        QWidget* w = e.createControlsWidget();
        QVERIFY(w);

        QSignalSpy spyChanged(&e, &PhotoEditorEffect::parametersChanged);
        QSignalSpy spyLive(&e, &PhotoEditorEffect::liveParametersChanged);

        auto sliders = w->findChildren<ParamSlider*>();
        QCOMPARE(sliders.size(), 4);

        for (auto* ps : sliders) {
            auto* qs = ps->findChild<QSlider*>();
            QVERIFY(qs);
            qs->setValue(qs->value() + 1);
            QMetaObject::invokeMethod(qs, "sliderReleased");
        }

        QVERIFY(spyChanged.count() >= 4);
        QVERIFY(spyLive.count()    >= 4);
    }

    // Heap-allocate so the destructor body is explicitly attributed.
    void destructor_heapAllocated_doesNotCrash() {
        auto* e = new VignetteEffect();
        e->createControlsWidget();
        delete e;
    }
};

QTEST_MAIN(TestVignette)
#include "test_vignette.moc"
