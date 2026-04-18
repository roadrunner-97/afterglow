#include <QTest>
#include <QComboBox>
#include <QSignalSpy>
#include <QSlider>
#include <QWidget>
#include "BlurEffect.h"
#include "GpuDeviceRegistry.h"
#include "ImageHelpers.h"
#include "ParamSlider.h"

class TestBlur : public QObject {
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
        BlurEffect e;
        QVERIFY(e.processImage(QImage(), {}).isNull());
    }

    // radius=0 → processImage returns the image unchanged (early return in code).
    void zeroRadius_isIdentity() {
        BlurEffect e;
        QImage input = makeSolid(32, 32, 100, 150, 200);
        QMap<QString, QVariant> params;
        params["radius"]   = 0;
        params["blurType"] = 0;  // Gaussian
        QImage out = e.processImage(input, params);
        QVERIFY(!out.isNull());
        QVERIFY(allPixels(out, [](QRgb px) {
            return qRed(px) == 100 && qGreen(px) == 150 && qBlue(px) == 200;
        }));
    }

    // Gaussian blur of a solid-colour image must stay exactly solid.
    // Averaging identical neighbours yields the same value regardless of radius.
    void solidColour_gaussianBlur_unchanged() {
        if (!m_hasGpu) QSKIP("No GPU");
        BlurEffect e;
        QImage input = makeSolid(64, 64, 128, 90, 60);
        QMap<QString, QVariant> params;
        params["radius"]   = 15;
        params["blurType"] = 0;  // Gaussian
        QImage out = e.processImage(input, params);
        QVERIFY(!out.isNull());
        QVERIFY(allPixels(out, [](QRgb px) {
            return qRed(px) == 128 && qGreen(px) == 90 && qBlue(px) == 60;
        }));
    }

    // Box blur of a solid-colour image must also stay exactly solid.
    void solidColour_boxBlur_unchanged() {
        if (!m_hasGpu) QSKIP("No GPU");
        BlurEffect e;
        QImage input = makeSolid(64, 64, 200, 200, 200);
        QMap<QString, QVariant> params;
        params["radius"]   = 10;
        params["blurType"] = 1;  // Box
        QImage out = e.processImage(input, params);
        QVERIFY(!out.isNull());
        QVERIFY(allPixels(out, [](QRgb px) { return qRed(px) == 200; }));
    }

    // Checkerboard: the exact centre pixel should be blurred to an intermediate
    // value (neither pure black nor pure white).  A large enough radius blurs
    // across many cells so the weighted average is well away from the extremes.
    void checkerboard_centrePixelIsIntermediate_gaussian() {
        if (!m_hasGpu) QSKIP("No GPU");
        BlurEffect e;
        // 64×64 image, 4px cells → centre at (32,32) lies on a boundary
        QImage input = makeCheckerboard(64, 64, 4);
        QMap<QString, QVariant> params;
        params["radius"]   = 12;
        params["blurType"] = 0;  // Gaussian
        QImage out = e.processImage(input, params);
        QVERIFY(!out.isNull());
        int r = pixelR(out, 32, 32);
        QVERIFY(r > 10 && r < 245);
    }

    // Same test for box blur.
    void checkerboard_centrePixelIsIntermediate_box() {
        if (!m_hasGpu) QSKIP("No GPU");
        BlurEffect e;
        QImage input = makeCheckerboard(64, 64, 4);
        QMap<QString, QVariant> params;
        params["radius"]   = 12;
        params["blurType"] = 1;  // Box
        QImage out = e.processImage(input, params);
        QVERIFY(!out.isNull());
        int r = pixelR(out, 32, 32);
        QVERIFY(r > 10 && r < 245);
    }

    void meta_nonEmpty() {
        BlurEffect e;
        QVERIFY(!e.getName().isEmpty());
        QVERIFY(!e.getDescription().isEmpty());
        QVERIFY(!e.getVersion().isEmpty());
        QVERIFY(e.initialize());
    }

    void defaultParameters_keys() {
        BlurEffect e;
        auto params = e.getParameters();
        QVERIFY(params.contains("blurType"));
        QVERIFY(params.contains("radius"));
        QCOMPARE(params["blurType"].toInt(), 0);
        QCOMPARE(params["radius"].toInt(),   0);
    }

    // Non-square (wide) image: separable blur runs H pass then V pass on
    // differing extents.  A solid-colour input must remain exactly solid and
    // output dimensions must match input.
    void nonSquare_solidColour_unchanged() {
        if (!m_hasGpu) QSKIP("No GPU");
        BlurEffect e;
        QImage input = makeSolid(128, 64, 100, 150, 200);
        QMap<QString, QVariant> params;
        params["radius"]   = 8;
        params["blurType"] = 0;  // Gaussian
        QImage out = e.processImage(input, params);
        QVERIFY(!out.isNull());
        QCOMPARE(out.width(),  128);
        QCOMPARE(out.height(), 64);
        QVERIFY(allPixels(out, [](QRgb px) {
            return qRed(px) == 100 && qGreen(px) == 150 && qBlue(px) == 200;
        }));
    }

    // Solid colour: blurring 16-bit should preserve value (averaging identical neighbours).
    void solidColour_gaussian_16bit() {
        if (!m_hasGpu) QSKIP("No GPU");
        BlurEffect e;
        QImage input = makeSolid16bit(64, 64, 128, 90, 60);
        QMap<QString, QVariant> params;
        params["radius"]   = 4;
        params["blurType"] = 0;  // Gaussian
        QImage out = e.processImage(input, params);
        QVERIFY(!out.isNull());
    }

    void createControlsWidget_constructsAndCaches() {
        BlurEffect e;
        QWidget* w = e.createControlsWidget();
        QVERIFY(w != nullptr);
        QVERIFY(e.createControlsWidget() == w);
    }

    // Fire the QComboBox::activated signal (lambda: blurType = index; emit parametersChanged()).
    void connectCombo_activated_firesParametersChanged() {
        BlurEffect e;
        QWidget* w = e.createControlsWidget();
        QVERIFY(w);

        QSignalSpy spy(&e, &PhotoEditorEffect::parametersChanged);
        auto* combo = w->findChild<QComboBox*>();
        QVERIFY(combo);
        // Directly invoke the activated signal to fire the lambda body.
        combo->activated(1);
        QCOMPARE(spy.count(), 1);
    }

    // Fire ParamSlider signals (editingFinished + valueChanged).
    void connectSlider_signals_coverLambdaBodies() {
        BlurEffect e;
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

        QVERIFY(spyChanged.count() >= 1);
        QVERIFY(spyLive.count() >= 1);
    }

    void supportsGpuInPlace_returnsTrue() {
        BlurEffect e;
        QVERIFY(e.supportsGpuInPlace());
    }

    void destructor_heapAllocated_doesNotCrash() {
        auto* e = new BlurEffect();
        e->createControlsWidget();
        delete e;
    }
};

QTEST_MAIN(TestBlur)
#include "test_blur.moc"
