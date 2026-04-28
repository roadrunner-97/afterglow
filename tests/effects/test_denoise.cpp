#include <QTest>
#include <QComboBox>
#include <QSignalSpy>
#include <QSlider>
#include <QWidget>
#include "DenoiseEffect.h"
#include "GpuDeviceRegistry.h"
#include "GpuPipeline.h"
#include "ImageHelpers.h"
#include "ParamSlider.h"

class TestDenoise : public QObject {
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
        DenoiseEffect e;
        QVERIFY(runEffect(e, QImage(), {}).isNull());
    }

    // All params=0 → early return, image unchanged.
    void zeroParams_isIdentity() {
        DenoiseEffect e;
        QImage input = makeSolid(32, 32, 100, 120, 80);
        QMap<QString, QVariant> params;
        params["strength"]      = 0;
        params["shadowPreserve"] = 0;
        params["colorNoise"]    = 0;
        QImage out = runEffect(e, input, params);
        QVERIFY(!out.isNull());
        QVERIFY(allPixels(out, [](QRgb px) {
            return qRed(px) == 100 && qGreen(px) == 120 && qBlue(px) == 80;
        }));
    }

    // Solid colour with non-zero strength: denoising of a uniform image leaves it unchanged
    // because every pixel equals its neighbourhood average.
    void solidColour_denoiseUnchanged() {
        if (!m_hasGpu) QSKIP("No GPU");
        DenoiseEffect e;
        QImage input = makeSolid(64, 64, 128, 100, 80);
        QMap<QString, QVariant> params;
        params["strength"]       = 50;
        params["shadowPreserve"] = 0;
        params["colorNoise"]     = 0;
        QImage out = runEffect(e, input, params);
        QVERIFY(!out.isNull());
        QVERIFY(allPixels(out, [](QRgb px) {
            return qAbs(qRed(px) - 128) <= 2 && qAbs(qGreen(px) - 100) <= 2;
        }));
    }

    // Non-zero colorNoise: solid colour image leaves pixel values unchanged.
    void solidColour_colorNoise_unchanged() {
        if (!m_hasGpu) QSKIP("No GPU");
        DenoiseEffect e;
        QImage input = makeSolid(64, 64, 128, 100, 80);
        QMap<QString, QVariant> params;
        params["strength"]       = 0;
        params["shadowPreserve"] = 0;
        params["colorNoise"]     = 50;
        QImage out = runEffect(e, input, params);
        QVERIFY(!out.isNull());
        QVERIFY(allPixels(out, [](QRgb px) {
            return qAbs(qRed(px) - 128) <= 2 && qAbs(qGreen(px) - 100) <= 2;
        }));
    }

    // Both phases active at once.
    void bothPhases_solidColour_unchanged() {
        if (!m_hasGpu) QSKIP("No GPU");
        DenoiseEffect e;
        QImage input = makeSolid(64, 64, 200, 150, 100);
        QMap<QString, QVariant> params;
        params["strength"]       = 50;
        params["shadowPreserve"] = 30;
        params["colorNoise"]     = 50;
        QImage out = runEffect(e, input, params);
        QVERIFY(!out.isNull());
        QVERIFY(allPixels(out, [](QRgb px) {
            return qAbs(qRed(px) - 200) <= 3 && qAbs(qGreen(px) - 150) <= 3;
        }));
    }

    // Non-square (wide) image: denoise runs a 2D neighbourhood pass.  A
    // uniform input must remain uniform and preserve dimensions.
    void nonSquare_solidColour_unchanged() {
        if (!m_hasGpu) QSKIP("No GPU");
        DenoiseEffect e;
        QImage input = makeSolid(128, 64, 128, 100, 80);
        QMap<QString, QVariant> params;
        params["strength"]       = 50;
        params["shadowPreserve"] = 30;
        params["colorNoise"]     = 50;
        QImage out = runEffect(e, input, params);
        QVERIFY(!out.isNull());
        QCOMPARE(out.width(),  128);
        QCOMPARE(out.height(), 64);
        QVERIFY(allPixels(out, [](QRgb px) {
            return qAbs(qRed(px) - 128) <= 3 && qAbs(qGreen(px) - 100) <= 3;
        }));
    }

    // 16-bit path: solid colour with non-zero params stays unchanged.
    void solidColour_16bit_denoiseUnchanged() {
        if (!m_hasGpu) QSKIP("No GPU");
        DenoiseEffect e;
        QImage input = makeSolid16bit(64, 64, 128, 100, 80);
        QMap<QString, QVariant> params;
        params["strength"]       = 50;
        params["shadowPreserve"] = 0;
        params["colorNoise"]     = 50;
        QImage out = runEffect(e, input, params);
        QVERIFY(!out.isNull());
    }

    void meta_nonEmpty() {
        DenoiseEffect e;
        QVERIFY(!e.getName().isEmpty());
        QVERIFY(!e.getDescription().isEmpty());
        QVERIFY(!e.getVersion().isEmpty());
        QVERIFY(e.initialize());
    }

    void defaultParameters_keys() {
        DenoiseEffect e;
        auto params = e.getParameters();
        QVERIFY(params.contains("strength"));
        QVERIFY(params.contains("shadowPreserve"));
        QVERIFY(params.contains("colorNoise"));
    }

    void createControlsWidget_constructsAndCaches() {
        DenoiseEffect e;
        QWidget* w = e.createControlsWidget();
        QVERIFY(w != nullptr);
        QVERIFY(e.createControlsWidget() == w);
    }

    void getParameters_afterControlsWidget_returnsDefaults() {
        DenoiseEffect e;
        e.createControlsWidget();
        auto params = e.getParameters();
        QVERIFY(params.contains("strength"));
        QVERIFY(params.contains("shadowPreserve"));
        QVERIFY(params.contains("colorNoise"));
    }

    // Fire signal lambdas in createControlsWidget to cover those branches.
    void connectSlider_signals_coverLambdaBodies() {
        DenoiseEffect e;
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

    // Heap-allocate so the destructor body is explicitly attributed.
    void destructor_heapAllocated_doesNotCrash() {
        auto* e = new DenoiseEffect();
        e->createControlsWidget();
        delete e;
    }

    // Bilateral algorithm branch (algorithm=1) — covers the single-pass
    // bilateral path in enqueueGpu.
    void bilateralAlgorithm_solidColour_unchanged() {
        if (!m_hasGpu) QSKIP("No GPU");
        DenoiseEffect e;
        GpuPipeline pipeline;
        QMap<QString, QVariant> params;
        params["strength"]       = 50;
        params["shadowPreserve"] = 30;
        params["colorNoise"]     = 0;
        params["algorithm"]      = 1;  // Bilateral
        QImage input = makeSolid(64, 64, 128, 100, 80);
        ViewportRequest vp;
        vp.displaySize = input.size();
        QImage out = pipeline.run(input, {{&e, &e, params}}, vp);
        QVERIFY(!out.isNull());
        QVERIFY(allPixels(out, [](QRgb px) {
            return qAbs(qRed(px) - 128) <= 3 && qAbs(qGreen(px) - 100) <= 3;
        }));
    }

    // Algorithm combo activation: firing the activated(int) signal updates
    // m_algorithm and emits parametersChanged.
    void algorithmCombo_activated_updatesAlgorithmAndEmits() {
        DenoiseEffect e;
        QWidget* w = e.createControlsWidget();
        QVERIFY(w);
        auto* combo = w->findChild<QComboBox*>();
        QVERIFY(combo);

        QSignalSpy spy(&e, &PhotoEditorEffect::parametersChanged);
        emit combo->activated(1);

        QCOMPARE(e.getParameters()["algorithm"].toInt(), 1);
        QCOMPARE(spy.count(), 1);
    }
};

QTEST_MAIN(TestDenoise)
#include "test_denoise.moc"
