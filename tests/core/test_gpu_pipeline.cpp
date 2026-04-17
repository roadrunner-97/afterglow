#include <QTest>
#include <QCheckBox>
#include "GpuPipeline.h"
#include "GpuDeviceRegistry.h"
#include "BrightnessEffect.h"
#include "SaturationEffect.h"
#include "BlurEffect.h"
#include "ExposureEffect.h"
#include "HotPixelEffect.h"
#include "UnsharpEffect.h"
#include "GrayscaleEffect.h"
#include "DenoiseEffect.h"
#include "WhiteBalanceEffect.h"

// A concrete PhotoEditorEffect that does NOT inherit IGpuEffect.
// Used to exercise the "missing IGpuEffect" warning path in GpuPipeline::run().
class NonGpuEffect : public PhotoEditorEffect {
    Q_OBJECT
public:
    QString getName()        const override { return "NonGpu"; }
    QString getDescription() const override { return ""; }
    QString getVersion()     const override { return "1.0"; }
    bool    initialize()           override { return true; }
    QImage processImage(const QImage& img, const QMap<QString,QVariant>&) override { return img; }
};

// An IGpuEffect whose initGpuKernels() always returns false.
// Exercises the "initGpuKernels failed" warning path (GpuPipeline.cpp lines 101-103).
class FailInitEffect : public PhotoEditorEffect, public IGpuEffect {
    Q_OBJECT
public:
    QString getName()        const override { return "FailInit"; }
    QString getDescription() const override { return ""; }
    QString getVersion()     const override { return "1.0"; }
    bool    initialize()           override { return true; }
    bool    supportsGpuInPlace() const override { return true; }
    QImage processImage(const QImage& img, const QMap<QString,QVariant>&) override { return img; }
    bool initGpuKernels(cl::Context&, cl::Device&) override { return false; }
    bool enqueueGpu(cl::CommandQueue&, cl::Buffer&, cl::Buffer&,
                    int, int, int, bool, const QMap<QString,QVariant>&) override { return true; }
};

// An IGpuEffect whose enqueueGpu() always returns false.
// Exercises the "enqueueGpu() failed" warning path (GpuPipeline.cpp lines 177-179).
class FailEnqueueEffect : public PhotoEditorEffect, public IGpuEffect {
    Q_OBJECT
public:
    QString getName()        const override { return "FailEnqueue"; }
    QString getDescription() const override { return ""; }
    QString getVersion()     const override { return "1.0"; }
    bool    initialize()           override { return true; }
    bool    supportsGpuInPlace() const override { return true; }
    QImage processImage(const QImage& img, const QMap<QString,QVariant>&) override { return img; }
    bool initGpuKernels(cl::Context&, cl::Device&) override { return true; }
    bool enqueueGpu(cl::CommandQueue&, cl::Buffer&, cl::Buffer&,
                    int, int, int, bool, const QMap<QString,QVariant>&) override { return false; }
};

// Effects are class members so their addresses are stable across test methods.
// GpuPipeline::m_initializedEffects tracks pointers — if a local effect goes
// out of scope and a new one lands at the same stack address, the pipeline
// would wrongly skip initGpuKernels().  Stable members prevent that.
class TestGpuPipeline : public QObject {
    Q_OBJECT

private:
    bool m_hasGpu = false;

    GpuPipeline      m_pipeline;
    BrightnessEffect m_brightness;
    SaturationEffect m_saturation;
    BlurEffect       m_blur;
    ExposureEffect   m_exposure;
    HotPixelEffect   m_hotpixel;
    UnsharpEffect    m_unsharp;
    GrayscaleEffect  m_grayscale;
    DenoiseEffect    m_denoise;
    WhiteBalanceEffect m_whitebalance;

    static QImage makeSolid(int w, int h, int r, int g, int b) {
        QImage img(w, h, QImage::Format_RGB32);
        img.fill(qRgb(r, g, b));
        return img;
    }

    static ViewportRequest fullViewport(const QImage& img) {
        ViewportRequest vp;
        vp.displaySize = img.size();
        vp.zoom   = 1.0f;
        vp.center = {0.5, 0.5};
        return vp;
    }

private slots:
    void initTestCase() {
        GpuDeviceRegistry::instance().enumerate();
        if (GpuDeviceRegistry::instance().count() == 0)
            QSKIP("No OpenCL device found — skipping GPU pipeline tests");
        GpuDeviceRegistry::instance().setDevice(0);
        m_hasGpu = true;
    }

    // Empty call list: just runs the preview downsample pass.
    void emptyPipeline_justDownsamples() {
        if (!m_hasGpu) QSKIP("No GPU");
        QImage input = makeSolid(64, 64, 100, 150, 200);
        QImage out = m_pipeline.run(input, {}, fullViewport(input));
        QVERIFY(!out.isNull());
        QCOMPARE(out.width(),  64);
        QCOMPARE(out.height(), 64);
    }

    void pipeline_brightness() {
        if (!m_hasGpu) QSKIP("No GPU");
        QMap<QString, QVariant> p;
        p["brightness"] = 20;
        p["contrast"]   = 0;
        QImage input = makeSolid(64, 64, 100, 100, 100);
        QImage out = m_pipeline.run(input, {{&m_brightness, p}}, fullViewport(input));
        QVERIFY(!out.isNull());
    }

    void pipeline_saturation() {
        if (!m_hasGpu) QSKIP("No GPU");
        QMap<QString, QVariant> p;
        p["saturation"] = 10.0;
        p["vibrancy"]   = 0.0;
        QImage input = makeSolid(64, 64, 200, 100, 100);
        QImage out = m_pipeline.run(input, {{&m_saturation, p}}, fullViewport(input));
        QVERIFY(!out.isNull());
    }

    void pipeline_blur() {
        if (!m_hasGpu) QSKIP("No GPU");
        QMap<QString, QVariant> p;
        p["radius"]   = 4;
        p["blurType"] = 0;  // Gaussian
        QImage input = makeSolid(64, 64, 128, 128, 128);
        QImage out = m_pipeline.run(input, {{&m_blur, p}}, fullViewport(input));
        QVERIFY(!out.isNull());
    }

    void pipeline_exposure() {
        if (!m_hasGpu) QSKIP("No GPU");
        QMap<QString, QVariant> p;
        p["exposure"]   = 0.5;
        p["whites"]     = 0.0;
        p["highlights"] = 0.0;
        p["shadows"]    = 0.0;
        p["blacks"]     = 0.0;
        QImage input = makeSolid(64, 64, 100, 100, 100);
        QImage out = m_pipeline.run(input, {{&m_exposure, p}}, fullViewport(input));
        QVERIFY(!out.isNull());
    }

    void pipeline_hotpixel() {
        if (!m_hasGpu) QSKIP("No GPU");
        QMap<QString, QVariant> p;
        p["threshold"] = 30;
        QImage input = makeSolid(64, 64, 80, 80, 80);
        QImage out = m_pipeline.run(input, {{&m_hotpixel, p}}, fullViewport(input));
        QVERIFY(!out.isNull());
    }

    void pipeline_unsharp() {
        if (!m_hasGpu) QSKIP("No GPU");
        QMap<QString, QVariant> p;
        p["amount"]    = 1.0;
        p["radius"]    = 2;
        p["threshold"] = 3;
        QImage input = makeSolid(64, 64, 128, 128, 128);
        QImage out = m_pipeline.run(input, {{&m_unsharp, p}}, fullViewport(input));
        QVERIFY(!out.isNull());
    }

    // Grayscale inactive (default): enqueueGpu is a no-op but initGpuKernels still runs.
    void pipeline_grayscale_inactive() {
        if (!m_hasGpu) QSKIP("No GPU");
        QImage input = makeSolid(64, 64, 200, 100, 50);
        QImage out = m_pipeline.run(input, {{&m_grayscale, {}}}, fullViewport(input));
        QVERIFY(!out.isNull());
    }

    // Grayscale active: exercises enqueueGpu body (lines 167-173).
    void pipeline_grayscale_active() {
        if (!m_hasGpu) QSKIP("No GPU");
        QWidget* w  = m_grayscale.createControlsWidget();
        auto*    cb = w->findChild<QCheckBox*>();
        QVERIFY(cb);
        cb->setChecked(true);

        QImage input = makeSolid(64, 64, 200, 100, 50);
        QImage out = m_pipeline.run(input, {{&m_grayscale, {}}}, fullViewport(input));
        QVERIFY(!out.isNull());

        cb->setChecked(false);  // reset for subsequent tests
    }

    void pipeline_denoise() {
        if (!m_hasGpu) QSKIP("No GPU");
        QMap<QString, QVariant> p;
        p["strength"]       = 50;
        p["shadowPreserve"] = 30;
        p["colorNoise"]     = 50;
        QImage input = makeSolid(64, 64, 128, 128, 128);
        QImage out = m_pipeline.run(input, {{&m_denoise, p}}, fullViewport(input));
        QVERIFY(!out.isNull());
    }

    void pipeline_whitebalance() {
        if (!m_hasGpu) QSKIP("No GPU");
        QMap<QString, QVariant> p;
        p["shot_temp"]   = 5500.0;
        p["temperature"] = 6500.0;
        p["tint"]        = 0.0;
        QImage input = makeSolid(64, 64, 128, 128, 128);
        QImage out = m_pipeline.run(input, {{&m_whitebalance, p}}, fullViewport(input));
        QVERIFY(!out.isNull());
    }

    // Calling setDevice with a different index bumps the revision counter.
    // The pipeline reinitialises on the next run, which must still succeed.
    void setDevice_switch_pipelineStillWorks() {
        if (!m_hasGpu) QSKIP("No GPU");
        // devices() getter (GpuDeviceRegistry.h line 28)
        QVERIFY(!GpuDeviceRegistry::instance().devices().empty());
        GpuDeviceRegistry::instance().setDevice(1);  // covers setDevice body (lines 77-80)
        GpuDeviceRegistry::instance().setDevice(0);  // restore
        QImage input = makeSolid(64, 64, 128, 128, 128);
        QImage out = m_pipeline.run(input, {}, fullViewport(input));
        QVERIFY(!out.isNull());
    }

    // setDevice with out-of-range index, then re-enumerate: covers the
    // bounds check in GpuDeviceRegistry::enumerate() (line 69).
    void enumerate_afterOutOfRangeDevice_resetsIndex() {
        if (!m_hasGpu) QSKIP("No GPU");
        GpuDeviceRegistry::instance().setDevice(99);   // index far out of range
        GpuDeviceRegistry::instance().enumerate();     // line 69: 99 >= devices.size() → resets to 0
        QCOMPARE(GpuDeviceRegistry::instance().currentIndex(), 0);
        // Pipeline must still work after the re-enumeration.
        QImage input = makeSolid(32, 32, 128, 128, 128);
        QImage out = m_pipeline.run(input, {}, fullViewport(input));
        QVERIFY(!out.isNull());
    }

    // Pass an effect that does NOT implement IGpuEffect.
    // GpuPipeline::run() should log a warning and return a null image.
    void nonGpuEffect_warnsAndReturnsNull() {
        if (!m_hasGpu) QSKIP("No GPU");
        NonGpuEffect nge;
        QImage input = makeSolid(32, 32, 100, 100, 100);
        QImage out = m_pipeline.run(input, {{&nge, {}}}, fullViewport(input));
        QVERIFY(out.isNull());
    }

    // Pass an IGpuEffect whose initGpuKernels() always returns false.
    // GpuPipeline::run() should log a warning and return a null image.
    void failInitEffect_warnsAndReturnsNull() {
        if (!m_hasGpu) QSKIP("No GPU");
        FailInitEffect fie;
        QImage input = makeSolid(32, 32, 100, 100, 100);
        QImage out = m_pipeline.run(input, {{&fie, {}}}, fullViewport(input));
        QVERIFY(out.isNull());
    }

    // Pass an IGpuEffect whose enqueueGpu() always returns false.
    // GpuPipeline::run() should log a warning and return a null image.
    void failEnqueueEffect_warnsAndReturnsNull() {
        if (!m_hasGpu) QSKIP("No GPU");
        FailEnqueueEffect fee;
        QImage input = makeSolid(32, 32, 100, 100, 100);
        QImage out = m_pipeline.run(input, {{&fee, {}}}, fullViewport(input));
        QVERIFY(out.isNull());
    }

    // viewportOnly=true after a full run: skips the effect chain, re-downsamples only.
    void viewportOnly_reusesLastFrame() {
        if (!m_hasGpu) QSKIP("No GPU");
        QMap<QString, QVariant> p;
        p["brightness"] = 10;
        p["contrast"]   = 0;
        QImage input = makeSolid(64, 64, 128, 128, 128);
        ViewportRequest vp = fullViewport(input);

        // Full run to populate the processed frame.
        QImage out1 = m_pipeline.run(input, {{&m_brightness, p}}, vp);
        QVERIFY(!out1.isNull());

        // viewportOnly=true: reuses the processed frame, skips effect kernels.
        QImage out2 = m_pipeline.run(input, {{&m_brightness, p}}, vp, /*viewportOnly=*/true);
        QVERIFY(!out2.isNull());
    }
};

QTEST_MAIN(TestGpuPipeline)
#include "test_gpu_pipeline.moc"
