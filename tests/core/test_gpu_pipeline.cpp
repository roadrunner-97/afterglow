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
#include "VignetteEffect.h"
#include "FilmGrainEffect.h"
#include "SplitToningEffect.h"
#include "ClarityEffect.h"
#include "ColorBalanceEffect.h"

// Build a GpuPipelineCall from any effect that derives from both
// PhotoEditorEffect and IGpuEffect.  Saves repeating the effect pointer
// twice on every call site; a non-GPU effect would simply fail to
// compile, since GpuPipelineCall::gpu cannot be null.
template<class T>
static GpuPipelineCall call(T* e, const QMap<QString, QVariant>& p = {}) {
    return {e, e, p};
}

// An IGpuEffect whose initGpuKernels() always returns false.
// Exercises the "initGpuKernels failed" warning path (GpuPipeline.cpp lines 101-103).
class FailInitEffect : public PhotoEditorEffect, public IGpuEffect {
    Q_OBJECT
public:
    QString getName()        const override { return "FailInit"; }
    QString getDescription() const override { return ""; }
    QString getVersion()     const override { return "1.0"; }
    bool    initialize()           override { return true; }
    QImage processImage(const QImage& img, const QMap<QString,QVariant>&) override { return img; }
    bool initGpuKernels(cl::Context&, cl::Device&) override { return false; }
    bool enqueueGpu(cl::CommandQueue&, cl::Buffer&, cl::Buffer&,
                    int, int, const QMap<QString,QVariant>&) override { return true; }
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
    QImage processImage(const QImage& img, const QMap<QString,QVariant>&) override { return img; }
    bool initGpuKernels(cl::Context&, cl::Device&) override { return true; }
    bool enqueueGpu(cl::CommandQueue&, cl::Buffer&, cl::Buffer&,
                    int, int, const QMap<QString,QVariant>&) override { return false; }
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
    VignetteEffect   m_vignette;
    FilmGrainEffect  m_filmgrain;
    SplitToningEffect m_splittoning;
    ClarityEffect    m_clarity;
    ColorBalanceEffect m_colorbalance;

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
        QImage out = m_pipeline.run(input, {call(&m_brightness, p)}, fullViewport(input));
        QVERIFY(!out.isNull());
    }

    void pipeline_saturation() {
        if (!m_hasGpu) QSKIP("No GPU");
        QMap<QString, QVariant> p;
        p["saturation"] = 10.0;
        p["vibrancy"]   = 0.0;
        QImage input = makeSolid(64, 64, 200, 100, 100);
        QImage out = m_pipeline.run(input, {call(&m_saturation, p)}, fullViewport(input));
        QVERIFY(!out.isNull());
    }

    void pipeline_blur() {
        if (!m_hasGpu) QSKIP("No GPU");
        QMap<QString, QVariant> p;
        p["radius"]   = 4;
        p["blurType"] = 0;  // Gaussian
        QImage input = makeSolid(64, 64, 128, 128, 128);
        QImage out = m_pipeline.run(input, {call(&m_blur, p)}, fullViewport(input));
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
        QImage out = m_pipeline.run(input, {call(&m_exposure, p)}, fullViewport(input));
        QVERIFY(!out.isNull());
    }

    // Exposure with all-zero params: enqueueGpu takes the no-op branch
    // (returns true without dispatching a kernel).
    void pipeline_exposure_allZero_noOp() {
        if (!m_hasGpu) QSKIP("No GPU");
        QMap<QString, QVariant> p;
        p["exposure"]   = 0.0;
        p["whites"]     = 0.0;
        p["highlights"] = 0.0;
        p["shadows"]    = 0.0;
        p["blacks"]     = 0.0;
        QImage input = makeSolid(64, 64, 100, 100, 100);
        QImage out = m_pipeline.run(input, {call(&m_exposure, p)}, fullViewport(input));
        QVERIFY(!out.isNull());
    }

    void pipeline_hotpixel() {
        if (!m_hasGpu) QSKIP("No GPU");
        QMap<QString, QVariant> p;
        p["threshold"] = 30;
        QImage input = makeSolid(64, 64, 80, 80, 80);
        QImage out = m_pipeline.run(input, {call(&m_hotpixel, p)}, fullViewport(input));
        QVERIFY(!out.isNull());
    }

    void pipeline_unsharp() {
        if (!m_hasGpu) QSKIP("No GPU");
        QMap<QString, QVariant> p;
        p["amount"]    = 1.0;
        p["radius"]    = 2;
        p["threshold"] = 3;
        QImage input = makeSolid(64, 64, 128, 128, 128);
        QImage out = m_pipeline.run(input, {call(&m_unsharp, p)}, fullViewport(input));
        QVERIFY(!out.isNull());
    }

    // Grayscale inactive (default): enqueueGpu is a no-op but initGpuKernels still runs.
    void pipeline_grayscale_inactive() {
        if (!m_hasGpu) QSKIP("No GPU");
        QImage input = makeSolid(64, 64, 200, 100, 50);
        QImage out = m_pipeline.run(input, {call(&m_grayscale)}, fullViewport(input));
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
        QImage out = m_pipeline.run(input, {call(&m_grayscale)}, fullViewport(input));
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
        QImage out = m_pipeline.run(input, {call(&m_denoise, p)}, fullViewport(input));
        QVERIFY(!out.isNull());
    }

    void pipeline_whitebalance() {
        if (!m_hasGpu) QSKIP("No GPU");
        QMap<QString, QVariant> p;
        p["shot_temp"]   = 5500.0;
        p["temperature"] = 6500.0;
        p["tint"]        = 0.0;
        QImage input = makeSolid(64, 64, 128, 128, 128);
        QImage out = m_pipeline.run(input, {call(&m_whitebalance, p)}, fullViewport(input));
        QVERIFY(!out.isNull());
    }

    // Vignette inactive (amount=0): enqueueGpu early-returns but initGpuKernels runs.
    void pipeline_vignette_inactive() {
        if (!m_hasGpu) QSKIP("No GPU");
        QMap<QString, QVariant> p;
        p["amount"]    = 0;
        p["midpoint"]  = 50;
        p["feather"]   = 50;
        p["roundness"] = 0;
        QImage input = makeSolid(64, 64, 180, 180, 180);
        QImage out = m_pipeline.run(input, {call(&m_vignette, p)}, fullViewport(input));
        QVERIFY(!out.isNull());
    }

    // Vignette active: exercises the enqueueGpu body (kernel dispatch).
    void pipeline_vignette_active() {
        if (!m_hasGpu) QSKIP("No GPU");
        QMap<QString, QVariant> p;
        p["amount"]    = -60;
        p["midpoint"]  = 40;
        p["feather"]   = 40;
        p["roundness"] = 0;
        QImage input = makeSolid(64, 64, 180, 180, 180);
        QImage out = m_pipeline.run(input, {call(&m_vignette, p)}, fullViewport(input));
        QVERIFY(!out.isNull());
    }

    // Film grain inactive (amount=0): enqueueGpu early-returns but initGpuKernels runs.
    void pipeline_filmgrain_inactive() {
        if (!m_hasGpu) QSKIP("No GPU");
        QMap<QString, QVariant> p;
        p["amount"]    = 0;
        p["size"]      = 1;
        p["lumWeight"] = true;
        QImage input = makeSolid(64, 64, 128, 128, 128);
        QImage out = m_pipeline.run(input, {call(&m_filmgrain, p)}, fullViewport(input));
        QVERIFY(!out.isNull());
    }

    // Film grain active: exercises the enqueueGpu body (kernel dispatch).
    void pipeline_filmgrain_active() {
        if (!m_hasGpu) QSKIP("No GPU");
        QMap<QString, QVariant> p;
        p["amount"]    = 50;
        p["size"]      = 2;
        p["lumWeight"] = false;
        QImage input = makeSolid(64, 64, 128, 128, 128);
        QImage out = m_pipeline.run(input, {call(&m_filmgrain, p)}, fullViewport(input));
        QVERIFY(!out.isNull());
    }

    // Split toning inactive (both sats=0): enqueueGpu early-returns but initGpuKernels runs.
    void pipeline_splittoning_inactive() {
        if (!m_hasGpu) QSKIP("No GPU");
        QMap<QString, QVariant> p;
        p["shadowHue"]    = 240;
        p["shadowSat"]    = 0;
        p["highlightHue"] = 60;
        p["highlightSat"] = 0;
        p["balance"]      = 0;
        QImage input = makeSolid(64, 64, 128, 128, 128);
        QImage out = m_pipeline.run(input, {call(&m_splittoning, p)}, fullViewport(input));
        QVERIFY(!out.isNull());
    }

    // Split toning active: exercises the enqueueGpu body (kernel dispatch).
    void pipeline_splittoning_active() {
        if (!m_hasGpu) QSKIP("No GPU");
        QMap<QString, QVariant> p;
        p["shadowHue"]    = 240;
        p["shadowSat"]    = 50;
        p["highlightHue"] = 60;
        p["highlightSat"] = 50;
        p["balance"]      = 0;
        QImage input = makeSolid(64, 64, 128, 128, 128);
        QImage out = m_pipeline.run(input, {call(&m_splittoning, p)}, fullViewport(input));
        QVERIFY(!out.isNull());
    }

    // Clarity inactive (amount=0): enqueueGpu early-returns but initGpuKernels runs.
    void pipeline_clarity_inactive() {
        if (!m_hasGpu) QSKIP("No GPU");
        QMap<QString, QVariant> p;
        p["amount"] = 0;
        p["radius"] = 30;
        QImage input = makeSolid(64, 64, 128, 128, 128);
        QImage out = m_pipeline.run(input, {call(&m_clarity, p)}, fullViewport(input));
        QVERIFY(!out.isNull());
    }

    // Clarity active: exercises H/V blur + combine + copy pipeline path.
    void pipeline_clarity_active() {
        if (!m_hasGpu) QSKIP("No GPU");
        QMap<QString, QVariant> p;
        p["amount"] = 50;
        p["radius"] = 20;
        QImage input = makeSolid(64, 64, 128, 128, 128);
        QImage out = m_pipeline.run(input, {call(&m_clarity, p)}, fullViewport(input));
        QVERIFY(!out.isNull());
    }

    // Color balance inactive (all offsets=0): enqueueGpu early-returns but initGpuKernels runs.
    void pipeline_colorbalance_inactive() {
        if (!m_hasGpu) QSKIP("No GPU");
        QMap<QString, QVariant> p;
        p["shadowR"]    = 0; p["shadowG"]    = 0; p["shadowB"]    = 0;
        p["midtoneR"]   = 0; p["midtoneG"]   = 0; p["midtoneB"]   = 0;
        p["highlightR"] = 0; p["highlightG"] = 0; p["highlightB"] = 0;
        QImage input = makeSolid(64, 64, 128, 128, 128);
        QImage out = m_pipeline.run(input, {call(&m_colorbalance, p)}, fullViewport(input));
        QVERIFY(!out.isNull());
    }

    // Color balance active: exercises the enqueueGpu body (kernel dispatch).
    void pipeline_colorbalance_active() {
        if (!m_hasGpu) QSKIP("No GPU");
        QMap<QString, QVariant> p;
        p["shadowR"]    = 40; p["shadowG"]    = 0;  p["shadowB"]    = -20;
        p["midtoneR"]   = 0;  p["midtoneG"]   = 30; p["midtoneB"]   = 0;
        p["highlightR"] = 0;  p["highlightG"] = 0;  p["highlightB"] = 40;
        QImage input = makeSolid(64, 64, 128, 128, 128);
        QImage out = m_pipeline.run(input, {call(&m_colorbalance, p)}, fullViewport(input));
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

    // Pass an IGpuEffect whose initGpuKernels() always returns false.
    // GpuPipeline::run() should log a warning and return a null image.
    void failInitEffect_warnsAndReturnsNull() {
        if (!m_hasGpu) QSKIP("No GPU");
        FailInitEffect fie;
        QImage input = makeSolid(32, 32, 100, 100, 100);
        QImage out = m_pipeline.run(input, {call(&fie)}, fullViewport(input));
        QVERIFY(out.isNull());
    }

    // Pass an IGpuEffect whose enqueueGpu() always returns false.
    // GpuPipeline::run() should log a warning and return a null image.
    void failEnqueueEffect_warnsAndReturnsNull() {
        if (!m_hasGpu) QSKIP("No GPU");
        FailEnqueueEffect fee;
        QImage input = makeSolid(32, 32, 100, 100, 100);
        QImage out = m_pipeline.run(input, {call(&fee)}, fullViewport(input));
        QVERIFY(out.isNull());
    }

    // PanZoom mode after a Commit: reuses the cached full-res frame, skips effect kernels.
    void panZoom_reusesCachedFrame() {
        if (!m_hasGpu) QSKIP("No GPU");
        QMap<QString, QVariant> p;
        p["brightness"] = 10;
        p["contrast"]   = 0;
        QImage input = makeSolid(64, 64, 128, 128, 128);
        ViewportRequest vp = fullViewport(input);

        // Commit run populates the full-res post-effect cache.
        QImage out1 = m_pipeline.run(input, {call(&m_brightness, p)}, vp, RunMode::Commit);
        QVERIFY(!out1.isNull());

        // PanZoom run: reuses the cache, skips effect kernels.
        QImage out2 = m_pipeline.run(input, {call(&m_brightness, p)}, vp, RunMode::PanZoom);
        QVERIFY(!out2.isNull());
    }

    // LiveDrag mode: bypasses the cache, runs the preview-sized pipeline
    // (decode+downsample srcBuf → workBuf, then effects at preview size).
    // Covers the preview fallback branch of GpuPipeline::run().
    void liveDrag_runsPreviewPipeline() {
        if (!m_hasGpu) QSKIP("No GPU");
        QMap<QString, QVariant> p;
        p["brightness"] = 10;
        p["contrast"]   = 0;
        QImage input = makeSolid(64, 64, 128, 128, 128);
        QImage out = m_pipeline.run(input, {call(&m_brightness, p)}, fullViewport(input),
                                    RunMode::LiveDrag);
        QVERIFY(!out.isNull());
    }

    // 16-bit sRGB input: selects the 16-bit sRGB downsample/decode kernels.
    void liveDrag_16bitSrgbInput() {
        if (!m_hasGpu) QSKIP("No GPU");
        QImage input(32, 32, QImage::Format_RGBX64);
        input.fill(QColor(128, 128, 128));
        QImage out = m_pipeline.run(input, {}, fullViewport(input), RunMode::LiveDrag);
        QVERIFY(!out.isNull());
    }

    // 16-bit linear input (tagged color_space=linear by RawLoader): selects the
    // 16-bit linear decode kernel in both LiveDrag and Commit paths.
    void linearInput_selectsLinearKernels() {
        if (!m_hasGpu) QSKIP("No GPU");
        QImage input(32, 32, QImage::Format_RGBX64);
        input.fill(QColor(128, 128, 128));
        input.setText("color_space", "linear");
        QImage outLive = m_pipeline.run(input, {}, fullViewport(input), RunMode::LiveDrag);
        QVERIFY(!outLive.isNull());
        QImage outCommit = m_pipeline.run(input, {}, fullViewport(input), RunMode::Commit);
        QVERIFY(!outCommit.isNull());
    }
};

QTEST_MAIN(TestGpuPipeline)
#include "test_gpu_pipeline.moc"
