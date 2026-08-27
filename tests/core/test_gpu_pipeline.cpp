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
template <class T> static GpuPipelineCall call(T *e, const QMap<QString, QVariant> &p = {}) {
    return {e, e, p};
}

// An IGpuEffect whose initGpuKernels() always returns false.
// Exercises the "initGpuKernels failed" warning path (GpuPipeline.cpp lines 101-103).
class FailInitEffect : public PhotoEditorEffect, public IGpuEffect {
    Q_OBJECT
public:
    QString getName() const override {
        return "FailInit";
    }
    QString getDescription() const override {
        return "";
    }
    QString getVersion() const override {
        return "1.0";
    }
    bool initialize() override {
        return true;
    }
    bool initGpuKernels(cl::Context &, cl::Device &) override {
        return false;
    }
    bool enqueueGpu(cl::CommandQueue &, cl::Buffer &, cl::Buffer &, int, int,
                    const QMap<QString, QVariant> &) override {
        return true;
    }
};

// An IGpuEffect whose enqueueGpu() always returns false.
// Exercises the "enqueueGpu() failed" warning path (GpuPipeline.cpp lines 177-179).
class FailEnqueueEffect : public PhotoEditorEffect, public IGpuEffect {
    Q_OBJECT
public:
    QString getName() const override {
        return "FailEnqueue";
    }
    QString getDescription() const override {
        return "";
    }
    QString getVersion() const override {
        return "1.0";
    }
    bool initialize() override {
        return true;
    }
    bool initGpuKernels(cl::Context &, cl::Device &) override {
        return true;
    }
    bool enqueueGpu(cl::CommandQueue &, cl::Buffer &, cl::Buffer &, int, int,
                    const QMap<QString, QVariant> &) override {
        return false;
    }
};

// Effects are class members so their addresses are stable across test methods.
// GpuPipeline::m_initializedEffects tracks pointers — if a local effect goes
// out of scope and a new one lands at the same stack address, the pipeline
// would wrongly skip initGpuKernels().  Stable members prevent that.
class TestGpuPipeline : public QObject {
    Q_OBJECT

private:
    bool m_hasGpu = false;

    GpuPipeline        m_pipeline;
    BrightnessEffect   m_brightness;
    SaturationEffect   m_saturation;
    BlurEffect         m_blur;
    ExposureEffect     m_exposure;
    HotPixelEffect     m_hotpixel;
    UnsharpEffect      m_unsharp;
    GrayscaleEffect    m_grayscale;
    DenoiseEffect      m_denoise;
    WhiteBalanceEffect m_whitebalance;
    VignetteEffect     m_vignette;
    FilmGrainEffect    m_filmgrain;
    SplitToningEffect  m_splittoning;
    ClarityEffect      m_clarity;
    ColorBalanceEffect m_colorbalance;

    static QImage makeSolid(int w, int h, int r, int g, int b) {
        QImage img(w, h, QImage::Format_RGB32);
        img.fill(qRgb(r, g, b));
        return img;
    }

    static ViewportRequest fullViewport(const QImage &img) {
        ViewportRequest vp;
        vp.displaySize = img.size();
        vp.zoom        = 1.0f;
        vp.center      = {0.5, 0.5};
        return vp;
    }

private slots:
    void initTestCase() {
        GpuDeviceRegistry::instance().enumerate();
        if (GpuDeviceRegistry::instance().count() == 0) QSKIP("No OpenCL device found — skipping GPU pipeline tests");
        GpuDeviceRegistry::instance().setDevice(0);
        m_hasGpu = true;
    }

    // Empty call list: just runs the preview downsample pass.
    void emptyPipeline_justDownsamples() {
        if (!m_hasGpu) QSKIP("No GPU");
        QImage input = makeSolid(64, 64, 100, 150, 200);
        QImage out   = m_pipeline.run(input, {}, fullViewport(input)).image;
        QVERIFY(!out.isNull());
        QCOMPARE(out.width(), 64);
        QCOMPARE(out.height(), 64);
    }

    void mutatedSameSizeImage_isReuploaded() {
        if (!m_hasGpu) QSKIP("No GPU");
        QImage          input = makeSolid(32, 32, 240, 10, 10);
        ViewportRequest vp    = fullViewport(input);

        const QImage redOutput = m_pipeline.run(input, {}, vp, RunMode::Commit).image;
        input.fill(qRgb(10, 10, 240));
        const QImage blueOutput = m_pipeline.run(input, {}, vp, RunMode::Commit).image;

        QVERIFY(!redOutput.isNull());
        QVERIFY(!blueOutput.isNull());
        const QColor redPixel  = redOutput.pixelColor(0, 0);
        const QColor bluePixel = blueOutput.pixelColor(0, 0);
        QVERIFY(redPixel.red() > redPixel.blue());
        QVERIFY(bluePixel.blue() > bluePixel.red());
    }

    void pipeline_brightness() {
        if (!m_hasGpu) QSKIP("No GPU");
        QMap<QString, QVariant> p;
        p["brightness"] = 20;
        p["contrast"]   = 0;
        QImage input    = makeSolid(64, 64, 100, 100, 100);
        QImage out      = m_pipeline.run(input, {call(&m_brightness, p)}, fullViewport(input)).image;
        QVERIFY(!out.isNull());
    }

    void pipeline_saturation() {
        if (!m_hasGpu) QSKIP("No GPU");
        QMap<QString, QVariant> p;
        p["saturation"] = 10.0;
        p["vibrancy"]   = 0.0;
        QImage input    = makeSolid(64, 64, 200, 100, 100);
        QImage out      = m_pipeline.run(input, {call(&m_saturation, p)}, fullViewport(input)).image;
        QVERIFY(!out.isNull());
    }

    void pipeline_blur() {
        if (!m_hasGpu) QSKIP("No GPU");
        QMap<QString, QVariant> p;
        p["radius"]   = 4;
        p["blurType"] = 0; // Gaussian
        QImage input  = makeSolid(64, 64, 128, 128, 128);
        QImage out    = m_pipeline.run(input, {call(&m_blur, p)}, fullViewport(input)).image;
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
        QImage input    = makeSolid(64, 64, 100, 100, 100);
        QImage out      = m_pipeline.run(input, {call(&m_exposure, p)}, fullViewport(input)).image;
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
        QImage input    = makeSolid(64, 64, 100, 100, 100);
        QImage out      = m_pipeline.run(input, {call(&m_exposure, p)}, fullViewport(input)).image;
        QVERIFY(!out.isNull());
    }

    void pipeline_hotpixel() {
        if (!m_hasGpu) QSKIP("No GPU");
        QMap<QString, QVariant> p;
        p["threshold"] = 30;
        QImage input   = makeSolid(64, 64, 80, 80, 80);
        QImage out     = m_pipeline.run(input, {call(&m_hotpixel, p)}, fullViewport(input)).image;
        QVERIFY(!out.isNull());
    }

    void pipeline_unsharp() {
        if (!m_hasGpu) QSKIP("No GPU");
        QMap<QString, QVariant> p;
        p["amount"]    = 1.0;
        p["radius"]    = 2;
        p["threshold"] = 3;
        QImage input   = makeSolid(64, 64, 128, 128, 128);
        QImage out     = m_pipeline.run(input, {call(&m_unsharp, p)}, fullViewport(input)).image;
        QVERIFY(!out.isNull());
    }

    // Grayscale inactive (default): enqueueGpu is a no-op but initGpuKernels still runs.
    void pipeline_grayscale_inactive() {
        if (!m_hasGpu) QSKIP("No GPU");
        QImage input = makeSolid(64, 64, 200, 100, 50);
        QImage out   = m_pipeline.run(input, {call(&m_grayscale)}, fullViewport(input)).image;
        QVERIFY(!out.isNull());
    }

    // Grayscale active: exercises enqueueGpu body (lines 167-173).
    void pipeline_grayscale_active() {
        if (!m_hasGpu) QSKIP("No GPU");
        QWidget *w  = m_grayscale.createControlsWidget();
        auto    *cb = w->findChild<QCheckBox *>();
        QVERIFY(cb);
        cb->setChecked(true);

        QImage input = makeSolid(64, 64, 200, 100, 50);
        QImage out   = m_pipeline.run(input, {call(&m_grayscale)}, fullViewport(input)).image;
        QVERIFY(!out.isNull());

        cb->setChecked(false); // reset for subsequent tests
    }

    void pipeline_denoise() {
        if (!m_hasGpu) QSKIP("No GPU");
        QMap<QString, QVariant> p;
        p["strength"]       = 50;
        p["shadowPreserve"] = 30;
        p["colorNoise"]     = 50;
        QImage input        = makeSolid(64, 64, 128, 128, 128);
        QImage out          = m_pipeline.run(input, {call(&m_denoise, p)}, fullViewport(input)).image;
        QVERIFY(!out.isNull());
    }

    void pipeline_whitebalance() {
        if (!m_hasGpu) QSKIP("No GPU");
        QMap<QString, QVariant> p;
        p["shot_temp"]   = 5500.0;
        p["temperature"] = 6500.0;
        p["tint"]        = 0.0;
        QImage input     = makeSolid(64, 64, 128, 128, 128);
        QImage out       = m_pipeline.run(input, {call(&m_whitebalance, p)}, fullViewport(input)).image;
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
        QImage input   = makeSolid(64, 64, 180, 180, 180);
        QImage out     = m_pipeline.run(input, {call(&m_vignette, p)}, fullViewport(input)).image;
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
        QImage input   = makeSolid(64, 64, 180, 180, 180);
        QImage out     = m_pipeline.run(input, {call(&m_vignette, p)}, fullViewport(input)).image;
        QVERIFY(!out.isNull());
    }

    // Film grain inactive (amount=0): enqueueGpu early-returns but initGpuKernels runs.
    void pipeline_filmgrain_inactive() {
        if (!m_hasGpu) QSKIP("No GPU");
        QMap<QString, QVariant> p;
        p["amount"]    = 0;
        p["size"]      = 1;
        p["lumWeight"] = true;
        QImage input   = makeSolid(64, 64, 128, 128, 128);
        QImage out     = m_pipeline.run(input, {call(&m_filmgrain, p)}, fullViewport(input)).image;
        QVERIFY(!out.isNull());
    }

    // Film grain active: exercises the enqueueGpu body (kernel dispatch).
    void pipeline_filmgrain_active() {
        if (!m_hasGpu) QSKIP("No GPU");
        QMap<QString, QVariant> p;
        p["amount"]    = 50;
        p["size"]      = 2;
        p["lumWeight"] = false;
        QImage input   = makeSolid(64, 64, 128, 128, 128);
        QImage out     = m_pipeline.run(input, {call(&m_filmgrain, p)}, fullViewport(input)).image;
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
        QImage input      = makeSolid(64, 64, 128, 128, 128);
        QImage out        = m_pipeline.run(input, {call(&m_splittoning, p)}, fullViewport(input)).image;
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
        QImage input      = makeSolid(64, 64, 128, 128, 128);
        QImage out        = m_pipeline.run(input, {call(&m_splittoning, p)}, fullViewport(input)).image;
        QVERIFY(!out.isNull());
    }

    // Clarity inactive (amount=0): enqueueGpu early-returns but initGpuKernels runs.
    void pipeline_clarity_inactive() {
        if (!m_hasGpu) QSKIP("No GPU");
        QMap<QString, QVariant> p;
        p["amount"]  = 0;
        p["radius"]  = 30;
        QImage input = makeSolid(64, 64, 128, 128, 128);
        QImage out   = m_pipeline.run(input, {call(&m_clarity, p)}, fullViewport(input)).image;
        QVERIFY(!out.isNull());
    }

    // Clarity active: exercises H/V blur + combine + copy pipeline path.
    void pipeline_clarity_active() {
        if (!m_hasGpu) QSKIP("No GPU");
        QMap<QString, QVariant> p;
        p["amount"]  = 50;
        p["radius"]  = 20;
        QImage input = makeSolid(64, 64, 128, 128, 128);
        QImage out   = m_pipeline.run(input, {call(&m_clarity, p)}, fullViewport(input)).image;
        QVERIFY(!out.isNull());
    }

    // Color balance inactive (all offsets=0): enqueueGpu early-returns but initGpuKernels runs.
    void pipeline_colorbalance_inactive() {
        if (!m_hasGpu) QSKIP("No GPU");
        QMap<QString, QVariant> p;
        p["shadowR"]    = 0;
        p["shadowG"]    = 0;
        p["shadowB"]    = 0;
        p["midtoneR"]   = 0;
        p["midtoneG"]   = 0;
        p["midtoneB"]   = 0;
        p["highlightR"] = 0;
        p["highlightG"] = 0;
        p["highlightB"] = 0;
        QImage input    = makeSolid(64, 64, 128, 128, 128);
        QImage out      = m_pipeline.run(input, {call(&m_colorbalance, p)}, fullViewport(input)).image;
        QVERIFY(!out.isNull());
    }

    // Color balance active: exercises the enqueueGpu body (kernel dispatch).
    void pipeline_colorbalance_active() {
        if (!m_hasGpu) QSKIP("No GPU");
        QMap<QString, QVariant> p;
        p["shadowR"]    = 40;
        p["shadowG"]    = 0;
        p["shadowB"]    = -20;
        p["midtoneR"]   = 0;
        p["midtoneG"]   = 30;
        p["midtoneB"]   = 0;
        p["highlightR"] = 0;
        p["highlightG"] = 0;
        p["highlightB"] = 40;
        QImage input    = makeSolid(64, 64, 128, 128, 128);
        QImage out      = m_pipeline.run(input, {call(&m_colorbalance, p)}, fullViewport(input)).image;
        QVERIFY(!out.isNull());
    }

    // Calling setDevice with a different index bumps the revision counter.
    // The pipeline reinitialises on the next run, which must still succeed.
    void setDevice_switch_pipelineStillWorks() {
        if (!m_hasGpu) QSKIP("No GPU");
        // devices() getter (GpuDeviceRegistry.h line 28)
        QVERIFY(!GpuDeviceRegistry::instance().devices().empty());
        GpuDeviceRegistry::instance().setDevice(1); // covers setDevice body (lines 77-80)
        GpuDeviceRegistry::instance().setDevice(0); // restore
        QImage input = makeSolid(64, 64, 128, 128, 128);
        QImage out   = m_pipeline.run(input, {}, fullViewport(input)).image;
        QVERIFY(!out.isNull());
    }

    // setDevice with out-of-range index, then re-enumerate: covers the
    // bounds check in GpuDeviceRegistry::enumerate() (line 69).
    void enumerate_afterOutOfRangeDevice_resetsIndex() {
        if (!m_hasGpu) QSKIP("No GPU");
        GpuDeviceRegistry::instance().setDevice(99); // index far out of range
        GpuDeviceRegistry::instance().enumerate();   // line 69: 99 >= devices.size() → resets to 0
        QCOMPARE(GpuDeviceRegistry::instance().currentIndex(), 0);
        // Pipeline must still work after the re-enumeration.
        QImage input = makeSolid(32, 32, 128, 128, 128);
        QImage out   = m_pipeline.run(input, {}, fullViewport(input)).image;
        QVERIFY(!out.isNull());
    }

    // Pass an IGpuEffect whose initGpuKernels() always returns false.
    // GpuPipeline::run() should log a warning and return a null image.
    void failInitEffect_warnsAndReturnsNull() {
        if (!m_hasGpu) QSKIP("No GPU");
        FailInitEffect fie;
        QImage         input = makeSolid(32, 32, 100, 100, 100);
        QImage         out   = m_pipeline.run(input, {call(&fie)}, fullViewport(input)).image;
        QVERIFY(out.isNull());
    }

    // Pass an IGpuEffect whose enqueueGpu() always returns false.
    // GpuPipeline::run() should log a warning and return a null image.
    void failEnqueueEffect_warnsAndReturnsNull() {
        if (!m_hasGpu) QSKIP("No GPU");
        FailEnqueueEffect fee;
        QImage            input = makeSolid(32, 32, 100, 100, 100);
        QImage            out   = m_pipeline.run(input, {call(&fee)}, fullViewport(input)).image;
        QVERIFY(out.isNull());
    }

    // PanZoom mode after a Commit: reuses the cached full-res frame, skips effect kernels.
    void panZoom_reusesCachedFrame() {
        if (!m_hasGpu) QSKIP("No GPU");
        QMap<QString, QVariant> p;
        p["brightness"]       = 10;
        p["contrast"]         = 0;
        QImage          input = makeSolid(64, 64, 128, 128, 128);
        ViewportRequest vp    = fullViewport(input);

        // Commit run populates the full-res post-effect cache.
        QImage out1 = m_pipeline.run(input, {call(&m_brightness, p)}, vp, RunMode::Commit).image;
        QVERIFY(!out1.isNull());

        // PanZoom run: reuses the cache, skips effect kernels.
        QImage out2 = m_pipeline.run(input, {call(&m_brightness, p)}, vp, RunMode::PanZoom).image;
        QVERIFY(!out2.isNull());
    }

    void panZoom_changedParameters_doesNotReuseStaleCache() {
        if (!m_hasGpu) QSKIP("No GPU");
        QImage          input = makeSolid(64, 64, 64, 64, 64);
        ViewportRequest vp    = fullViewport(input);

        QMap<QString, QVariant> oldParams{{"brightness", 0}, {"contrast", 0}};
        QImage oldOutput = m_pipeline.run(input, {call(&m_brightness, oldParams)}, vp, RunMode::Commit).image;

        QMap<QString, QVariant> newParams{{"brightness", 50}, {"contrast", 0}};
        QImage newOutput = m_pipeline.run(input, {call(&m_brightness, newParams)}, vp, RunMode::PanZoom).image;

        QVERIFY(!oldOutput.isNull());
        QVERIFY(!newOutput.isNull());
        const auto *oldRow = reinterpret_cast<const QRgb *>(oldOutput.constScanLine(0));
        const auto *newRow = reinterpret_cast<const QRgb *>(newOutput.constScanLine(0));
        QVERIFY(qRed(newRow[0]) > qRed(oldRow[0]));
    }

    // LiveDrag mode: bypasses the cache, runs the preview-sized pipeline
    // (decode+downsample srcBuf → workBuf, then effects at preview size).
    // Covers the preview fallback branch of GpuPipeline::run().
    void liveDrag_runsPreviewPipeline() {
        if (!m_hasGpu) QSKIP("No GPU");
        QMap<QString, QVariant> p;
        p["brightness"] = 10;
        p["contrast"]   = 0;
        QImage input    = makeSolid(64, 64, 128, 128, 128);
        QImage out      = m_pipeline.run(input, {call(&m_brightness, p)}, fullViewport(input), RunMode::LiveDrag).image;
        QVERIFY(!out.isNull());
    }

    // 16-bit sRGB input: selects the 16-bit sRGB downsample/decode kernels.
    void liveDrag_16bitSrgbInput() {
        if (!m_hasGpu) QSKIP("No GPU");
        QImage input(32, 32, QImage::Format_RGBX64);
        input.fill(QColor(128, 128, 128));
        QImage out = m_pipeline.run(input, {}, fullViewport(input), RunMode::LiveDrag).image;
        QVERIFY(!out.isNull());
    }

    // ── Letterbox correctness ────────────────────────────────────────────────
    // Regression tests for the bug where additive effects (brightness offset,
    // contrast midpoint, colour balance) painted onto the black letterbox
    // pixels.  The architectural fix: the pipeline returns image-only pixels
    // with a viewport offset, so there's no letterbox in the output for
    // effects to corrupt.  Letterbox is the viewport widget's job.

    // Square image in square viewport, fit-to-window: no letterbox.
    // Output spans the full viewport, offset is (0, 0).
    void letterbox_squareInSquare_noLetterbox() {
        if (!m_hasGpu) QSKIP("No GPU");
        QImage          input = makeSolid(64, 64, 100, 150, 200);
        ViewportRequest vp;
        vp.displaySize = QSize(64, 64);
        auto r         = m_pipeline.run(input, {}, vp, RunMode::LiveDrag);
        QCOMPARE(r.image.size(), QSize(64, 64));
        QCOMPARE(r.offset, QPoint(0, 0));
    }

    // Landscape 64×32 in 64×64 viewport: letterbox top + bottom.
    // fitScale = min(64/64, 64/32) = 1, regionH = 64, cropY0 = -16,
    // clipY0 = 0, clipY1 = 32 → imgH = 32, imgY0 = 16.
    void letterbox_landscapeInSquare_verticalLetterbox() {
        if (!m_hasGpu) QSKIP("No GPU");
        QImage          input = makeSolid(64, 32, 100, 100, 100);
        ViewportRequest vp;
        vp.displaySize = QSize(64, 64);
        auto r         = m_pipeline.run(input, {}, vp, RunMode::LiveDrag);
        QCOMPARE(r.image.size(), QSize(64, 32));
        QCOMPARE(r.offset, QPoint(0, 16));
    }

    // Portrait 32×64 in 64×64 viewport: letterbox left + right.
    void letterbox_portraitInSquare_horizontalLetterbox() {
        if (!m_hasGpu) QSKIP("No GPU");
        QImage          input = makeSolid(32, 64, 100, 100, 100);
        ViewportRequest vp;
        vp.displaySize = QSize(64, 64);
        auto r         = m_pipeline.run(input, {}, vp, RunMode::LiveDrag);
        QCOMPARE(r.image.size(), QSize(32, 64));
        QCOMPARE(r.offset, QPoint(16, 0));
    }

    // Same letterbox behaviour in Commit mode (effects-at-full-res path).
    void letterbox_commitMode_landscapeInSquare() {
        if (!m_hasGpu) QSKIP("No GPU");
        QImage          input = makeSolid(64, 32, 100, 100, 100);
        ViewportRequest vp;
        vp.displaySize = QSize(64, 64);
        auto r         = m_pipeline.run(input, {}, vp, RunMode::Commit);
        QCOMPARE(r.image.size(), QSize(64, 32));
        QCOMPARE(r.offset, QPoint(0, 16));
    }

    // Same letterbox behaviour in PanZoom cache-hit path.
    void letterbox_panZoomMode_landscapeInSquare() {
        if (!m_hasGpu) QSKIP("No GPU");
        QImage          input = makeSolid(64, 32, 100, 100, 100);
        ViewportRequest vp;
        vp.displaySize = QSize(64, 64);
        // Populate cache first.
        (void)m_pipeline.run(input, {}, vp, RunMode::Commit);
        auto r = m_pipeline.run(input, {}, vp, RunMode::PanZoom);
        QCOMPARE(r.image.size(), QSize(64, 32));
        QCOMPARE(r.offset, QPoint(0, 16));
    }

    // The original bug: a brightness-with-contrast change on a black image
    // turned the black letterbox bars grey.  With the architectural fix the
    // output is image-only, so there's no letterbox in the output at all.
    // Verify both the dimensions and that the output pixels reflect the
    // effect being applied (i.e. effects ran on real image pixels).
    void letterbox_brightnessOnBlackImage_outputIsImageOnly() {
        if (!m_hasGpu) QSKIP("No GPU");
        QMap<QString, QVariant> p;
        p["brightness"]       = 50;
        p["contrast"]         = 0;
        QImage          input = makeSolid(64, 32, 0, 0, 0);
        ViewportRequest vp;
        vp.displaySize = QSize(64, 64);
        auto r         = m_pipeline.run(input, {call(&m_brightness, p)}, vp, RunMode::LiveDrag);
        // Output is image-sized, not viewport-sized — letterbox can't exist
        // in something that isn't there.
        QCOMPARE(r.image.size(), QSize(64, 32));
        QCOMPARE(r.offset, QPoint(0, 16));
        // And every pixel in the output got the brightness applied (because
        // every pixel in the output is a real image pixel, not letterbox).
        for (int y = 0; y < r.image.height(); ++y) {
            const QRgb *row = reinterpret_cast<const QRgb *>(r.image.constScanLine(y));
            for (int x = 0; x < r.image.width(); ++x)
                QVERIFY(qRed(row[x]) > 0 && qGreen(row[x]) > 0 && qBlue(row[x]) > 0);
        }
    }

    // Same regression but exercising the additive colour-balance offset.
    void letterbox_colorBalanceOnBlackImage_outputIsImageOnly() {
        if (!m_hasGpu) QSKIP("No GPU");
        QMap<QString, QVariant> p;
        p["shadowR"]          = 40;
        p["shadowG"]          = 0;
        p["shadowB"]          = 0;
        p["midtoneR"]         = 0;
        p["midtoneG"]         = 0;
        p["midtoneB"]         = 0;
        p["highlightR"]       = 0;
        p["highlightG"]       = 0;
        p["highlightB"]       = 0;
        QImage          input = makeSolid(64, 32, 0, 0, 0);
        ViewportRequest vp;
        vp.displaySize = QSize(64, 64);
        auto r         = m_pipeline.run(input, {call(&m_colorbalance, p)}, vp, RunMode::LiveDrag);
        QCOMPARE(r.image.size(), QSize(64, 32));
        QCOMPARE(r.offset, QPoint(0, 16));
    }

    // Zoomed in: cropX0..cropX1 stays inside [0, srcW], no clipping.  Output
    // spans the full viewport with offset (0, 0).
    void letterbox_zoomedIn_noLetterbox() {
        if (!m_hasGpu) QSKIP("No GPU");
        QImage          input = makeSolid(64, 64, 100, 100, 100);
        ViewportRequest vp;
        vp.displaySize = QSize(64, 64);
        vp.zoom        = 2.0f;
        auto r         = m_pipeline.run(input, {}, vp, RunMode::LiveDrag);
        QCOMPARE(r.image.size(), QSize(64, 64));
        QCOMPARE(r.offset, QPoint(0, 0));
    }

    // Default (empty) ViewportRequest — the export path.  No viewport, no
    // letterbox: output equals source size, offset (0, 0).
    void letterbox_emptyViewport_returnsFullImage() {
        if (!m_hasGpu) QSKIP("No GPU");
        QImage input = makeSolid(40, 60, 50, 100, 150);
        auto   r     = m_pipeline.run(input, {}, {}, RunMode::Commit);
        QCOMPARE(r.image.size(), QSize(40, 60));
        QCOMPARE(r.offset, QPoint(0, 0));
    }

    // 16-bit linear input (tagged color_space=linear by RawLoader): selects the
    // 16-bit linear decode kernel in both LiveDrag and Commit paths.
    void linearInput_selectsLinearKernels() {
        if (!m_hasGpu) QSKIP("No GPU");
        QImage input(32, 32, QImage::Format_RGBX64);
        input.fill(QColor(128, 128, 128));
        input.setText("color_space", "linear");
        QImage outLive = m_pipeline.run(input, {}, fullViewport(input), RunMode::LiveDrag).image;
        QVERIFY(!outLive.isNull());
        QImage outCommit = m_pipeline.run(input, {}, fullViewport(input), RunMode::Commit).image;
        QVERIFY(!outCommit.isNull());
    }
};

QTEST_MAIN(TestGpuPipeline)
#include "test_gpu_pipeline.moc"
