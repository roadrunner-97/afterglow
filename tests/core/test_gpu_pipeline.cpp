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
