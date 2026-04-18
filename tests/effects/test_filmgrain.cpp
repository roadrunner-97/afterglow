#include <QTest>
#include <QSignalSpy>
#include <QCheckBox>
#include <QSlider>
#include <QWidget>
#include "FilmGrainEffect.h"
#include "GpuDeviceRegistry.h"
#include "ImageHelpers.h"
#include "ParamSlider.h"

class TestFilmGrain : public QObject {
    Q_OBJECT

private:
    bool m_hasGpu = false;

    // Count how many pixels of `out` differ from the single reference colour `ref`.
    static int differingPixels(const QImage& out, QRgb ref) {
        int n = 0;
        for (int y = 0; y < out.height(); ++y) {
            const auto* row = reinterpret_cast<const QRgb*>(out.constScanLine(y));
            for (int x = 0; x < out.width(); ++x)
                if (row[x] != ref) ++n;
        }
        return n;
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
        FilmGrainEffect e;
        QVERIFY(e.processImage(QImage(), {}).isNull());
    }

    // amount=0 → identity: every pixel unchanged regardless of other params.
    void zeroAmount_isIdentity() {
        FilmGrainEffect e;
        QImage input = makeSolid(64, 64, 100, 120, 80);
        QMap<QString, QVariant> params;
        params["amount"]    = 0;
        params["size"]      = 3;
        params["lumWeight"] = true;
        QImage out = e.processImage(input, params);
        QVERIFY(!out.isNull());
        QVERIFY(allPixels(out, [](QRgb px) {
            return qRed(px) == 100 && qGreen(px) == 120 && qBlue(px) == 80;
        }));
    }

    // Non-zero amount perturbs a solid midtone image: most pixels should differ
    // from the constant input colour.
    void nonZeroAmount_perturbsMidtone() {
        if (!m_hasGpu) QSKIP("No GPU");
        FilmGrainEffect e;
        QImage input = makeSolid(64, 64, 128, 128, 128);
        QMap<QString, QVariant> params;
        params["amount"]    = 50;
        params["size"]      = 1;
        params["lumWeight"] = false;  // full strength everywhere
        QImage out = e.processImage(input, params);
        QVERIFY(!out.isNull());

        int changed = differingPixels(out, qRgb(128, 128, 128));
        // With uniform noise in [-0.5, 0.5]*255 almost every pixel should move.
        QVERIFY2(changed > 64 * 64 * 0.9,
                 qPrintable(QString("changed=%1").arg(changed)));
    }

    // With luminance weighting, pure black and pure white pixels should remain
    // untouched because the weight (4*L*(1-L)) vanishes at L=0 and L=1.
    void lumWeight_protectsExtremes() {
        if (!m_hasGpu) QSKIP("No GPU");
        FilmGrainEffect e;
        QMap<QString, QVariant> params;
        params["amount"]    = 100;
        params["size"]      = 1;
        params["lumWeight"] = true;

        QImage black = makeSolid(32, 32, 0, 0, 0);
        QImage white = makeSolid(32, 32, 255, 255, 255);
        QImage bOut  = e.processImage(black, params);
        QImage wOut  = e.processImage(white, params);
        QVERIFY(!bOut.isNull() && !wOut.isNull());

        QCOMPARE(differingPixels(bOut, qRgb(0, 0, 0)),         0);
        QCOMPARE(differingPixels(wOut, qRgb(255, 255, 255)),   0);
    }

    // Without luminance weighting, even pure-black pixels should be lifted
    // by positive noise samples — so at least some pixels differ.
    void lumWeight_disabled_affectsExtremes() {
        if (!m_hasGpu) QSKIP("No GPU");
        FilmGrainEffect e;
        QMap<QString, QVariant> params;
        params["amount"]    = 100;
        params["size"]      = 1;
        params["lumWeight"] = false;

        QImage black = makeSolid(32, 32, 0, 0, 0);
        QImage bOut  = e.processImage(black, params);
        QVERIFY(!bOut.isNull());

        int changed = differingPixels(bOut, qRgb(0, 0, 0));
        QVERIFY2(changed > 0, qPrintable(QString("changed=%1").arg(changed)));
    }

    // Value noise must vary smoothly, not in solid on/off blocks.  With
    // size=8 adjacent pixels share a lattice cell and their noise values
    // differ gradually.  We assert two things:
    //   (a) the output is not piecewise-constant (pixels within a cell
    //       are not all identical) — rules out the old blocky hash.
    //   (b) pixel-to-pixel deltas are bounded well below the full noise
    //       range — rules out pure per-pixel white noise.
    void grainSize_smoothNoiseNotBlocky() {
        if (!m_hasGpu) QSKIP("No GPU");
        FilmGrainEffect e;
        QImage input = makeSolid(64, 64, 128, 128, 128);
        QMap<QString, QVariant> params;
        params["amount"]    = 50;
        params["size"]      = 8;
        params["seed"]      = 42;
        params["lumWeight"] = false;
        QImage out = e.processImage(input, params);
        QVERIFY(!out.isNull());

        int distinct = 0, maxDelta = 0;
        for (int y = 0; y < 64; ++y) {
            const auto* row = reinterpret_cast<const QRgb*>(out.constScanLine(y));
            for (int x = 1; x < 64; ++x) {
                const int d = std::abs(qRed(row[x]) - qRed(row[x - 1]));
                if (d != 0) ++distinct;
                if (d > maxDelta) maxDelta = d;
            }
        }
        // (a) lots of non-equal neighbours — not a block pattern
        QVERIFY2(distinct > 64 * 32,
                 qPrintable(QString("distinct=%1").arg(distinct)));
        // (b) smooth noise: pure per-pixel white noise with amount=50 can
        // swing the full ~127 units between neighbours; value noise over
        // 8-pixel cells caps per-step change at roughly 2*1.875/size*0.5*255
        // ≈ 60.  A bound of 80 comfortably rules out the per-pixel pattern.
        QVERIFY2(maxDelta < 80,
                 qPrintable(QString("maxDelta=%1").arg(maxDelta)));
    }

    // Different seeds must produce visibly different grain patterns.
    void seed_affectsOutput() {
        if (!m_hasGpu) QSKIP("No GPU");
        FilmGrainEffect e;
        QImage input = makeSolid(64, 64, 128, 128, 128);
        QMap<QString, QVariant> base;
        base["amount"]    = 50;
        base["size"]      = 2;
        base["lumWeight"] = false;

        QMap<QString, QVariant> p0 = base; p0["seed"] = 0;
        QMap<QString, QVariant> p1 = base; p1["seed"] = 1;
        QImage a = e.processImage(input, p0);
        QImage b = e.processImage(input, p1);
        QVERIFY(!a.isNull() && !b.isNull());
        QVERIFY(a != b);
    }

    // Noise pattern must be deterministic for a given seed — running the
    // effect twice on the same input should yield identical output.
    void determinism_twoRunsMatch() {
        if (!m_hasGpu) QSKIP("No GPU");
        FilmGrainEffect e;
        QImage input = makeSolid(32, 32, 128, 128, 128);
        QMap<QString, QVariant> params;
        params["amount"]    = 50;
        params["size"]      = 2;
        params["lumWeight"] = false;
        QImage a = e.processImage(input, params);
        QImage b = e.processImage(input, params);
        QVERIFY(!a.isNull() && !b.isNull());
        QCOMPARE(a, b);
    }

    // Non-square (wide) image: grain noise is generated per-pixel with a
    // stride-dependent index.  Dimensions must be preserved and most pixels
    // of a solid midtone should be perturbed.
    void nonSquare_nonZeroAmount_perturbsMidtone() {
        if (!m_hasGpu) QSKIP("No GPU");
        FilmGrainEffect e;
        QImage input = makeSolid(128, 64, 128, 128, 128);
        QMap<QString, QVariant> params;
        params["amount"]    = 50;
        params["size"]      = 1;
        params["lumWeight"] = false;
        QImage out = e.processImage(input, params);
        QVERIFY(!out.isNull());
        QCOMPARE(out.width(),  128);
        QCOMPARE(out.height(), 64);
        int changed = differingPixels(out, qRgb(128, 128, 128));
        QVERIFY2(changed > 128 * 64 * 0.9,
                 qPrintable(QString("changed=%1").arg(changed)));
    }

    // 16-bit path: no crash and output is non-null when amount is non-zero.
    void grain_16bit() {
        if (!m_hasGpu) QSKIP("No GPU");
        FilmGrainEffect e;
        QImage input = makeSolid16bit(64, 64, 128, 128, 128);
        QMap<QString, QVariant> params;
        params["amount"]    = 50;
        params["size"]      = 2;
        params["lumWeight"] = true;
        QImage out = e.processImage(input, params);
        QVERIFY(!out.isNull());
    }

    void meta_nonEmpty() {
        FilmGrainEffect e;
        QVERIFY(!e.getName().isEmpty());
        QVERIFY(!e.getDescription().isEmpty());
        QVERIFY(!e.getVersion().isEmpty());
        QVERIFY(e.initialize());
    }

    void defaultParameters_keysAndValues() {
        FilmGrainEffect e;
        auto params = e.getParameters();
        QVERIFY(params.contains("amount"));
        QVERIFY(params.contains("size"));
        QVERIFY(params.contains("seed"));
        QVERIFY(params.contains("lumWeight"));
        // Widget not yet built → fall-through defaults
        QCOMPARE(params["amount"].toInt(),     0);
        QCOMPARE(params["size"].toInt(),       8);
        QCOMPARE(params["seed"].toInt(),       0);
        QCOMPARE(params["lumWeight"].toBool(), true);
    }

    void createControlsWidget_constructsAndCaches() {
        FilmGrainEffect e;
        QWidget* w = e.createControlsWidget();
        QVERIFY(w != nullptr);
        QVERIFY(e.createControlsWidget() == w);
    }

    void getParameters_afterWidget_returnsUiDefaults() {
        FilmGrainEffect e;
        e.createControlsWidget();
        auto params = e.getParameters();
        QCOMPARE(params["amount"].toInt(),     0);
        QCOMPARE(params["size"].toInt(),       8);
        QCOMPARE(params["seed"].toInt(),       0);
        QCOMPARE(params["lumWeight"].toBool(), true);
    }

    void supportsGpuInPlace_returnsTrue() {
        FilmGrainEffect e;
        QVERIFY(e.supportsGpuInPlace());
    }

    // Fire every slider's editingFinished + valueChanged path and the checkbox toggle.
    void controlSignals_coverLambdaBodies() {
        FilmGrainEffect e;
        QWidget* w = e.createControlsWidget();
        QVERIFY(w);

        QSignalSpy spyChanged(&e, &PhotoEditorEffect::parametersChanged);
        QSignalSpy spyLive(&e, &PhotoEditorEffect::liveParametersChanged);

        auto sliders = w->findChildren<ParamSlider*>();
        QCOMPARE(sliders.size(), 3);
        for (auto* ps : sliders) {
            auto* qs = ps->findChild<QSlider*>();
            QVERIFY(qs);
            qs->setValue(qs->value() + 1);
            QMetaObject::invokeMethod(qs, "sliderReleased");
        }

        auto* check = w->findChild<QCheckBox*>();
        QVERIFY(check);
        check->toggle();

        QVERIFY(spyChanged.count() >= 4);  // 3 sliders + 1 checkbox
        QVERIFY(spyLive.count()    >= 3);
    }

    // Heap-allocate so the destructor body is explicitly attributed.
    void destructor_heapAllocated_doesNotCrash() {
        auto* e = new FilmGrainEffect();
        e->createControlsWidget();
        delete e;
    }
};

QTEST_MAIN(TestFilmGrain)
#include "test_filmgrain.moc"
