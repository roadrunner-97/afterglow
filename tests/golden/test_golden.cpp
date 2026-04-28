// Golden-image regression test.  For each cases/<name>/ directory it loads
// input.*, applies settings.yaml, runs the GPU pipeline (and bakes crop +
// rotate exactly like File → Save Image), then compares the result against
// expected.* using SSIM on luminance.  Mismatches dump actual.png + diff.png.
#include <QTest>
#include <QDir>
#include <QFile>
#include <QFileInfo>
#include <QImage>
#include <QPainter>
#include <QTransform>
#include <QtMath>
#include <cmath>
#include <memory>

#include "EffectManager.h"
#include "GpuDeviceRegistry.h"
#include "GpuPipeline.h"
#include "ICropSource.h"
#include "ImageMetadata.h"
#include "RawLoader.h"
#include "SettingsImporter.h"

#include "BlurEffect.h"
#include "BrightnessEffect.h"
#include "ClarityEffect.h"
#include "ColorBalanceEffect.h"
#include "CropRotateEffect.h"
#include "DenoiseEffect.h"
#include "ExposureEffect.h"
#include "FilmGrainEffect.h"
#include "GrayscaleEffect.h"
#include "HotPixelEffect.h"
#include "SaturationEffect.h"
#include "SplitToningEffect.h"
#include "UnsharpEffect.h"
#include "VignetteEffect.h"
#include "WhiteBalanceEffect.h"

#ifndef GOLDEN_CASES_DIR
#define GOLDEN_CASES_DIR ""
#endif

namespace {

constexpr double kSsimThreshold = 0.98;
constexpr int    kBlock         = 8;     // SSIM block size
constexpr double kC1            = (0.01 * 255) * (0.01 * 255);
constexpr double kC2            = (0.03 * 255) * (0.03 * 255);

double luma(QRgb p) {
    return 0.299 * qRed(p) + 0.587 * qGreen(p) + 0.114 * qBlue(p);
}

// Block-averaged SSIM on luminance.  Both inputs must have identical sizes.
double computeSSIM(const QImage& aIn, const QImage& bIn) {
    const QImage a = aIn.convertToFormat(QImage::Format_RGB32);
    const QImage b = bIn.convertToFormat(QImage::Format_RGB32);
    const int W = a.width();
    const int H = a.height();
    const int N = kBlock * kBlock;

    double sum   = 0.0;
    int    count = 0;
    for (int by = 0; by + kBlock <= H; by += kBlock) {
        for (int bx = 0; bx + kBlock <= W; bx += kBlock) {
            double mx = 0.0, my = 0.0;
            for (int y = 0; y < kBlock; ++y) {
                const QRgb* ar = reinterpret_cast<const QRgb*>(a.scanLine(by + y));
                const QRgb* br = reinterpret_cast<const QRgb*>(b.scanLine(by + y));
                for (int x = 0; x < kBlock; ++x) {
                    mx += luma(ar[bx + x]);
                    my += luma(br[bx + x]);
                }
            }
            mx /= N;
            my /= N;

            double vx = 0.0, vy = 0.0, cxy = 0.0;
            for (int y = 0; y < kBlock; ++y) {
                const QRgb* ar = reinterpret_cast<const QRgb*>(a.scanLine(by + y));
                const QRgb* br = reinterpret_cast<const QRgb*>(b.scanLine(by + y));
                for (int x = 0; x < kBlock; ++x) {
                    const double da = luma(ar[bx + x]) - mx;
                    const double db = luma(br[bx + x]) - my;
                    vx  += da * da;
                    vy  += db * db;
                    cxy += da * db;
                }
            }
            vx  /= N;
            vy  /= N;
            cxy /= N;

            const double num = (2 * mx * my + kC1) * (2 * cxy + kC2);
            const double den = (mx * mx + my * my + kC1) * (vx + vy + kC2);
            sum += num / den;
            ++count;
        }
    }
    return count ? sum / count : 1.0;
}

// 4× amplified absolute per-channel diff so subtle mismatches are visible.
QImage makeDiff(const QImage& aIn, const QImage& bIn) {
    const QImage a = aIn.convertToFormat(QImage::Format_RGB32);
    const QImage b = bIn.convertToFormat(QImage::Format_RGB32);
    const int W = qMin(a.width(),  b.width());
    const int H = qMin(a.height(), b.height());
    QImage diff(W, H, QImage::Format_RGB32);
    for (int y = 0; y < H; ++y) {
        const QRgb* ar = reinterpret_cast<const QRgb*>(a.scanLine(y));
        const QRgb* br = reinterpret_cast<const QRgb*>(b.scanLine(y));
        QRgb*       dr = reinterpret_cast<QRgb*>(diff.scanLine(y));
        for (int x = 0; x < W; ++x) {
            int dR = qAbs(qRed  (ar[x]) - qRed  (br[x])) * 4;
            int dG = qAbs(qGreen(ar[x]) - qGreen(br[x])) * 4;
            int dB = qAbs(qBlue (ar[x]) - qBlue (br[x])) * 4;
            dr[x] = qRgb(qMin(255, dR), qMin(255, dG), qMin(255, dB));
        }
    }
    return diff;
}

// Mirrors PhotoEditorApp::applyCropAndRotate — bakes the user's
// non-destructive crop + rotation into the exported QImage.
QImage applyCropAndRotate(const QImage& image, const ICropSource& cs) {
    if (image.isNull()) return image;
    const QRectF cropN = cs.userCropRect();
    const double cx = cropN.center().x() * image.width();
    const double cy = cropN.center().y() * image.height();
    const QSize dst(static_cast<int>(std::round(cropN.width()  * image.width())),
                    static_cast<int>(std::round(cropN.height() * image.height())));
    if (dst.isEmpty()) return image;

    QTransform t;
    t.translate(dst.width() * 0.5, dst.height() * 0.5);
    t.rotate(-static_cast<double>(cs.userCropAngle()));
    t.translate(-cx, -cy);

    QImage out(dst, image.format());
    out.fill(Qt::black);
    QPainter p(&out);
    p.setRenderHint(QPainter::SmoothPixmapTransform);
    p.setTransform(t);
    p.drawImage(0, 0, image);
    p.end();
    return out;
}

QFileInfo findSingle(const QDir& d, const QString& base) {
    for (const QFileInfo& fi : d.entryInfoList(QDir::Files | QDir::NoDotAndDotDot))
        if (fi.completeBaseName() == base) return fi;
    return {};
}

} // namespace

class TestGolden : public QObject {
    Q_OBJECT

private:
    bool          m_hasGpu = false;
    GpuPipeline   m_pipeline;
    EffectManager m_effects;
    QMap<QString, QMap<QString, QVariant>> m_defaults;  // effect name → default params

    // Restore every effect to its default parameter map and re-enable it,
    // so each case starts from a clean slate before applying its YAML.
    void resetEffects() {
        const auto& entries = m_effects.entries();
        for (int i = 0; i < entries.size(); ++i) {
            m_effects.setEnabled(i, true);
            entries[i].effect->applyParameters(m_defaults[entries[i].effect->getName()]);
        }
    }

private slots:
    void initTestCase() {
        GpuDeviceRegistry::instance().enumerate();
        if (GpuDeviceRegistry::instance().count() == 0)
            QSKIP("No OpenCL device found — skipping golden-image tests");
        GpuDeviceRegistry::instance().setDevice(0);
        m_hasGpu = true;

        // Effect order must match src/main.cpp so the pipeline runs the same
        // sequence as the editor that produced the reference outputs.
        m_effects.addEffect(std::make_unique<CropRotateEffect>());
        m_effects.addEffect(std::make_unique<HotPixelEffect>());
        m_effects.addEffect(std::make_unique<ExposureEffect>());
        m_effects.addEffect(std::make_unique<WhiteBalanceEffect>());
        m_effects.addEffect(std::make_unique<BrightnessEffect>());
        m_effects.addEffect(std::make_unique<SaturationEffect>());
        m_effects.addEffect(std::make_unique<BlurEffect>());
        m_effects.addEffect(std::make_unique<GrayscaleEffect>());
        m_effects.addEffect(std::make_unique<UnsharpEffect>());
        m_effects.addEffect(std::make_unique<DenoiseEffect>());
        m_effects.addEffect(std::make_unique<VignetteEffect>());
        m_effects.addEffect(std::make_unique<FilmGrainEffect>());
        m_effects.addEffect(std::make_unique<SplitToningEffect>());
        m_effects.addEffect(std::make_unique<ClarityEffect>());
        m_effects.addEffect(std::make_unique<ColorBalanceEffect>());

        for (const auto& e : m_effects.entries()) {
            e.effect->initialize();
            m_defaults[e.effect->getName()] = e.effect->getParameters();
        }
    }

    void run_data() {
        QTest::addColumn<QString>("caseDir");
        const QDir root(QStringLiteral(GOLDEN_CASES_DIR));
        int rows = 0;
        if (root.exists()) {
            for (const QFileInfo& fi : root.entryInfoList(QDir::Dirs | QDir::NoDotAndDotDot,
                                                          QDir::Name)) {
                QTest::newRow(fi.fileName().toUtf8().constData()) << fi.absoluteFilePath();
                ++rows;
            }
        }
        // QTest aborts a data-driven slot with zero rows; emit one sentinel row
        // so an empty cases dir reports as a clean skip instead.
        if (rows == 0) QTest::newRow("(no fixtures)") << QString();
    }

    void run() {
        if (!m_hasGpu) QSKIP("No GPU");
        QFETCH(QString, caseDir);
        if (caseDir.isEmpty())
            QSKIP("No cases under " GOLDEN_CASES_DIR " — drop fixtures in to enable");
        const QDir d(caseDir);

        const QFileInfo input    = findSingle(d, "input");
        const QFileInfo expected = findSingle(d, "expected");
        QVERIFY2(input.exists(),    qPrintable("missing input.* in "    + caseDir));
        QVERIFY2(expected.exists(), qPrintable("missing expected.* in " + caseDir));

        QString settingsPath = d.filePath("settings.yaml");
        if (!QFile::exists(settingsPath)) settingsPath = d.filePath("settings.yml");
        QVERIFY2(QFile::exists(settingsPath),
                 qPrintable("missing settings.yaml/yml in " + caseDir));

        // Load source image with metadata.
        ImageMetadata meta;
        QImage src;
        if (RawLoader::isRawFile(input.fileName())) {
            src = RawLoader::load(input.absoluteFilePath(), &meta);
            if (src.isNull()) QSKIP("RAW input could not be loaded (LibRaw missing?)");
        } else {
            src = QImage(input.absoluteFilePath());
            QVERIFY2(!src.isNull(),
                     qPrintable("could not load input " + input.absoluteFilePath()));
        }

        // Reset, then run the same load-time hooks the editor runs before the
        // user's YAML settings are applied (white balance reads colorTempK,
        // crop reads the source size, ...).
        resetEffects();
        for (const auto& e : m_effects.entries()) {
            e.effect->onImageLoaded(meta);
            if (auto* cs = dynamic_cast<ICropSource*>(e.effect))
                cs->setSourceImageSize(src.size());
        }

        SettingsImporter::Settings parsed;
        QString error;
        QVERIFY2(SettingsImporter::readYaml(settingsPath, &parsed, &error),
                 qPrintable("settings parse failed: " + error));
        SettingsImporter::applyToManager(parsed, m_effects);

        // Build the same _userCrop* injection ImageProcessor uses.
        QMap<QString, QVariant> cropInjected;
        for (const auto& e : m_effects.entries()) {
            if (auto* cs = dynamic_cast<ICropSource*>(e.effect)) {
                const QRectF r = cs->userCropRect();
                cropInjected = {
                    {"_userCropX0", r.left()},
                    {"_userCropY0", r.top()},
                    {"_userCropX1", r.right()},
                    {"_userCropY1", r.bottom()},
                    {"_userCropAngle", static_cast<double>(cs->userCropAngle())},
                };
                break;
            }
        }
        QVector<GpuPipelineCall> calls;
        for (const auto& e : m_effects.entries()) {
            if (!e.enabled) continue;
            QMap<QString, QVariant> p = e.effect->getParameters();
            for (auto it = cropInjected.cbegin(); it != cropInjected.cend(); ++it)
                p.insert(it.key(), it.value());
            calls.append({e.effect, e.gpu, p});
        }

        // Same call shape as ImageProcessor::exportImageAsync.
        QImage out = m_pipeline.run(src, calls, {}, RunMode::Commit);
        QVERIFY2(!out.isNull(), "pipeline returned null image");

        // Bake crop + rotate, mirroring PhotoEditorApp::onExportComplete.
        for (const auto& e : m_effects.entries()) {
            if (!e.enabled) continue;
            if (auto* cs = dynamic_cast<ICropSource*>(e.effect)) {
                out = applyCropAndRotate(out, *cs);
                break;
            }
        }

        const QImage exp(expected.absoluteFilePath());
        QVERIFY2(!exp.isNull(),
                 qPrintable("could not load " + expected.absoluteFilePath()));

        if (out.size() != exp.size()) {
            out.save(d.filePath("actual.png"));
            QFAIL(qPrintable(QString("size mismatch: pipeline=%1x%2, expected=%3x%4 "
                                     "(actual.png written to %5)")
                .arg(out.width()).arg(out.height())
                .arg(exp.width()).arg(exp.height())
                .arg(d.absolutePath())));
        }

        const double ssim = computeSSIM(out, exp);
        if (ssim < kSsimThreshold) {
            out.save(d.filePath("actual.png"));
            makeDiff(out, exp).save(d.filePath("diff.png"));
            QFAIL(qPrintable(QString("SSIM %1 < threshold %2 — actual.png and "
                                     "diff.png written to %3")
                .arg(ssim, 0, 'f', 4).arg(kSsimThreshold).arg(d.absolutePath())));
        }
        qInfo("SSIM = %.4f", ssim);
    }
};

QTEST_MAIN(TestGolden)
#include "test_golden.moc"
