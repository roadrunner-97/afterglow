#include <QTest>
#include <QCheckBox>
#include <QWidget>
#include "GrayscaleEffect.h"
#include "GpuDeviceRegistry.h"
#include "ImageHelpers.h"

class TestGrayscale : public QObject {
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
        GrayscaleEffect e;
        QVERIFY(runEffect(e, QImage(), {}).isNull());
    }

    // m_active=false (default): processImage returns the input image unchanged.
    void inactive_isIdentity() {
        GrayscaleEffect e;  // m_active defaults to false
        QImage input = makeSolid(32, 32, 200, 100, 50);
        QImage out   = runEffect(e, input, {});
        // When inactive, the code does: if (!m_active) return image;
        // The returned copy shares pixel data, so pixel values are identical.
        QVERIFY(!out.isNull());
        QVERIFY(allPixels(out, [](QRgb px) {
            return qRed(px) == 200 && qGreen(px) == 100 && qBlue(px) == 50;
        }));
    }

    // m_active=true (via checkbox): all output pixels must have R=G=B.
    void active_convertsToGray() {
        if (!m_hasGpu) QSKIP("No GPU");
        GrayscaleEffect e;

        // Activate the effect through its checkbox (the only public API).
        QWidget* w  = e.createControlsWidget();
        auto*    cb = w->findChild<QCheckBox*>();
        QVERIFY(cb);
        cb->setChecked(true);  // triggers the lambda: m_active = true

        QImage input = makeSolid(32, 32, 200, 100, 50);
        QImage out   = runEffect(e, input, {});
        QVERIFY(!out.isNull());

        // Every pixel must be grey: R=G=B (kernel uses luminosity formula).
        QVERIFY(allPixels(out, [](QRgb px) {
            return qRed(px) == qGreen(px) && qGreen(px) == qBlue(px);
        }));
    }

    // The pipeline does grayscale in linear light using Rec.709 weights
    // (0.2126 R + 0.7152 G + 0.0722 B), then re-encodes to sRGB.  For input
    // sRGB (200, 100, 50) → linear (0.578, 0.127, 0.032) → L ≈ 0.216 →
    // sRGB ≈ 128.
    void active_greyValueMatchesLuminosity() {
        if (!m_hasGpu) QSKIP("No GPU");
        GrayscaleEffect e;
        QWidget* w  = e.createControlsWidget();
        auto*    cb = w->findChild<QCheckBox*>();
        QVERIFY(cb);
        cb->setChecked(true);

        QImage input = makeSolid(32, 32, 200, 100, 50);
        QImage out   = runEffect(e, input, {});

        int grey = pixelR(out, 0, 0);
        QVERIFY2(qAbs(grey - 128) <= 2,
                 qPrintable(QString("expected ~128, got %1").arg(grey)));
    }

    void meta_nonEmpty() {
        GrayscaleEffect e;
        QVERIFY(!e.getName().isEmpty());
        QVERIFY(!e.getDescription().isEmpty());
        QVERIFY(!e.getVersion().isEmpty());
        QVERIFY(e.initialize());
    }

    // Active grayscale on a 16-bit RGBX64 image: all output pixels must be grey (R=G=B).
    void active_16bit() {
        if (!m_hasGpu) QSKIP("No GPU");
        GrayscaleEffect e;
        QWidget* w  = e.createControlsWidget();
        auto*    cb = w->findChild<QCheckBox*>();
        QVERIFY(cb);
        cb->setChecked(true);

        QImage input = makeSolid16bit(32, 32, 200, 100, 50);
        QImage out   = runEffect(e, input, {});
        QVERIFY(!out.isNull());
        QImage out32 = out.convertToFormat(QImage::Format_RGB32);
        QVERIFY(allPixels(out32, [](QRgb px) {
            return qRed(px) == qGreen(px) && qGreen(px) == qBlue(px);
        }));
    }

    // Activating and then deactivating via the checkbox: subsequent processImage
    // should return the image unchanged (because m_active is back to false).
    void toggle_offRestoresIdentity() {
        if (!m_hasGpu) QSKIP("No GPU");
        GrayscaleEffect e;
        QWidget* w  = e.createControlsWidget();
        auto*    cb = w->findChild<QCheckBox*>();
        QVERIFY(cb);

        cb->setChecked(true);
        cb->setChecked(false);  // back to m_active = false

        QImage input = makeSolid(32, 32, 200, 100, 50);
        QImage out   = runEffect(e, input, {});
        QVERIFY(allPixels(out, [](QRgb px) {
            return qRed(px) == 200 && qGreen(px) == 100 && qBlue(px) == 50;
        }));
    }

    // Non-square (wide) image: luminosity conversion runs per-pixel.  Output
    // dimensions must match input and every pixel must be grey (R=G=B).
    void nonSquare_active_convertsToGrey() {
        if (!m_hasGpu) QSKIP("No GPU");
        GrayscaleEffect e;
        QWidget* w  = e.createControlsWidget();
        auto*    cb = w->findChild<QCheckBox*>();
        QVERIFY(cb);
        cb->setChecked(true);

        QImage input = makeSolid(128, 64, 200, 100, 50);
        QImage out   = runEffect(e, input, {});
        QVERIFY(!out.isNull());
        QCOMPARE(out.width(),  128);
        QCOMPARE(out.height(), 64);
        QVERIFY(allPixels(out, [](QRgb px) {
            return qRed(px) == qGreen(px) && qGreen(px) == qBlue(px);
        }));
    }

    void destructor_heapAllocated_doesNotCrash() {
        auto* e = new GrayscaleEffect();
        e->createControlsWidget();
        delete e;
    }
};

// QWidget requires a full QApplication, not just QCoreApplication.
QTEST_MAIN(TestGrayscale)
#include "test_grayscale.moc"
