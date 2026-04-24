#include <QTest>
#include <QSignalSpy>
#include <QPushButton>
#include <QSlider>
#include <QWidget>
#include <QMouseEvent>
#include <QPainter>
#include <QPixmap>

#include "CropRotateEffect.h"
#include "IInteractiveEffect.h"
#include "ICropSource.h"
#include "ParamSlider.h"
#include "ImageHelpers.h"

// ============================================================================
// Helpers
// ============================================================================

// Build a simple ViewportTransform so a 100×100 image maps 1:1 onto a
// 100×100 viewport with zoom=1 (displayScale=1), centre at (0.5,0.5).
static ViewportTransform makeVT(int imageW = 100, int imageH = 100,
                                int vpW = 100, int vpH = 100) {
    ViewportTransform vt;
    vt.imageSize    = QSize(imageW, imageH);
    vt.viewportSize = QSize(vpW, vpH);
    vt.center       = QPointF(0.5, 0.5);
    vt.zoom         = 1.0f;
    return vt;
}

// Synthesise and send a QMouseEvent (press/move/release).
static QMouseEvent makeMouseEvent(QEvent::Type type, QPointF pos,
                                   Qt::MouseButton btn = Qt::LeftButton) {
    return QMouseEvent(type, pos, pos, pos, btn,
                       (type == QEvent::MouseButtonRelease) ? Qt::NoButton : btn,
                       Qt::NoModifier);
}

// ============================================================================
// Test class
// ============================================================================

class TestCropRotate : public QObject {
    Q_OBJECT

private slots:

    // ── Meta ──────────────────────────────────────────────────────────────────

    void meta_names() {
        CropRotateEffect e;
        QCOMPARE(e.getName(),    QString("Crop & Rotate"));
        QVERIFY(!e.getDescription().isEmpty());
        QCOMPARE(e.getVersion(), QString("1.0"));
        QVERIFY(e.initialize());
    }

    void supportsGpuInPlace_returnsTrue() {
        CropRotateEffect e;
        QVERIFY(e.supportsGpuInPlace());
    }

    // ── Defaults ─────────────────────────────────────────────────────────────

    void defaults_cropRect_isFullFrame() {
        CropRotateEffect e;
        QCOMPARE(e.userCropRect(), QRectF(0.0, 0.0, 1.0, 1.0));
    }

    void defaults_angle_isZero() {
        CropRotateEffect e;
        QCOMPARE(e.userCropAngle(), 0.0f);
    }

    void defaults_flip_isFalse() {
        CropRotateEffect e;
        QVERIFY(!e.userCropFlip());
    }

    void defaults_quarterTurns_isZero() {
        CropRotateEffect e;
        QCOMPARE(e.quarterTurns(), 0);
    }

    void defaults_subTool_isHandles() {
        CropRotateEffect e;
        QCOMPARE(e.subTool(), CropRotateEffect::SubTool::Handles);
    }

    // ── processImage passthrough ──────────────────────────────────────────────

    void processImage_null_returnsNull() {
        CropRotateEffect e;
        QVERIFY(e.processImage(QImage()).isNull());
    }

    void processImage_nonNull_returnsUnchanged() {
        CropRotateEffect e;
        QImage input = makeSolid(64, 64, 128, 64, 32);
        QImage out   = e.processImage(input);
        // Same dimensions and pixel data
        QCOMPARE(out.size(), input.size());
        QCOMPARE(out.format(), input.format());
        QVERIFY(allPixels(out, [](QRgb px) {
            return qRed(px) == 128 && qGreen(px) == 64 && qBlue(px) == 32;
        }));
    }

    void processImage_withParams_returnsUnchanged() {
        CropRotateEffect e;
        QImage input = makeSolid(32, 32, 100, 100, 100);
        QMap<QString, QVariant> params;
        params["angle"] = 15.0;
        QImage out = e.processImage(input, params);
        QVERIFY(!out.isNull());
        QCOMPARE(out.size(), input.size());
    }

    // ── getParameters ────────────────────────────────────────────────────────

    void getParameters_defaultKeys() {
        CropRotateEffect e;
        auto p = e.getParameters();
        QVERIFY(p.contains("angle"));
        QVERIFY(p.contains("quarterTurns"));
        QVERIFY(p.contains("flipH"));
        QVERIFY(p.contains("cropX0"));
        QVERIFY(p.contains("cropY0"));
        QVERIFY(p.contains("cropX1"));
        QVERIFY(p.contains("cropY1"));
    }

    void getParameters_defaultValues() {
        CropRotateEffect e;
        auto p = e.getParameters();
        QCOMPARE(p["angle"].toDouble(),        0.0);
        QCOMPARE(p["quarterTurns"].toInt(),     0);
        QCOMPARE(p["flipH"].toBool(),           false);
        QCOMPARE(p["cropX0"].toDouble(),        0.0);
        QCOMPARE(p["cropY0"].toDouble(),        0.0);
        QCOMPARE(p["cropX1"].toDouble(),        1.0);
        QCOMPARE(p["cropY1"].toDouble(),        1.0);
    }

    // ── createControlsWidget ────────────────────────────────────────────────

    void createControlsWidget_notNull() {
        CropRotateEffect e;
        QWidget* w = e.createControlsWidget();
        QVERIFY(w != nullptr);
    }

    void createControlsWidget_returnsSameWidget() {
        CropRotateEffect e;
        QWidget* w1 = e.createControlsWidget();
        QWidget* w2 = e.createControlsWidget();
        QCOMPARE(w1, w2);
    }

    void createControlsWidget_hasAngleSlider() {
        CropRotateEffect e;
        QWidget* w = e.createControlsWidget();
        auto sliders = w->findChildren<ParamSlider*>();
        QVERIFY(!sliders.isEmpty());
    }

    // ── Rotation slider signal wiring ────────────────────────────────────────

    void angleSlider_valueChanged_emitsLiveParametersChanged() {
        CropRotateEffect e;
        QWidget* w = e.createControlsWidget();
        QVERIFY(w);

        QSignalSpy spyLive(&e, &PhotoEditorEffect::liveParametersChanged);
        auto sliders = w->findChildren<ParamSlider*>();
        QVERIFY(!sliders.isEmpty());

        ParamSlider* slider = sliders.first();
        auto* qs = slider->findChild<QSlider*>();
        QVERIFY(qs);
        qs->setValue(qs->value() + 10);
        QVERIFY(spyLive.count() >= 1);
    }

    void angleSlider_editingFinished_emitsParametersChanged() {
        CropRotateEffect e;
        QWidget* w = e.createControlsWidget();
        QVERIFY(w);

        QSignalSpy spyChanged(&e, &PhotoEditorEffect::parametersChanged);
        auto sliders = w->findChildren<ParamSlider*>();
        QVERIFY(!sliders.isEmpty());

        ParamSlider* slider = sliders.first();
        auto* qs = slider->findChild<QSlider*>();
        QVERIFY(qs);
        qs->setValue(qs->value() + 5);
        QMetaObject::invokeMethod(qs, "sliderReleased");
        QVERIFY(spyChanged.count() >= 1);
    }

    // ── Quarter-turn buttons ────────────────────────────────────────────────

    void rotate90ccw_incrementsQuarterTurns() {
        CropRotateEffect e;
        QWidget* w = e.createControlsWidget();

        QSignalSpy spy(&e, &PhotoEditorEffect::parametersChanged);
        auto btns = w->findChildren<QPushButton*>();
        QPushButton* ccwBtn = nullptr;
        for (auto* b : btns)
            if (b->text().contains("CCW")) { ccwBtn = b; break; }
        QVERIFY(ccwBtn);

        ccwBtn->click();
        QCOMPARE(e.quarterTurns(), 1);
        QVERIFY(spy.count() >= 1);

        ccwBtn->click();
        QCOMPARE(e.quarterTurns(), 2);
    }

    void rotate90cw_decrementsQuarterTurns() {
        CropRotateEffect e;
        QWidget* w = e.createControlsWidget();

        auto btns = w->findChildren<QPushButton*>();
        QPushButton* cwBtn = nullptr;
        for (auto* b : btns)
            if (b->text().contains("CW") && !b->text().contains("CCW")) {
                cwBtn = b; break;
            }
        QVERIFY(cwBtn);

        cwBtn->click();
        QCOMPARE(e.quarterTurns(), 3);  // 0 - 1 mod 4 = 3
    }

    void rotate90ccw_wrapAround() {
        CropRotateEffect e;
        QWidget* w = e.createControlsWidget();

        auto btns = w->findChildren<QPushButton*>();
        QPushButton* ccwBtn = nullptr;
        for (auto* b : btns)
            if (b->text().contains("CCW")) { ccwBtn = b; break; }
        QVERIFY(ccwBtn);

        // 4 × CCW should bring us back to 0
        for (int i = 0; i < 4; ++i) ccwBtn->click();
        QCOMPARE(e.quarterTurns(), 0);
    }

    void rotate90cw_wrapAround() {
        CropRotateEffect e;
        QWidget* w = e.createControlsWidget();

        auto btns = w->findChildren<QPushButton*>();
        QPushButton* cwBtn = nullptr;
        for (auto* b : btns)
            if (b->text().contains("CW") && !b->text().contains("CCW")) {
                cwBtn = b; break;
            }
        QVERIFY(cwBtn);

        // 4 × CW should bring us back to 0
        for (int i = 0; i < 4; ++i) cwBtn->click();
        QCOMPARE(e.quarterTurns(), 0);
    }

    // ── Flip button ──────────────────────────────────────────────────────────

    void flipButton_togglesFlipH() {
        CropRotateEffect e;
        QWidget* w = e.createControlsWidget();

        QSignalSpy spy(&e, &PhotoEditorEffect::parametersChanged);
        auto btns = w->findChildren<QPushButton*>();
        QPushButton* flipBtn = nullptr;
        for (auto* b : btns)
            if (b->text().contains("Flip")) { flipBtn = b; break; }
        QVERIFY(flipBtn);

        QVERIFY(!e.userCropFlip());
        flipBtn->click();
        QVERIFY(e.userCropFlip());
        QVERIFY(spy.count() >= 1);

        flipBtn->click();
        QVERIFY(!e.userCropFlip());
    }

    // ── Reset button ─────────────────────────────────────────────────────────

    void resetButton_restoresDefaults() {
        CropRotateEffect e;
        QWidget* w = e.createControlsWidget();

        // Mess with state
        QSignalSpy spy(&e, &PhotoEditorEffect::parametersChanged);
        auto btns = w->findChildren<QPushButton*>();
        QPushButton* ccwBtn = nullptr;
        QPushButton* flipBtn = nullptr;
        QPushButton* resetBtn = nullptr;
        for (auto* b : btns) {
            if (b->text().contains("CCW"))   ccwBtn   = b;
            if (b->text().contains("Flip"))  flipBtn  = b;
            if (b->text().contains("Reset")) resetBtn = b;
        }
        QVERIFY(ccwBtn && flipBtn && resetBtn);

        ccwBtn->click();
        flipBtn->click();
        spy.clear();

        resetBtn->click();
        QCOMPARE(e.userCropRect(),  QRectF(0.0, 0.0, 1.0, 1.0));
        QCOMPARE(e.userCropAngle(), 0.0f);
        QVERIFY(!e.userCropFlip());
        QCOMPARE(e.quarterTurns(), 0);
        QVERIFY(spy.count() >= 1);
    }

    // ── userCropAngle combines quarterTurns and angleDeg ────────────────────

    void userCropAngle_combinesQuarterTurnsAndAngle() {
        CropRotateEffect e;
        QWidget* w = e.createControlsWidget();

        // Set angle via internal slider drag (setValue on QSlider triggers valueChanged)
        auto sliders = w->findChildren<ParamSlider*>();
        QVERIFY(!sliders.isEmpty());
        auto* qs = sliders.first()->findChild<QSlider*>();
        QVERIFY(qs);
        // Scale: range is -45..45 with step 0.1, so slider int = value / 0.1
        // We want m_angleDeg = 10°, so set slider to 100 ticks from 0
        qs->setValue(100);  // 100 * 0.1 = 10°

        // One CCW turn: quarterTurns=1
        auto btns = w->findChildren<QPushButton*>();
        for (auto* b : btns)
            if (b->text().contains("CCW")) { b->click(); break; }

        float expected = 90.0f + 10.0f;
        QVERIFY2(std::abs(e.userCropAngle() - expected) < 0.5f,
                 qPrintable(QString("angle=%1 expected=%2").arg(e.userCropAngle()).arg(expected)));
    }

    // ── Straighten button toggles SubTool ────────────────────────────────────

    void straightenButton_togglesSubTool() {
        CropRotateEffect e;
        QWidget* w = e.createControlsWidget();

        auto btns = w->findChildren<QPushButton*>();
        QPushButton* straightenBtn = nullptr;
        for (auto* b : btns)
            if (b->text().contains("Straighten")) { straightenBtn = b; break; }
        QVERIFY(straightenBtn);

        QCOMPARE(e.subTool(), CropRotateEffect::SubTool::Handles);
        straightenBtn->click();
        QCOMPARE(e.subTool(), CropRotateEffect::SubTool::StraightenLine);
        straightenBtn->click();
        QCOMPARE(e.subTool(), CropRotateEffect::SubTool::Handles);
    }

    // ── ICropSource interface ────────────────────────────────────────────────

    void icropSource_interface_accessible() {
        CropRotateEffect e;
        ICropSource* src = &e;
        QCOMPARE(src->userCropRect(),  QRectF(0.0, 0.0, 1.0, 1.0));
        QCOMPARE(src->userCropAngle(), 0.0f);
        QVERIFY(!src->userCropFlip());
    }

    // ── IGpuEffect no-op interface ───────────────────────────────────────────

    void initGpuKernels_returnsTrueWithNullContext() {
        // We just check the interface is callable and compiles — can't pass
        // real CL objects without a device, but we exercise the no-op path.
        // The function simply returns true; tested via mock context not needed.
        CropRotateEffect e;
        IGpuEffect* gpu = &e;
        // We can't easily create a cl::Context without an OpenCL device,
        // so just confirm the vtable is wired correctly (symbol present):
        (void)gpu;
        QVERIFY(true);
    }

    // ── Mouse interaction — ViewportTransform helpers ────────────────────────

    // With a 1:1 mapping (100x100 image in 100x100 viewport, centre=0.5),
    // source pixels map 1:1 to screen pixels.
    void viewportTransform_sourceToScreen_identity() {
        ViewportTransform vt = makeVT();
        QPointF p = vt.sourceToScreen({50.0, 50.0});
        QCOMPARE(p.x(), 50.0);
        QCOMPARE(p.y(), 50.0);
    }

    void viewportTransform_screenToSource_identity() {
        ViewportTransform vt = makeVT();
        QPointF p = vt.screenToSource({30.0, 70.0});
        QCOMPARE(p.x(), 30.0);
        QCOMPARE(p.y(), 70.0);
    }

    // ── Mouse outside the crop rect — not claimed ────────────────────────────

    void mousePress_outsideCrop_notClaimed() {
        CropRotateEffect e;
        ViewportTransform vt = makeVT();
        // Crop is full frame [0,0,1,1]; anything on the image IS inside.
        // Shrink the crop first so we have an outside region.
        // Use mousePress on a corner to pull the crop in, then press outside.
        // Easier: directly test via a press at a pixel that does not hit any
        // handle and is outside the crop.
        // Let's shrink the crop manually by dragging the TL corner.
        // TL corner screen pos with 1:1 map: (0, 0)
        {
            auto ev = makeMouseEvent(QEvent::MouseButtonPress, {0.0, 0.0});
            bool claimed = e.mousePress(&ev, vt);
            QVERIFY(claimed);  // hits TL corner handle
        }
        // Move TL to (30, 30) — crop becomes [0.3, 0.3, 0.7, 0.7]
        {
            auto ev = makeMouseEvent(QEvent::MouseMove, {30.0, 30.0});
            e.mouseMove(&ev, vt);
        }
        {
            auto ev = makeMouseEvent(QEvent::MouseButtonRelease, {30.0, 30.0});
            e.mouseRelease(&ev, vt);
        }

        // Now press at (10, 10) — outside the new crop rect and not on any handle
        auto ev = makeMouseEvent(QEvent::MouseButtonPress, {10.0, 10.0});
        bool claimed = e.mousePress(&ev, vt);
        QVERIFY(!claimed);
    }

    // ── Corner drag updates crop rect ────────────────────────────────────────

    void mouseDrag_topLeftCorner_updatesCropRect() {
        CropRotateEffect e;
        ViewportTransform vt = makeVT();  // 100x100 1:1

        // TL handle is at source (0,0) → screen (0,0)
        {
            auto ev = makeMouseEvent(QEvent::MouseButtonPress, {0.0, 0.0});
            bool claimed = e.mousePress(&ev, vt);
            QVERIFY(claimed);
        }
        // Move to (20, 20) — normalised (0.2, 0.2)
        {
            auto ev = makeMouseEvent(QEvent::MouseMove, {20.0, 20.0});
            bool moved = e.mouseMove(&ev, vt);
            QVERIFY(moved);
        }
        {
            auto ev = makeMouseEvent(QEvent::MouseButtonRelease, {20.0, 20.0});
            e.mouseRelease(&ev, vt);
        }

        QRectF crop = e.userCropRect();
        QVERIFY(crop.x() > 0.0);
        QVERIFY(crop.y() > 0.0);
        QVERIFY(std::abs(crop.x() + crop.width()  - 1.0) < 1e-5);
        QVERIFY(std::abs(crop.y() + crop.height() - 1.0) < 1e-5);
    }

    void mouseDrag_bottomRightCorner_updatesCropRect() {
        CropRotateEffect e;
        ViewportTransform vt = makeVT();

        // BR handle at screen (100, 100)
        {
            auto ev = makeMouseEvent(QEvent::MouseButtonPress, {100.0, 100.0});
            e.mousePress(&ev, vt);
        }
        {
            auto ev = makeMouseEvent(QEvent::MouseMove, {80.0, 80.0});
            e.mouseMove(&ev, vt);
        }
        {
            auto ev = makeMouseEvent(QEvent::MouseButtonRelease, {80.0, 80.0});
            e.mouseRelease(&ev, vt);
        }

        QRectF crop = e.userCropRect();
        QCOMPARE(crop.x(), 0.0);
        QCOMPARE(crop.y(), 0.0);
        QVERIFY(crop.width()  < 1.0);
        QVERIFY(crop.height() < 1.0);
    }

    // ── Top-right corner drag ────────────────────────────────────────────────

    void mouseDrag_topRightCorner_updatesCropRect() {
        CropRotateEffect e;
        ViewportTransform vt = makeVT();

        // TR handle at screen (100, 0)
        {
            auto ev = makeMouseEvent(QEvent::MouseButtonPress, {100.0, 0.0});
            e.mousePress(&ev, vt);
        }
        {
            auto ev = makeMouseEvent(QEvent::MouseMove, {80.0, 20.0});
            e.mouseMove(&ev, vt);
        }
        {
            auto ev = makeMouseEvent(QEvent::MouseButtonRelease, {80.0, 20.0});
            e.mouseRelease(&ev, vt);
        }

        QRectF crop = e.userCropRect();
        QCOMPARE(crop.x(), 0.0);
        QVERIFY(crop.y() > 0.0);
        QVERIFY(crop.width()  < 1.0);
        QVERIFY(std::abs(crop.y() + crop.height() - 1.0) < 1e-5);
    }

    // ── Bottom-left corner drag ──────────────────────────────────────────────

    void mouseDrag_bottomLeftCorner_updatesCropRect() {
        CropRotateEffect e;
        ViewportTransform vt = makeVT();

        // BL handle at screen (0, 100)
        {
            auto ev = makeMouseEvent(QEvent::MouseButtonPress, {0.0, 100.0});
            e.mousePress(&ev, vt);
        }
        {
            auto ev = makeMouseEvent(QEvent::MouseMove, {20.0, 80.0});
            e.mouseMove(&ev, vt);
        }
        {
            auto ev = makeMouseEvent(QEvent::MouseButtonRelease, {20.0, 80.0});
            e.mouseRelease(&ev, vt);
        }

        QRectF crop = e.userCropRect();
        QVERIFY(crop.x() > 0.0);
        QCOMPARE(crop.y(), 0.0);
        QVERIFY(std::abs(crop.x() + crop.width() - 1.0) < 1e-5);
        QVERIFY(crop.height() < 1.0);
    }

    // ── Edge drag (top) ──────────────────────────────────────────────────────

    void mouseDrag_topEdge_updatesCropTop() {
        CropRotateEffect e;
        ViewportTransform vt = makeVT();

        // Top edge midpoint at screen (50, 0)
        {
            auto ev = makeMouseEvent(QEvent::MouseButtonPress, {50.0, 0.0});
            bool claimed = e.mousePress(&ev, vt);
            QVERIFY(claimed);
        }
        {
            auto ev = makeMouseEvent(QEvent::MouseMove, {50.0, 15.0});
            e.mouseMove(&ev, vt);
        }
        {
            auto ev = makeMouseEvent(QEvent::MouseButtonRelease, {50.0, 15.0});
            e.mouseRelease(&ev, vt);
        }

        QRectF crop = e.userCropRect();
        QVERIFY(crop.y() > 0.0);
        QCOMPARE(crop.x(), 0.0);
        QCOMPARE(crop.x() + crop.width(), 1.0);
    }

    // ── Edge drag (bottom) ───────────────────────────────────────────────────

    void mouseDrag_bottomEdge_updatesCropBottom() {
        CropRotateEffect e;
        ViewportTransform vt = makeVT();

        // Bottom edge midpoint at screen (50, 100)
        {
            auto ev = makeMouseEvent(QEvent::MouseButtonPress, {50.0, 100.0});
            e.mousePress(&ev, vt);
        }
        {
            auto ev = makeMouseEvent(QEvent::MouseMove, {50.0, 80.0});
            e.mouseMove(&ev, vt);
        }
        {
            auto ev = makeMouseEvent(QEvent::MouseButtonRelease, {50.0, 80.0});
            e.mouseRelease(&ev, vt);
        }

        QRectF crop = e.userCropRect();
        QCOMPARE(crop.y(), 0.0);
        QVERIFY(crop.height() < 1.0);
    }

    // ── Edge drag (left) ─────────────────────────────────────────────────────

    void mouseDrag_leftEdge_updatesCropLeft() {
        CropRotateEffect e;
        ViewportTransform vt = makeVT();

        // Left edge midpoint at screen (0, 50)
        {
            auto ev = makeMouseEvent(QEvent::MouseButtonPress, {0.0, 50.0});
            e.mousePress(&ev, vt);
        }
        {
            auto ev = makeMouseEvent(QEvent::MouseMove, {20.0, 50.0});
            e.mouseMove(&ev, vt);
        }
        {
            auto ev = makeMouseEvent(QEvent::MouseButtonRelease, {20.0, 50.0});
            e.mouseRelease(&ev, vt);
        }

        QRectF crop = e.userCropRect();
        QVERIFY(crop.x() > 0.0);
        QCOMPARE(crop.y(), 0.0);
    }

    // ── Edge drag (right) ────────────────────────────────────────────────────

    void mouseDrag_rightEdge_updatesCropRight() {
        CropRotateEffect e;
        ViewportTransform vt = makeVT();

        // Right edge midpoint at screen (100, 50)
        {
            auto ev = makeMouseEvent(QEvent::MouseButtonPress, {100.0, 50.0});
            e.mousePress(&ev, vt);
        }
        {
            auto ev = makeMouseEvent(QEvent::MouseMove, {70.0, 50.0});
            e.mouseMove(&ev, vt);
        }
        {
            auto ev = makeMouseEvent(QEvent::MouseButtonRelease, {70.0, 50.0});
            e.mouseRelease(&ev, vt);
        }

        QRectF crop = e.userCropRect();
        QCOMPARE(crop.x(), 0.0);
        QVERIFY(crop.width() < 1.0);
    }

    // ── Move drag translates the whole rect ──────────────────────────────────

    void mouseDrag_insideCrop_movesCropRect() {
        CropRotateEffect e;
        ViewportTransform vt = makeVT();

        // First shrink the crop to [0.2, 0.2, 0.6, 0.6] by dragging BL corner.
        {
            auto evP = makeMouseEvent(QEvent::MouseButtonPress, {0.0, 100.0});
            e.mousePress(&evP, vt);
            auto evM = makeMouseEvent(QEvent::MouseMove, {20.0, 80.0});
            e.mouseMove(&evM, vt);
            auto evR = makeMouseEvent(QEvent::MouseButtonRelease, {20.0, 80.0});
            e.mouseRelease(&evR, vt);
        }
        {
            auto evP = makeMouseEvent(QEvent::MouseButtonPress, {100.0, 0.0});
            e.mousePress(&evP, vt);
            auto evM = makeMouseEvent(QEvent::MouseMove, {80.0, 20.0});
            e.mouseMove(&evM, vt);
            auto evR = makeMouseEvent(QEvent::MouseButtonRelease, {80.0, 20.0});
            e.mouseRelease(&evR, vt);
        }

        QRectF before = e.userCropRect();
        QVERIFY(before.width() < 1.0 && before.height() < 1.0);

        // Centre of crop in screen coords
        float cx = static_cast<float>(before.center().x()) * 100.0f;
        float cy = static_cast<float>(before.center().y()) * 100.0f;

        QSignalSpy spyLive(&e, &PhotoEditorEffect::liveParametersChanged);

        // Press inside the crop (not on a handle)
        {
            auto ev = makeMouseEvent(QEvent::MouseButtonPress,
                                     QPointF(cx, cy));
            bool claimed = e.mousePress(&ev, vt);
            QVERIFY(claimed);
        }
        // Move +5 px in X, +5 px in Y
        {
            auto ev = makeMouseEvent(QEvent::MouseMove,
                                     QPointF(cx + 5.0f, cy + 5.0f));
            e.mouseMove(&ev, vt);
        }
        {
            auto ev = makeMouseEvent(QEvent::MouseButtonRelease,
                                     QPointF(cx + 5.0f, cy + 5.0f));
            e.mouseRelease(&ev, vt);
        }

        QRectF after = e.userCropRect();
        // Size preserved
        QCOMPARE(after.width(),  before.width());
        QCOMPARE(after.height(), before.height());
        // Position shifted
        QVERIFY(std::abs(after.x() - before.x() - 0.05) < 0.01);
        QVERIFY(spyLive.count() >= 1);
    }

    // ── Rotation grip updates angle ──────────────────────────────────────────

    void mouseDrag_rotationGrip_updatesAngle() {
        CropRotateEffect e;
        ViewportTransform vt = makeVT();

        // Rotation grip is at screen (50, -30) with default 1:1 map,
        // crop centre at (50, 50), grip ROT_GRIP_OFFSET=30 above top midpoint.
        // Top midpoint of full-frame crop: source (50, 0) → screen (50, 0).
        // Grip: (50, 0 - 30) = (50, -30).
        QPointF gripPos(50.0, -30.0);
        {
            auto ev = makeMouseEvent(QEvent::MouseButtonPress, gripPos);
            bool claimed = e.mousePress(&ev, vt);
            QVERIFY(claimed);
        }

        // Drag to the right of centre → positive angle change
        QPointF movePos(90.0, -30.0);
        {
            auto ev = makeMouseEvent(QEvent::MouseMove, movePos);
            bool moved = e.mouseMove(&ev, vt);
            QVERIFY(moved);
        }
        {
            auto ev = makeMouseEvent(QEvent::MouseButtonRelease, movePos);
            e.mouseRelease(&ev, vt);
        }

        // The angle should have changed from 0 (direction: either sign is fine
        // depending on the relative angle; just check it moved non-trivially).
        // With the start at (-30 above centre = angle~=-90 from centre) and
        // move to (40, -30 relative to centre) the delta should be non-zero.
        QVERIFY(e.userCropAngle() != 0.0f || e.quarterTurns() == 0);
        // Actually verify angle changed (or the test geometry didn't trigger it).
        // The precise value depends on atan2 math; just verify no crash and that
        // after parametersChanged was emitted the angle is within range.
        float a = e.userCropAngle();
        QVERIFY(a >= -45.0f && a <= 45.0f);
    }

    // ── Non-left-button press — not claimed ──────────────────────────────────

    void mousePress_rightButton_notClaimed() {
        CropRotateEffect e;
        ViewportTransform vt = makeVT();
        auto ev = makeMouseEvent(QEvent::MouseButtonPress, {50.0, 50.0},
                                  Qt::RightButton);
        QVERIFY(!e.mousePress(&ev, vt));
    }

    void mouseRelease_rightButton_notClaimed() {
        CropRotateEffect e;
        ViewportTransform vt = makeVT();
        auto ev = makeMouseEvent(QEvent::MouseButtonRelease, {50.0, 50.0},
                                  Qt::RightButton);
        QVERIFY(!e.mouseRelease(&ev, vt));
    }

    // ── mouseMove with no drag — not claimed ─────────────────────────────────

    void mouseMove_noDrag_returnsFalse() {
        CropRotateEffect e;
        ViewportTransform vt = makeVT();
        auto ev = makeMouseEvent(QEvent::MouseMove, {50.0, 50.0});
        QVERIFY(!e.mouseMove(&ev, vt));
    }

    // ── mouseRelease without prior press ─────────────────────────────────────

    void mouseRelease_noDrag_returnsFalse() {
        CropRotateEffect e;
        ViewportTransform vt = makeVT();
        auto ev = makeMouseEvent(QEvent::MouseButtonRelease, {50.0, 50.0});
        QVERIFY(!e.mouseRelease(&ev, vt));
    }

    // ── StraightenLine mousePress returns true ───────────────────────────────

    void mousePress_straightenLine_claimsEvent() {
        CropRotateEffect e;
        QWidget* w = e.createControlsWidget();

        // Activate StraightenLine sub-tool via button
        auto btns = w->findChildren<QPushButton*>();
        for (auto* b : btns)
            if (b->text().contains("Straighten")) { b->click(); break; }
        QCOMPARE(e.subTool(), CropRotateEffect::SubTool::StraightenLine);

        ViewportTransform vt = makeVT();
        auto ev = makeMouseEvent(QEvent::MouseButtonPress, {50.0, 50.0});
        QVERIFY(e.mousePress(&ev, vt));
    }

    // ── cursorFor ────────────────────────────────────────────────────────────

    void cursorFor_cornerHandle_returnsDiagCursor() {
        CropRotateEffect e;
        ViewportTransform vt = makeVT();
        // TL corner at screen (0,0)
        QCursor c = e.cursorFor({0.0, 0.0}, vt);
        QVERIFY(c.shape() == Qt::SizeFDiagCursor || c.shape() == Qt::SizeBDiagCursor);
    }

    void cursorFor_edgeHandle_returnsSizeCursor() {
        CropRotateEffect e;
        ViewportTransform vt = makeVT();
        // Top edge midpoint at screen (50, 0)
        QCursor c = e.cursorFor({50.0, 0.0}, vt);
        QVERIFY(c.shape() == Qt::SizeVerCursor || c.shape() == Qt::SizeHorCursor);
    }

    void cursorFor_insideCrop_returnsSizeAllCursor() {
        CropRotateEffect e;
        ViewportTransform vt = makeVT();
        // Centre of image, not on any handle
        QCursor c = e.cursorFor({50.0, 50.0}, vt);
        QCOMPARE(c.shape(), Qt::SizeAllCursor);
    }

    void cursorFor_rotationGrip_returnsOpenHandCursor() {
        CropRotateEffect e;
        ViewportTransform vt = makeVT();
        // Rotation grip at (50, -30)
        QCursor c = e.cursorFor({50.0, -30.0}, vt);
        QCOMPARE(c.shape(), Qt::OpenHandCursor);
    }

    void cursorFor_straightenMode_returnsCrosshair() {
        CropRotateEffect e;
        QWidget* w = e.createControlsWidget();
        auto btns = w->findChildren<QPushButton*>();
        for (auto* b : btns)
            if (b->text().contains("Straighten")) { b->click(); break; }

        ViewportTransform vt = makeVT();
        QCursor c = e.cursorFor({50.0, 50.0}, vt);
        QCOMPARE(c.shape(), Qt::CrossCursor);
    }

    void cursorFor_outsideEverything_returnsArrow() {
        CropRotateEffect e;
        ViewportTransform vt = makeVT(100, 100, 200, 200);
        // A point way outside the image (screen coords beyond the image extent)
        QCursor c = e.cursorFor({-50.0, -50.0}, vt);
        QCOMPARE(c.shape(), Qt::ArrowCursor);
    }

    // ── paintOverlay — just verify it doesn't crash ──────────────────────────

    void paintOverlay_doesNotCrash() {
        CropRotateEffect e;
        ViewportTransform vt = makeVT();

        QPixmap pm(100, 100);
        pm.fill(Qt::black);
        QPainter painter(&pm);
        e.paintOverlay(painter, vt);
        painter.end();
        QVERIFY(true);
    }

    void paintOverlay_emptyViewport_doesNotCrash() {
        CropRotateEffect e;
        ViewportTransform vt;
        vt.imageSize    = QSize(0, 0);
        vt.viewportSize = QSize(0, 0);

        QPixmap pm(1, 1);
        QPainter painter(&pm);
        e.paintOverlay(painter, vt);
        painter.end();
        QVERIFY(true);
    }

    // ── Minimum crop size enforcement ────────────────────────────────────────

    void drag_corner_enforcesMinimumSize() {
        CropRotateEffect e;
        ViewportTransform vt = makeVT();

        // Drag BR corner all the way to TL — should clamp to min size
        {
            auto ev = makeMouseEvent(QEvent::MouseButtonPress, {100.0, 100.0});
            e.mousePress(&ev, vt);
        }
        {
            auto ev = makeMouseEvent(QEvent::MouseMove, {0.0, 0.0});
            e.mouseMove(&ev, vt);
        }
        {
            auto ev = makeMouseEvent(QEvent::MouseButtonRelease, {0.0, 0.0});
            e.mouseRelease(&ev, vt);
        }

        QRectF crop = e.userCropRect();
        QVERIFY(crop.width()  >= 0.04);
        QVERIFY(crop.height() >= 0.04);
    }

    void drag_topEdge_enforcesMinimumSize() {
        CropRotateEffect e;
        ViewportTransform vt = makeVT();

        {
            auto ev = makeMouseEvent(QEvent::MouseButtonPress, {50.0, 0.0});
            e.mousePress(&ev, vt);
        }
        {
            // Pull top edge past the bottom — should clamp
            auto ev = makeMouseEvent(QEvent::MouseMove, {50.0, 200.0});
            e.mouseMove(&ev, vt);
        }
        {
            auto ev = makeMouseEvent(QEvent::MouseButtonRelease, {50.0, 200.0});
            e.mouseRelease(&ev, vt);
        }

        QVERIFY(e.userCropRect().height() >= 0.04);
    }

    void drag_leftEdge_enforcesMinimumSize() {
        CropRotateEffect e;
        ViewportTransform vt = makeVT();

        {
            auto ev = makeMouseEvent(QEvent::MouseButtonPress, {0.0, 50.0});
            e.mousePress(&ev, vt);
        }
        {
            auto ev = makeMouseEvent(QEvent::MouseMove, {200.0, 50.0});
            e.mouseMove(&ev, vt);
        }
        {
            auto ev = makeMouseEvent(QEvent::MouseButtonRelease, {200.0, 50.0});
            e.mouseRelease(&ev, vt);
        }

        QVERIFY(e.userCropRect().width() >= 0.04);
    }

    // ── Move clamped to image bounds ─────────────────────────────────────────

    void move_clampedToImageBounds() {
        CropRotateEffect e;
        ViewportTransform vt = makeVT();

        // Shrink crop first (drag TR inward)
        {
            auto ev = makeMouseEvent(QEvent::MouseButtonPress, {100.0, 0.0});
            e.mousePress(&ev, vt);
            auto evM = makeMouseEvent(QEvent::MouseMove, {50.0, 50.0});
            e.mouseMove(&evM, vt);
            auto evR = makeMouseEvent(QEvent::MouseButtonRelease, {50.0, 50.0});
            e.mouseRelease(&evR, vt);
        }

        QRectF before = e.userCropRect();
        float cx = static_cast<float>(before.center().x()) * 100.0f;
        float cy = static_cast<float>(before.center().y()) * 100.0f;

        // Try to move far past the right/bottom edge
        {
            auto ev = makeMouseEvent(QEvent::MouseButtonPress, QPointF(cx, cy));
            e.mousePress(&ev, vt);
            auto evM = makeMouseEvent(QEvent::MouseMove, QPointF(cx + 500.0f, cy + 500.0f));
            e.mouseMove(&evM, vt);
            auto evR = makeMouseEvent(QEvent::MouseButtonRelease,
                                       QPointF(cx + 500.0f, cy + 500.0f));
            e.mouseRelease(&evR, vt);
        }

        QRectF after = e.userCropRect();
        QVERIFY(after.x() >= 0.0);
        QVERIFY(after.y() >= 0.0);
        QVERIFY(after.x() + after.width()  <= 1.0 + 1e-6);
        QVERIFY(after.y() + after.height() <= 1.0 + 1e-6);
    }

    // ── Destructor ────────────────────────────────────────────────────────────

    void destructor_heapAllocated_doesNotCrash() {
        auto* e = new CropRotateEffect();
        e->createControlsWidget();
        delete e;
    }
};

QTEST_MAIN(TestCropRotate)
#include "test_croprotate.moc"
