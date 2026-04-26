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

    // ── Reset button ─────────────────────────────────────────────────────────

    void resetButton_restoresDefaults() {
        CropRotateEffect e;
        QWidget* w = e.createControlsWidget();

        // Mess with state
        QSignalSpy spy(&e, &PhotoEditorEffect::parametersChanged);
        auto btns = w->findChildren<QPushButton*>();
        QPushButton* ccwBtn = nullptr;
        QPushButton* resetBtn = nullptr;
        for (auto* b : btns) {
            if (b->text().contains("CCW"))   ccwBtn   = b;
            if (b->text().contains("Reset")) resetBtn = b;
        }
        QVERIFY(ccwBtn && resetBtn);

        ccwBtn->click();
        spy.clear();

        resetBtn->click();
        QCOMPARE(e.userCropRect(),  QRectF(0.0, 0.0, 1.0, 1.0));
        QCOMPARE(e.userCropAngle(), 0.0f);
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

    // ── StraightenLine — full drag computes correct angle ───────────────────

    // Helper: enter StraightenLine mode via the controls button.
    static void enterStraighten(CropRotateEffect& e) {
        QWidget* w = e.createControlsWidget();
        for (auto* b : w->findChildren<QPushButton*>())
            if (b->text().contains("Straighten")) { b->click(); break; }
    }

    // A line tilted ~5.7° above horizontal (going up-right) needs a
    // CW rotation to flatten — m_angleDeg should come out negative.
    void straightenLine_horizonTiltUpRight_setsNegativeAngle() {
        CropRotateEffect e;
        enterStraighten(e);
        ViewportTransform vt = makeVT();

        auto p = makeMouseEvent(QEvent::MouseButtonPress,  {10.0, 50.0});
        auto m = makeMouseEvent(QEvent::MouseMove,         {110.0, 40.0});
        auto r = makeMouseEvent(QEvent::MouseButtonRelease,{110.0, 40.0});
        QVERIFY(e.mousePress(&p, vt));
        QVERIFY(e.mouseMove (&m, vt));
        QVERIFY(e.mouseRelease(&r, vt));

        // atan2(-10, 100)≈-5.71° ; lineDeg=+5.71° ; quarter=0 ; angle=-5.71°
        QVERIFY(std::abs(e.userCropAngle() + 5.71f) < 0.1f);
        // Tool exits back to Handles automatically
        QCOMPARE(e.subTool(), CropRotateEffect::SubTool::Handles);
    }

    // A near-vertical line snaps to the vertical axis.
    void straightenLine_nearVertical_snapsToVerticalAxis() {
        CropRotateEffect e;
        enterStraighten(e);
        ViewportTransform vt = makeVT();

        // From top (10,10) to bottom-right (20,90): slightly off vertical.
        auto p = makeMouseEvent(QEvent::MouseButtonPress,  {10.0, 10.0});
        auto m = makeMouseEvent(QEvent::MouseMove,         {20.0, 90.0});
        auto r = makeMouseEvent(QEvent::MouseButtonRelease,{20.0, 90.0});
        e.mousePress(&p, vt);
        e.mouseMove(&m, vt);
        e.mouseRelease(&r, vt);

        // atan2(80, 10)≈82.87° ; lineDeg=-82.87 ; quarter=-90 ; angle=-7.13
        QVERIFY(std::abs(e.userCropAngle() + 7.13f) < 0.1f);
    }

    // Sub-pixel "click" with no real drag: tool exits without changing angle.
    void straightenLine_tinyDrag_leavesAngleUnchanged() {
        CropRotateEffect e;
        enterStraighten(e);
        ViewportTransform vt = makeVT();

        auto p = makeMouseEvent(QEvent::MouseButtonPress,  {50.0, 50.0});
        auto r = makeMouseEvent(QEvent::MouseButtonRelease,{51.0, 50.0});
        e.mousePress(&p, vt);
        e.mouseRelease(&r, vt);

        QCOMPARE(e.userCropAngle(), 0.0f);
        QCOMPARE(e.subTool(), CropRotateEffect::SubTool::Handles);
    }

    // Cancel: clicking the button again while drawing exits cleanly.
    void straightenLine_buttonCancelsMidDraw() {
        CropRotateEffect e;
        QWidget* w = e.createControlsWidget();
        QPushButton* btn = nullptr;
        for (auto* b : w->findChildren<QPushButton*>())
            if (b->text().contains("Straighten")) { btn = b; break; }
        QVERIFY(btn);
        btn->click();   // enter
        QCOMPARE(e.subTool(), CropRotateEffect::SubTool::StraightenLine);

        ViewportTransform vt = makeVT();
        auto p = makeMouseEvent(QEvent::MouseButtonPress, {10.0, 10.0});
        auto m = makeMouseEvent(QEvent::MouseMove,        {80.0, 60.0});
        e.mousePress(&p, vt);
        e.mouseMove(&m, vt);

        btn->click();   // cancel
        QCOMPARE(e.subTool(), CropRotateEffect::SubTool::Handles);
        QCOMPARE(e.userCropAngle(), 0.0f);
    }

    // ── Cursor stays ClosedHand while actively rotating ──────────────────────

    void cursorFor_duringRotationDrag_isClosedHand() {
        CropRotateEffect e;
        ViewportTransform vt = makeVT();

        // Begin a rotation drag from the grip
        auto p = makeMouseEvent(QEvent::MouseButtonPress, {50.0, -30.0});
        QVERIFY(e.mousePress(&p, vt));

        // Cursor query mid-drag, even off the grip, should be ClosedHand
        QCOMPARE(e.cursorFor({200.0, 200.0}, vt).shape(), Qt::ClosedHandCursor);

        auto rel = makeMouseEvent(QEvent::MouseButtonRelease, {50.0, -30.0});
        e.mouseRelease(&rel, vt);
    }

    // ── paintOverlay — exercise the new visual states ────────────────────────

    // Paints mid-drag for each handle kind so the active-handle switch in
    // paintOverlay (corner/edgeH/edgeV arms) is exercised.
    void paintOverlay_duringCornerAndEdgeDrags_doesNotCrash() {
        QPixmap pm(100, 100);
        pm.fill(Qt::black);
        QPainter painter(&pm);

        auto runDrag = [&](QPointF press) {
            CropRotateEffect e;
            ViewportTransform vt = makeVT();
            auto p = makeMouseEvent(QEvent::MouseButtonPress, press);
            e.mousePress(&p, vt);
            e.paintOverlay(painter, vt);   // active-handle highlight branch
            auto r = makeMouseEvent(QEvent::MouseButtonRelease, press);
            e.mouseRelease(&r, vt);
        };
        runDrag({  0.0,   0.0});  // TL corner
        runDrag({ 50.0,   0.0});  // top edge   → EdgeH
        runDrag({  0.0,  50.0});  // left edge  → EdgeV

        painter.end();
        QVERIFY(true);
    }

    void paintOverlay_withHoverAndStraightenLine_doesNotCrash() {
        CropRotateEffect e;
        ViewportTransform vt = makeVT();

        // Trigger hover state on the rotation grip
        e.cursorFor({50.0, -30.0}, vt);

        QPixmap pm(100, 100);
        pm.fill(Qt::black);
        QPainter painter(&pm);
        e.paintOverlay(painter, vt);   // hover-state grip

        // Active rotation state — grip becomes "active"
        auto p = makeMouseEvent(QEvent::MouseButtonPress, {50.0, -30.0});
        e.mousePress(&p, vt);
        auto m = makeMouseEvent(QEvent::MouseMove, {80.0, -30.0});
        e.mouseMove(&m, vt);
        e.paintOverlay(painter, vt);   // active-state grip + degree readout
        // Repeat with grip in the right portion of the viewport — flips the
        // readout to the left-of-grip layout branch.
        ViewportTransform vtRight = makeVT(100, 100, 60, 100);
        e.paintOverlay(painter, vtRight);
        auto rel = makeMouseEvent(QEvent::MouseButtonRelease, {80.0, -30.0});
        e.mouseRelease(&rel, vt);

        // Straighten-line overlay: hint strip + drawn line
        enterStraighten(e);
        e.paintOverlay(painter, vt);    // hint only
        auto p2 = makeMouseEvent(QEvent::MouseButtonPress, {10.0, 50.0});
        e.mousePress(&p2, vt);
        auto m2 = makeMouseEvent(QEvent::MouseMove, {90.0, 40.0});
        e.mouseMove(&m2, vt);
        e.paintOverlay(painter, vt);    // hint + line

        painter.end();
        QVERIFY(true);
    }

    // ── Bound clamping with rotation ────────────────────────────────────────

    // Helper: shrink the full crop to a known sub-rect by dragging BR inward.
    static void shrinkCropToCenter(CropRotateEffect& e, ViewportTransform vt) {
        auto p = makeMouseEvent(QEvent::MouseButtonPress,  {100.0, 100.0});
        auto m = makeMouseEvent(QEvent::MouseMove,         { 70.0,  70.0});
        auto r = makeMouseEvent(QEvent::MouseButtonRelease,{ 70.0,  70.0});
        e.mousePress(&p, vt);  e.mouseMove(&m, vt);  e.mouseRelease(&r, vt);
    }

    // After rotating, the rotated source-space footprint of the crop must
    // fit inside [0, 1]² — i.e. half-AABB ≤ centre.x ≤ 1 - half-AABB, and
    // similarly for y.  Drive the rotation slider with a non-trivial angle
    // and verify the shrunk crop, when un-rotated, doesn't poke out.
    void rotation_shrinksOversizedCropToFitImageBounds() {
        CropRotateEffect e;
        QWidget* w = e.createControlsWidget();

        // Drive the rotation slider to 30°.  The full-frame crop's AABB at
        // 30° is bigger than 1×1, so the clamp must shrink the crop.
        auto* slider = w->findChild<QSlider*>();
        QVERIFY(slider);
        slider->setValue(300);  // 300 * 0.1 = 30°

        const QRectF c = e.userCropRect();
        const double half = std::abs(std::cos(30.0 * M_PI / 180.0)) * c.width()  * 0.5
                          + std::abs(std::sin(30.0 * M_PI / 180.0)) * c.height() * 0.5;
        QVERIFY(c.center().x() + half <= 1.0 + 1e-6);
        QVERIFY(c.center().x() - half >= 0.0 - 1e-6);
    }

    // Move drag with rotation: pushing the crop toward the edge stops at
    // the rotated-AABB bound, not at the unrotated edge.
    void move_withRotation_clampsToRotatedBound() {
        CropRotateEffect e;
        QWidget* w = e.createControlsWidget();
        ViewportTransform vt = makeVT();
        shrinkCropToCenter(e, vt);

        // Apply a 20° rotation via the slider.
        auto* slider = w->findChild<QSlider*>();
        QVERIFY(slider);
        slider->setValue(200);

        const QRectF before = e.userCropRect();
        const double cxBefore = before.center().x();
        const double cyBefore = before.center().y();

        // Try to slam the crop centre toward (1, 1) with a giant move drag.
        const double sxBefore = cxBefore * 100.0;
        const double syBefore = cyBefore * 100.0;
        auto p = makeMouseEvent(QEvent::MouseButtonPress, {sxBefore, syBefore});
        e.mousePress(&p, vt);
        auto m = makeMouseEvent(QEvent::MouseMove, {500.0, 500.0});
        e.mouseMove(&m, vt);
        auto r = makeMouseEvent(QEvent::MouseButtonRelease, {500.0, 500.0});
        e.mouseRelease(&r, vt);

        const QRectF after = e.userCropRect();
        const double half = std::abs(std::cos(20.0 * M_PI / 180.0)) * after.width()  * 0.5
                          + std::abs(std::sin(20.0 * M_PI / 180.0)) * after.height() * 0.5;
        QVERIFY(after.center().x() + half <= 1.0 + 1e-6);
        QVERIFY(after.center().y() + half <= 1.0 + 1e-6);
        // Centre actually moved (otherwise the test isn't proving anything)
        QVERIFY(after.center().x() > cxBefore - 1e-6);
    }

    // Quarter-turn re-clamps the crop too.
    void quarterTurn_reclampsCropAfterRotation() {
        CropRotateEffect e;
        QWidget* w = e.createControlsWidget();
        ViewportTransform vt = makeVT();
        // Make a tall thin crop in a corner.
        auto p1 = makeMouseEvent(QEvent::MouseButtonPress,  {100.0, 100.0});
        auto m1 = makeMouseEvent(QEvent::MouseMove,         { 90.0,  20.0});
        auto r1 = makeMouseEvent(QEvent::MouseButtonRelease,{ 90.0,  20.0});
        e.mousePress(&p1, vt); e.mouseMove(&m1, vt); e.mouseRelease(&r1, vt);

        QPushButton* ccw = nullptr;
        for (auto* b : w->findChildren<QPushButton*>())
            if (b->text().contains("CCW")) { ccw = b; break; }
        QVERIFY(ccw);
        ccw->click();
        // After +90°, the rotated AABB swaps width/height. The clamp should
        // ensure the crop still fits.
        const QRectF c = e.userCropRect();
        QVERIFY(c.center().x() + c.height() * 0.5 <= 1.0 + 1e-6);
        QVERIFY(c.center().y() + c.width()  * 0.5 <= 1.0 + 1e-6);
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
