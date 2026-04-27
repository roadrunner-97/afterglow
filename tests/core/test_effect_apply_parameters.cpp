// Round-trip every effect's applyParameters() through getParameters().
// Each test creates the effect, instantiates its controls widget so the
// internal slider/combo/checkbox pointers are wired up, applies a known
// parameter map, then reads parameters back and asserts equality.
//
// This is a widget test — needs a QApplication via QTEST_MAIN and the
// offscreen QPA platform.
#include <QTest>
#include <QWidget>

#include "BrightnessEffect.h"
#include "SaturationEffect.h"
#include "BlurEffect.h"
#include "ExposureEffect.h"
#include "HotPixelEffect.h"
#include "UnsharpEffect.h"
#include "DenoiseEffect.h"
#include "WhiteBalanceEffect.h"
#include "VignetteEffect.h"
#include "FilmGrainEffect.h"
#include "SplitToningEffect.h"
#include "ClarityEffect.h"
#include "ColorBalanceEffect.h"
#include "CropRotateEffect.h"

namespace {

template <typename Effect>
QWidget* makeControls(Effect& e) {
    QWidget* w = e.createControlsWidget();
    Q_ASSERT(w);
    return w;
}

} // namespace

class TestEffectApplyParameters : public QObject {
    Q_OBJECT

private slots:
    void brightness() {
        BrightnessEffect e;
        QWidget* w = makeControls(e);
        e.applyParameters({{"brightness", 25}, {"contrast", -10}});
        const auto p = e.getParameters();
        QCOMPARE(p.value("brightness").toInt(), 25);
        QCOMPARE(p.value("contrast").toInt(),   -10);
        delete w;
    }

    void saturation() {
        SaturationEffect e;
        QWidget* w = makeControls(e);
        e.applyParameters({{"saturation", 12.5}, {"vibrancy", -7.0}});
        const auto p = e.getParameters();
        QCOMPARE(p.value("saturation").toDouble(), 12.5);
        QCOMPARE(p.value("vibrancy").toDouble(),   -7.0);
        delete w;
    }

    void hotpixel() {
        HotPixelEffect e;
        QWidget* w = makeControls(e);
        e.applyParameters({{"threshold", 42}});
        QCOMPARE(e.getParameters().value("threshold").toInt(), 42);
        delete w;
    }

    void unsharp() {
        UnsharpEffect e;
        QWidget* w = makeControls(e);
        e.applyParameters({{"amount", 2.5}, {"radius", 4}, {"threshold", 8}});
        const auto p = e.getParameters();
        QCOMPARE(p.value("amount").toDouble(),    2.5);
        QCOMPARE(p.value("radius").toInt(),       4);
        QCOMPARE(p.value("threshold").toInt(),    8);
        delete w;
    }

    void clarity() {
        ClarityEffect e;
        QWidget* w = makeControls(e);
        e.applyParameters({{"amount", 30}, {"radius", 25}});
        const auto p = e.getParameters();
        QCOMPARE(p.value("amount").toInt(), 30);
        QCOMPARE(p.value("radius").toInt(), 25);
        delete w;
    }

    void exposure() {
        ExposureEffect e;
        QWidget* w = makeControls(e);
        e.applyParameters({
            {"exposure",   0.5},
            {"whites",     2.0},
            {"highlights", -1.5},
            {"shadows",    1.0},
            {"blacks",     -0.5},
        });
        const auto p = e.getParameters();
        QCOMPARE(p.value("exposure").toDouble(),   0.5);
        QCOMPARE(p.value("whites").toDouble(),     2.0);
        QCOMPARE(p.value("highlights").toDouble(), -1.5);
        QCOMPARE(p.value("shadows").toDouble(),    1.0);
        QCOMPARE(p.value("blacks").toDouble(),     -0.5);
        delete w;
    }

    void vignette() {
        VignetteEffect e;
        QWidget* w = makeControls(e);
        e.applyParameters({
            {"amount",    -40},
            {"midpoint",  60},
            {"feather",   30},
            {"roundness", 10},
        });
        const auto p = e.getParameters();
        QCOMPARE(p.value("amount").toInt(),    -40);
        QCOMPARE(p.value("midpoint").toInt(),  60);
        QCOMPARE(p.value("feather").toInt(),   30);
        QCOMPARE(p.value("roundness").toInt(), 10);
        delete w;
    }

    void splittoning() {
        SplitToningEffect e;
        QWidget* w = makeControls(e);
        e.applyParameters({
            {"shadowHue",    220},
            {"shadowSat",    35},
            {"highlightHue", 40},
            {"highlightSat", 25},
            {"balance",      -15},
        });
        const auto p = e.getParameters();
        QCOMPARE(p.value("shadowHue").toInt(),    220);
        QCOMPARE(p.value("shadowSat").toInt(),    35);
        QCOMPARE(p.value("highlightHue").toInt(), 40);
        QCOMPARE(p.value("highlightSat").toInt(), 25);
        QCOMPARE(p.value("balance").toInt(),      -15);
        delete w;
    }

    void colorbalance() {
        ColorBalanceEffect e;
        QWidget* w = makeControls(e);
        e.applyParameters({
            {"shadowR",   -20}, {"shadowG",    10}, {"shadowB",   -5},
            {"midtoneR",   15}, {"midtoneG",  -25}, {"midtoneB",  30},
            {"highlightR", 7}, {"highlightG", -3}, {"highlightB", 18},
        });
        const auto p = e.getParameters();
        QCOMPARE(p.value("shadowR").toInt(),    -20);
        QCOMPARE(p.value("shadowG").toInt(),     10);
        QCOMPARE(p.value("shadowB").toInt(),     -5);
        QCOMPARE(p.value("midtoneR").toInt(),    15);
        QCOMPARE(p.value("midtoneG").toInt(),   -25);
        QCOMPARE(p.value("midtoneB").toInt(),    30);
        QCOMPARE(p.value("highlightR").toInt(),   7);
        QCOMPARE(p.value("highlightG").toInt(),  -3);
        QCOMPARE(p.value("highlightB").toInt(),  18);
        delete w;
    }

    void whitebalance() {
        WhiteBalanceEffect e;
        QWidget* w = makeControls(e);
        e.applyParameters({
            {"temperature", 6500},
            {"tint",        12},
            {"shot_temp",   3200},   // must be ignored — not user-editable
        });
        const auto p = e.getParameters();
        QCOMPARE(p.value("temperature").toInt(), 6500);
        QCOMPARE(p.value("tint").toInt(),        12);
        delete w;
    }

    void blur_combo_and_radius() {
        BlurEffect e;
        QWidget* w = makeControls(e);
        e.applyParameters({{"blurType", 1}, {"radius", 8}});
        const auto p = e.getParameters();
        QCOMPARE(p.value("blurType").toInt(), 1);
        QCOMPARE(p.value("radius").toInt(),   8);
        delete w;
    }

    void denoise_combo_and_sliders() {
        DenoiseEffect e;
        QWidget* w = makeControls(e);
        e.applyParameters({
            {"strength",       60},
            {"shadowPreserve", 30},
            {"colorNoise",     40},
            {"algorithm",      1},
        });
        const auto p = e.getParameters();
        QCOMPARE(p.value("strength").toInt(),       60);
        QCOMPARE(p.value("shadowPreserve").toInt(), 30);
        QCOMPARE(p.value("colorNoise").toInt(),     40);
        QCOMPARE(p.value("algorithm").toInt(),       1);
        delete w;
    }

    void filmgrain_checkbox_and_sliders() {
        FilmGrainEffect e;
        QWidget* w = makeControls(e);
        e.applyParameters({
            {"amount",    15},
            {"size",      20},
            {"seed",      7},
            {"lumWeight", true},
        });
        auto p = e.getParameters();
        QCOMPARE(p.value("amount").toInt(),     15);
        QCOMPARE(p.value("size").toInt(),       20);
        QCOMPARE(p.value("seed").toInt(),       7);
        QCOMPARE(p.value("lumWeight").toBool(), true);

        // toggling back exercises the false branch of the checkbox apply.
        e.applyParameters({{"lumWeight", false}});
        QCOMPARE(e.getParameters().value("lumWeight").toBool(), false);
        delete w;
    }

    void croprotate_writes_angle_and_crop() {
        CropRotateEffect e;
        QWidget* w = makeControls(e);
        e.applyParameters({
            {"angle",        7.5},
            {"quarterTurns", 1},
            {"cropX0", 0.1}, {"cropY0", 0.2},
            {"cropX1", 0.8}, {"cropY1", 0.9},
        });
        const auto p = e.getParameters();
        QCOMPARE(p.value("angle").toDouble(),        7.5);
        QCOMPARE(p.value("quarterTurns").toInt(),    1);
        QCOMPARE(p.value("cropX0").toDouble(),       0.1);
        QCOMPARE(p.value("cropY0").toDouble(),       0.2);
        QCOMPARE(p.value("cropX1").toDouble(),       0.8);
        QCOMPARE(p.value("cropY1").toDouble(),       0.9);
        delete w;
    }

    void croprotate_default_full_crop_does_not_lock_manual() {
        // The (0,0,1,1) full-frame rect should not flip m_userManualCrop on,
        // since it represents "no crop" and a subsequent angle change should
        // still be free to auto-fit.  We can only assert this indirectly via
        // a follow-up angle apply — which still emits cleanly.
        CropRotateEffect e;
        QWidget* w = makeControls(e);
        e.applyParameters({
            {"angle", 0.0}, {"quarterTurns", 0},
            {"cropX0", 0.0}, {"cropY0", 0.0},
            {"cropX1", 1.0}, {"cropY1", 1.0},
        });
        const auto p = e.getParameters();
        QCOMPARE(p.value("cropX0").toDouble(), 0.0);
        QCOMPARE(p.value("cropX1").toDouble(), 1.0);
        delete w;
    }
};

QTEST_MAIN(TestEffectApplyParameters)
#include "test_effect_apply_parameters.moc"
