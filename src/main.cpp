#include <QApplication>
#include <QFileInfo>
#include <memory>
#include "ui/PhotoEditorApp.h"
#include "ui/Appearance.h"
#include "core/EffectManager.h"
#include "core/GpuDeviceRegistry.h"
#include "HotPixelEffect.h"
#include "ExposureEffect.h"
#include "WhiteBalanceEffect.h"
#include "BrightnessEffect.h"
#include "SaturationEffect.h"
#include "BlurEffect.h"
#include "GrayscaleEffect.h"
#include "UnsharpEffect.h"
#include "DenoiseEffect.h"
#include "VignetteEffect.h"
#include "FilmGrainEffect.h"
#include "SplitToningEffect.h"
#include "ClarityEffect.h"
#include "ColorBalanceEffect.h"
#include "CropRotateEffect.h"

// Create a fully-initialized EffectManager with the standard effect set.
// Called twice: once for the Develop pipeline and once for the background proofer.
static std::unique_ptr<EffectManager> makeEffects() {
    auto mgr = std::make_unique<EffectManager>();
    mgr->addEffect(std::make_unique<CropRotateEffect>());
    mgr->addEffect(std::make_unique<HotPixelEffect>());
    mgr->addEffect(std::make_unique<ExposureEffect>());
    mgr->addEffect(std::make_unique<WhiteBalanceEffect>());
    mgr->addEffect(std::make_unique<BrightnessEffect>());
    mgr->addEffect(std::make_unique<SaturationEffect>());
    mgr->addEffect(std::make_unique<BlurEffect>());
    mgr->addEffect(std::make_unique<GrayscaleEffect>());
    mgr->addEffect(std::make_unique<UnsharpEffect>());
    mgr->addEffect(std::make_unique<DenoiseEffect>());
    mgr->addEffect(std::make_unique<VignetteEffect>());
    mgr->addEffect(std::make_unique<FilmGrainEffect>());
    mgr->addEffect(std::make_unique<SplitToningEffect>());
    mgr->addEffect(std::make_unique<ClarityEffect>());
    mgr->addEffect(std::make_unique<ColorBalanceEffect>());
    for (const auto &e : mgr->entries()) e.effect->initialize();
    return mgr;
}

int main(int argc, char *argv[]) {
    QApplication app(argc, argv);
    Appearance::initialize();

    GpuDeviceRegistry::instance().enumerate();

    std::unique_ptr<EffectManager> effects = makeEffects();
    PhotoEditorApp                 window(effects.get());
    window.initProofer(makeEffects());
    window.show();

    if (argc > 1) window.openImagePath(QFileInfo(QString::fromLocal8Bit(argv[1])).absoluteFilePath());

    return app.exec();
}
