#include <QApplication>
#include <memory>
#include "ui/PhotoEditorApp.h"
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

int main(int argc, char *argv[]) {
    QApplication app(argc, argv);

    GpuDeviceRegistry::instance().enumerate();

    EffectManager* effects = new EffectManager();
    effects->addEffect(std::make_unique<CropRotateEffect>());
    effects->addEffect(std::make_unique<HotPixelEffect>());
    effects->addEffect(std::make_unique<ExposureEffect>());
    effects->addEffect(std::make_unique<WhiteBalanceEffect>());
    effects->addEffect(std::make_unique<BrightnessEffect>());
    effects->addEffect(std::make_unique<SaturationEffect>());
    effects->addEffect(std::make_unique<BlurEffect>());
    effects->addEffect(std::make_unique<GrayscaleEffect>());
    effects->addEffect(std::make_unique<UnsharpEffect>());
    effects->addEffect(std::make_unique<DenoiseEffect>());
    effects->addEffect(std::make_unique<VignetteEffect>());
    effects->addEffect(std::make_unique<FilmGrainEffect>());
    effects->addEffect(std::make_unique<SplitToningEffect>());
    effects->addEffect(std::make_unique<ClarityEffect>());
    effects->addEffect(std::make_unique<ColorBalanceEffect>());

    for (const auto& e : effects->entries())
        e.effect->initialize();

    PhotoEditorApp window(effects);
    window.show();

    return app.exec();
}
