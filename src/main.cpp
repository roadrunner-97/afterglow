#include <QApplication>
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

int main(int argc, char *argv[]) {
    QApplication app(argc, argv);

    GpuDeviceRegistry::instance().enumerate();

    EffectManager* effects = new EffectManager();
    effects->addEffect(new HotPixelEffect());
    effects->addEffect(new ExposureEffect());
    effects->addEffect(new WhiteBalanceEffect());
    effects->addEffect(new BrightnessEffect());
    effects->addEffect(new SaturationEffect());
    effects->addEffect(new BlurEffect());
    effects->addEffect(new GrayscaleEffect());
    effects->addEffect(new UnsharpEffect());
    effects->addEffect(new DenoiseEffect());
    effects->addEffect(new VignetteEffect());
    effects->addEffect(new FilmGrainEffect());
    effects->addEffect(new SplitToningEffect());

    for (const auto& e : effects->entries())
        e.effect->initialize();

    PhotoEditorApp window(effects);
    window.show();

    return app.exec();
}
