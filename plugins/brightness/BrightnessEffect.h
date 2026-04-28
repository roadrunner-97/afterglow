#ifndef BRIGHTNESSEFFECT_H
#define BRIGHTNESSEFFECT_H

#include "PhotoEditorEffect.h"
#include "IGpuEffect.h"

class ParamSlider;

class BrightnessEffect : public PhotoEditorEffect, public IGpuEffect {
    Q_OBJECT

public:
    BrightnessEffect();
    ~BrightnessEffect() override;

    QString getName() const override;
    QString getDescription() const override;
    QString getVersion() const override;
    bool initialize() override;

    QWidget* createControlsWidget() override;
    QMap<QString, QVariant> getParameters() const override;
    void applyParameters(const QMap<QString, QVariant>& parameters) override;

    bool initGpuKernels(cl::Context& ctx, cl::Device& dev) override;
    bool enqueueGpu(cl::CommandQueue& queue, cl::Buffer& buf, cl::Buffer& aux,
                    int w, int h,
                    const QMap<QString, QVariant>& params) override;

private:
    QWidget*     controlsWidget   = nullptr;
    ParamSlider* brightnessParam  = nullptr;
    ParamSlider* contrastParam    = nullptr;
    cl::Kernel   m_kernelLinear;
};

#endif // BRIGHTNESSEFFECT_H
