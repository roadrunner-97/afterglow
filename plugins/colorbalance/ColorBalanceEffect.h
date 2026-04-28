#ifndef COLORBALANCEEFFECT_H
#define COLORBALANCEEFFECT_H

#include "PhotoEditorEffect.h"
#include "IGpuEffect.h"

class ParamSlider;

class ColorBalanceEffect : public PhotoEditorEffect, public IGpuEffect {
    Q_OBJECT

public:
    ColorBalanceEffect();
    ~ColorBalanceEffect() override;

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
    QWidget*     controlsWidget;
    ParamSlider* shadowRParam;
    ParamSlider* shadowGParam;
    ParamSlider* shadowBParam;
    ParamSlider* midtoneRParam;
    ParamSlider* midtoneGParam;
    ParamSlider* midtoneBParam;
    ParamSlider* highlightRParam;
    ParamSlider* highlightGParam;
    ParamSlider* highlightBParam;

    // GPU pipeline kernel (float4 linear, compiled into the shared pipeline context).
    cl::Kernel m_kernelLinear;
};

#endif // COLORBALANCEEFFECT_H
