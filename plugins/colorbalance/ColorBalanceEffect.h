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
    QImage processImage(const QImage &image, const QMap<QString, QVariant> &parameters = QMap<QString, QVariant>()) override;

    QWidget* createControlsWidget() override;
    QMap<QString, QVariant> getParameters() const override;

    bool supportsGpuInPlace() const override { return true; }
    bool initGpuKernels(cl::Context& ctx, cl::Device& dev) override;
    bool enqueueGpu(cl::CommandQueue& queue, cl::Buffer& buf, cl::Buffer& aux,
                    int w, int h, int stride, bool is16bit,
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

    cl::Kernel m_kernel;
    cl::Kernel m_kernel16;
};

#endif // COLORBALANCEEFFECT_H
