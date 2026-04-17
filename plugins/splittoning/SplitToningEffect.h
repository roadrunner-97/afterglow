#ifndef SPLITTONINGEFFECT_H
#define SPLITTONINGEFFECT_H

#include "PhotoEditorEffect.h"
#include "IGpuEffect.h"

class ParamSlider;

class SplitToningEffect : public PhotoEditorEffect, public IGpuEffect {
    Q_OBJECT

public:
    SplitToningEffect();
    ~SplitToningEffect() override;

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
    ParamSlider* shadowHueParam;
    ParamSlider* shadowSatParam;
    ParamSlider* highlightHueParam;
    ParamSlider* highlightSatParam;
    ParamSlider* balanceParam;

    cl::Kernel m_kernel;
    cl::Kernel m_kernel16;
};

#endif // SPLITTONINGEFFECT_H
