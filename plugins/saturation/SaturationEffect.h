#ifndef SATURATIONEFFECT_H
#define SATURATIONEFFECT_H

#include "PhotoEditorEffect.h"
#include "IGpuEffect.h"

class ParamSlider;

class SaturationEffect : public PhotoEditorEffect, public IGpuEffect {
    Q_OBJECT

public:
    SaturationEffect();
    ~SaturationEffect() override;

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
    ParamSlider* saturationParam;
    ParamSlider* vibrancyParam;

    cl::Kernel m_kernel;
    cl::Kernel m_kernel16;
};

#endif // SATURATIONEFFECT_H
