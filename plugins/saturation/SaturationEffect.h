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
    void applyParameters(const QMap<QString, QVariant>& parameters) override;

    bool initGpuKernels(cl::Context& ctx, cl::Device& dev) override;
    bool enqueueGpu(cl::CommandQueue& queue, cl::Buffer& buf, cl::Buffer& aux,
                    int w, int h,
                    const QMap<QString, QVariant>& params) override;

private:
    QWidget*     controlsWidget;
    ParamSlider* saturationParam;
    ParamSlider* vibrancyParam;

    // GPU pipeline kernel (float4 linear, compiled into the shared pipeline context).
    // The 8-bit and 16-bit sRGB kernels live only in the per-effect processImage
    // path (tests); the pipeline uses only m_kernelLinear.
    cl::Kernel m_kernelLinear;
};

#endif // SATURATIONEFFECT_H
