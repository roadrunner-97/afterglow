#ifndef VIGNETTEEFFECT_H
#define VIGNETTEEFFECT_H

#include "PhotoEditorEffect.h"
#include "IGpuEffect.h"

class ParamSlider;

class VignetteEffect : public PhotoEditorEffect, public IGpuEffect {
    Q_OBJECT

public:
    VignetteEffect();
    ~VignetteEffect() override;

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
                    int w, int h,
                    const QMap<QString, QVariant>& params) override;

private:
    QWidget*     controlsWidget;
    ParamSlider* amountParam;
    ParamSlider* midpointParam;
    ParamSlider* featherParam;
    ParamSlider* roundnessParam;

    // GPU pipeline kernel (float4 linear, compiled into the shared pipeline context).
    // The 8-bit and 16-bit sRGB kernels live only in the per-effect processImage
    // path (tests); the pipeline uses only m_kernelLinear.
    cl::Kernel m_kernelLinear;
};

#endif // VIGNETTEEFFECT_H
