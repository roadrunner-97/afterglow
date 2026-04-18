#ifndef UNSHARPEFFECT_H
#define UNSHARPEFFECT_H

#include "PhotoEditorEffect.h"
#include "IGpuEffect.h"

class ParamSlider;

class UnsharpEffect : public PhotoEditorEffect, public IGpuEffect {
    Q_OBJECT

public:
    UnsharpEffect();
    ~UnsharpEffect() override;

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
    ParamSlider* amountParam;   // 0.0 – 5.0: sharpening strength
    ParamSlider* radiusParam;   // 1 – 15:   Gaussian blur radius for the mask
    ParamSlider* thresholdParam;// 0 – 20:   minimum channel diff to sharpen

    cl::Context m_pipelineCtx;  // saved in initGpuKernels for temp buffer allocation
    // Pipeline (float4 linear) kernels.
    cl::Kernel  m_kernelBlurHLinear;
    cl::Kernel  m_kernelBlurVLinear;
    cl::Kernel  m_kernelUnsharpLinear;
};

#endif // UNSHARPEFFECT_H
