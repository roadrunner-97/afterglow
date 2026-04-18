#ifndef DENOISEEFFECT_H
#define DENOISEEFFECT_H

#include "PhotoEditorEffect.h"
#include "IGpuEffect.h"

class ParamSlider;

class DenoiseEffect : public PhotoEditorEffect, public IGpuEffect {
    Q_OBJECT

public:
    DenoiseEffect();
    ~DenoiseEffect() override;

    QString getName() const override;
    QString getDescription() const override;
    QString getVersion() const override;
    bool initialize() override;
    QImage processImage(const QImage &image,
                        const QMap<QString, QVariant> &parameters = QMap<QString, QVariant>()) override;

    QWidget* createControlsWidget() override;
    QMap<QString, QVariant> getParameters() const override;

    bool supportsGpuInPlace() const override { return true; }
    bool initGpuKernels(cl::Context& ctx, cl::Device& dev) override;
    bool enqueueGpu(cl::CommandQueue& queue, cl::Buffer& buf, cl::Buffer& aux,
                    int w, int h,
                    const QMap<QString, QVariant>& params) override;

private:
    QWidget*     m_controls;
    ParamSlider* m_strengthParam;       // 0–100: overall denoise blend
    ParamSlider* m_shadowPreserveParam; // 0–100: protect dark regions
    ParamSlider* m_colorNoiseParam;     // 0–100: chroma smoothing

    cl::Context m_pipelineCtx;  // saved in initGpuKernels for temp buffer allocation
    // Pipeline (float4 linear) kernels.
    cl::Kernel  m_kernelBlurHLinear;
    cl::Kernel  m_kernelBlurVLinear;
    cl::Kernel  m_kernelDenoiseBlendLinear;
    cl::Kernel  m_kernelColorNoiseBlendLinear;
};

#endif // DENOISEEFFECT_H
