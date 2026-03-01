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
                    int w, int h, int stride, bool is16bit,
                    const QMap<QString, QVariant>& params) override;

private:
    QWidget*     m_controls;
    ParamSlider* m_strengthParam;       // 0–100: overall denoise blend
    ParamSlider* m_shadowPreserveParam; // 0–100: protect dark regions
    ParamSlider* m_colorNoiseParam;     // 0–100: chroma smoothing

    cl::Context m_pipelineCtx;  // saved in initGpuKernels for temp buffer allocation
    cl::Kernel  m_kernelBlurH;
    cl::Kernel  m_kernelBlurV;
    cl::Kernel  m_kernelDenoiseBlend;
    cl::Kernel  m_kernelColorNoiseBlend;
    cl::Kernel  m_kernelBlurH16;
    cl::Kernel  m_kernelBlurV16;
    cl::Kernel  m_kernelDenoiseBlend16;
    cl::Kernel  m_kernelColorNoiseBlend16;
};

#endif // DENOISEEFFECT_H
