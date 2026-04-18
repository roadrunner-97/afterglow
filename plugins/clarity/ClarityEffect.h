#ifndef CLARITYEFFECT_H
#define CLARITYEFFECT_H

#include "PhotoEditorEffect.h"
#include "IGpuEffect.h"

class ParamSlider;

class ClarityEffect : public PhotoEditorEffect, public IGpuEffect {
    Q_OBJECT

public:
    ClarityEffect();
    ~ClarityEffect() override;

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
    ParamSlider* amountParam;   // -100 – 100: clarity strength (%)
    ParamSlider* radiusParam;   //   10 – 100: blur radius in source pixels

    cl::Context m_pipelineCtx;  // saved in initGpuKernels for temp buffer allocation
    // Pipeline (float4 linear) kernels.
    cl::Kernel  m_kernelBlurHLinear;
    cl::Kernel  m_kernelBlurVLinear;
    cl::Kernel  m_kernelClarityLinear;

    // Cached scratch buffer for the blurred image.  Reused across calls;
    // reallocated only when preview dimensions change (or context is reset).
    cl::Buffer  m_blurBuf;
    int         m_blurBufW = 0;
    int         m_blurBufH = 0;
};

#endif // CLARITYEFFECT_H
