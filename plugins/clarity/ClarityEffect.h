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
                    int w, int h, int stride, bool is16bit,
                    const QMap<QString, QVariant>& params) override;

private:
    QWidget*     controlsWidget;
    ParamSlider* amountParam;   // -100 – 100: clarity strength (%)
    ParamSlider* radiusParam;   //   10 – 100: blur radius in source pixels

    cl::Context m_pipelineCtx;  // saved in initGpuKernels for temp buffer allocation
    cl::Kernel  m_kernelH;
    cl::Kernel  m_kernelV;
    cl::Kernel  m_kernelClarity;
    cl::Kernel  m_kernelH16;
    cl::Kernel  m_kernelV16;
    cl::Kernel  m_kernelClarity16;
};

#endif // CLARITYEFFECT_H
