#ifndef GRAYSCALEEFFECT_H
#define GRAYSCALEEFFECT_H

#include "PhotoEditorEffect.h"
#include "IGpuEffect.h"

class GrayscaleEffect : public PhotoEditorEffect, public IGpuEffect {
    Q_OBJECT

public:
    GrayscaleEffect();
    ~GrayscaleEffect() override;

    QString getName() const override;
    QString getDescription() const override;
    QString getVersion() const override;
    bool initialize() override;
    QWidget* createControlsWidget() override;

    bool initGpuKernels(cl::Context& ctx, cl::Device& dev) override;
    bool enqueueGpu(cl::CommandQueue& queue, cl::Buffer& buf, cl::Buffer& aux,
                    int w, int h,
                    const QMap<QString, QVariant>& params) override;

private:
    // GPU pipeline kernel (float4 linear, compiled into the shared pipeline context).
    cl::Kernel m_kernelLinear;
    bool m_active = false;
};

#endif // GRAYSCALEEFFECT_H
