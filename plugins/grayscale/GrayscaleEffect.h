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
    QImage processImage(const QImage &image, const QMap<QString, QVariant> &parameters = QMap<QString, QVariant>()) override;
    QWidget* createControlsWidget() override;

    bool supportsGpuInPlace() const override { return true; }
    bool initGpuKernels(cl::Context& ctx, cl::Device& dev) override;
    bool enqueueGpu(cl::CommandQueue& queue, cl::Buffer& buf, cl::Buffer& aux,
                    int w, int h, int stride, bool is16bit,
                    const QMap<QString, QVariant>& params) override;

private:
    cl::Kernel m_kernel;
    cl::Kernel m_kernel16;
    bool m_active = false;
};

#endif // GRAYSCALEEFFECT_H
