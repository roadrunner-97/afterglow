#ifndef BLUREFFECT_H
#define BLUREFFECT_H

#include "PhotoEditorEffect.h"
#include "IGpuEffect.h"

class ParamSlider;
class QComboBox;
class QLabel;

class BlurEffect : public PhotoEditorEffect, public IGpuEffect {
    Q_OBJECT

public:
    BlurEffect();
    ~BlurEffect() override;

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
    QComboBox*   blurTypeCombo;
    ParamSlider* radiusParam;

    int blurType;  // 0 = Gaussian, 1 = Box

    cl::Kernel m_kernelH;
    cl::Kernel m_kernelV;
    cl::Kernel m_kernelH16;
    cl::Kernel m_kernelV16;
};

#endif // BLUREFFECT_H
