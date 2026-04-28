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
    void applyParameters(const QMap<QString, QVariant>& parameters) override;

    bool initGpuKernels(cl::Context& ctx, cl::Device& dev) override;
    bool enqueueGpu(cl::CommandQueue& queue, cl::Buffer& buf, cl::Buffer& aux,
                    int w, int h,
                    const QMap<QString, QVariant>& params) override;

private:
    QWidget*     controlsWidget;
    QComboBox*   blurTypeCombo;
    ParamSlider* radiusParam;

    int blurType;  // 0 = Gaussian, 1 = Box

    // Pipeline (float4 linear) kernels.  The sRGB kernels used by processImage
    // live in the per-effect GpuContext (tests path) and aren't duplicated here.
    cl::Kernel m_kernelBlurHLinear;
    cl::Kernel m_kernelBlurVLinear;
};

#endif // BLUREFFECT_H
