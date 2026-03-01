#ifndef WHITEBALANCEEFFECT_H
#define WHITEBALANCEEFFECT_H

#include "PhotoEditorEffect.h"
#include "IGpuEffect.h"

class ParamSlider;

class WhiteBalanceEffect : public PhotoEditorEffect, public IGpuEffect {
    Q_OBJECT

public:
    WhiteBalanceEffect();
    ~WhiteBalanceEffect() override;

    QString getName() const override;
    QString getDescription() const override;
    QString getVersion() const override;
    bool initialize() override;
    QImage processImage(const QImage &image, const QMap<QString, QVariant> &parameters = QMap<QString, QVariant>()) override;

    QWidget* createControlsWidget() override;
    QMap<QString, QVariant> getParameters() const override;

    // Sets the slider default to the image's as-shot color temperature.
    void onImageLoaded(const ImageMetadata& meta) override;

    bool supportsGpuInPlace() const override { return true; }
    bool initGpuKernels(cl::Context& ctx, cl::Device& dev) override;
    bool enqueueGpu(cl::CommandQueue& queue, cl::Buffer& buf, cl::Buffer& aux,
                    int w, int h, int stride, bool is16bit,
                    const QMap<QString, QVariant>& params) override;

private:
    QWidget*     controlsWidget;
    ParamSlider* temperatureParam;  // 2000 – 12000 K
    ParamSlider* tintParam;         // −100 (green) … +100 (magenta)

    float m_shotK = 5500.0f;  // as-shot color temperature; slider default

    cl::Kernel m_kernel;
    cl::Kernel m_kernel16;
};

#endif // WHITEBALANCEEFFECT_H
