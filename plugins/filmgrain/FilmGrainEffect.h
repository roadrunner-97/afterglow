#ifndef FILMGRAINEFFECT_H
#define FILMGRAINEFFECT_H

#include "PhotoEditorEffect.h"
#include "IGpuEffect.h"

class ParamSlider;
class QCheckBox;

class FilmGrainEffect : public PhotoEditorEffect, public IGpuEffect {
    Q_OBJECT

public:
    FilmGrainEffect();
    ~FilmGrainEffect() override;

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
    ParamSlider* amountParam;
    ParamSlider* sizeParam;
    ParamSlider* seedParam;
    QCheckBox*   lumWeightBox;

    cl::Kernel m_kernel;
    cl::Kernel m_kernel16;
};

#endif // FILMGRAINEFFECT_H
