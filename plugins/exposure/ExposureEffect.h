#ifndef EXPOSUREEFFECT_H
#define EXPOSUREEFFECT_H

#include "PhotoEditorEffect.h"
#include "IGpuEffect.h"
#include <cstdint>
#include <functional>
#include <vector>

class ParamSlider;

class ExposureEffect : public PhotoEditorEffect, public IGpuEffect {
    Q_OBJECT

public:
    ExposureEffect();
    ~ExposureEffect() override;

    QString getName() const override;
    QString getDescription() const override;
    QString getVersion() const override;
    bool initialize() override;

    QWidget* createControlsWidget() override;
    QMap<QString, QVariant> getParameters() const override;
    void applyParameters(const QMap<QString, QVariant>& parameters) override;
    void onImageLoaded(const ImageMetadata& meta) override;

    bool initGpuKernels(cl::Context& ctx, cl::Device& dev) override;
    bool enqueueGpu(cl::CommandQueue& queue, cl::Buffer& buf, cl::Buffer& aux,
                    int w, int h,
                    const QMap<QString, QVariant>& params) override;

private:
    QWidget*     controlsWidget;
    ParamSlider* exposureParam;
    ParamSlider* whitesParam;
    ParamSlider* highlightsParam;
    ParamSlider* shadowsParam;
    ParamSlider* blacksParam;

    // Callback that pushes a histogram into the tone-curve widget.  Wired in
    // createControlsWidget() (ToneCurveWidget is an anon-namespace type, so
    // the widget pointer stays hidden behind this lambda).  Empty until the
    // panel is built.
    std::function<void(const std::vector<uint32_t>&)> m_applyHistogram;

    // Cached input-luminance histogram from the most recent onImageLoaded.
    // Pushed to the widget when the panel is built (image may be loaded
    // before the Exposure panel is ever expanded).
    std::vector<uint32_t> m_histogram;

    // GPU pipeline kernel (float4 linear, compiled into the shared pipeline context).
    cl::Kernel m_kernelLinear;
};

#endif // EXPOSUREEFFECT_H
