#ifndef VIGNETTEEFFECT_H
#define VIGNETTEEFFECT_H

#include "PhotoEditorEffect.h"
#include "IGpuEffect.h"
#include "IInteractiveEffect.h"

class ParamSlider;

class VignetteEffect : public PhotoEditorEffect, public IGpuEffect, public IInteractiveEffect {
    Q_OBJECT

public:
    VignetteEffect();
    ~VignetteEffect() override;

    QString getName() const override;
    QString getDescription() const override;
    QString getVersion() const override;
    bool    initialize() override;

    QWidget                *createControlsWidget() override;
    QMap<QString, QVariant> getParameters() const override;
    void                    applyParameters(const QMap<QString, QVariant> &parameters) override;

    bool initGpuKernels(cl::Context &ctx, cl::Device &dev) override;
    bool enqueueGpu(cl::CommandQueue &queue, cl::Buffer &buf, cl::Buffer &aux, int w, int h,
                    const QMap<QString, QVariant> &params) override;

    void    paintOverlay(QPainter &painter, const ViewportTransform &vt) override;
    bool    mousePress(QMouseEvent *event, const ViewportTransform &vt) override;
    bool    mouseMove(QMouseEvent *event, const ViewportTransform &vt) override;
    bool    mouseRelease(QMouseEvent *event, const ViewportTransform &vt) override;
    QCursor cursorFor(QPointF screenPx, const ViewportTransform &vt) override;

private:
    QWidget     *controlsWidget;
    ParamSlider *amountParam;
    ParamSlider *midpointParam;
    ParamSlider *featherParam;
    ParamSlider *roundnessParam;
    ParamSlider *centerXParam;
    ParamSlider *centerYParam;
    QPointF      m_center{0.5, 0.5};
    bool         m_draggingCenter = false;

    // GPU pipeline kernel (float4 linear, compiled into the shared pipeline context).
    cl::Kernel m_kernelLinear;
};

#endif // VIGNETTEEFFECT_H
