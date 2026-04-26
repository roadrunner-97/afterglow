#ifndef CROPROTATEEFFECT_H
#define CROPROTATEEFFECT_H

#include "PhotoEditorEffect.h"
#include "IGpuEffect.h"
#include "IInteractiveEffect.h"
#include "ICropSource.h"

#include <QRectF>
#include <QPointF>

class ParamSlider;

class CropRotateEffect : public PhotoEditorEffect,
                         public IGpuEffect,
                         public IInteractiveEffect,
                         public ICropSource
{
    Q_OBJECT

public:
    enum class SubTool { Handles, StraightenLine, RotationGrip };

    CropRotateEffect();
    ~CropRotateEffect() override;

    // PhotoEditorEffect
    QString getName()        const override;
    QString getDescription() const override;
    QString getVersion()     const override;
    bool    initialize()           override;

    QImage processImage(const QImage& image,
                        const QMap<QString, QVariant>& parameters = {}) override;

    QWidget*                 createControlsWidget() override;
    QMap<QString, QVariant>  getParameters()  const override;

    bool supportsGpuInPlace() const override { return true; }

    // IGpuEffect — no-op: crop/rotate is non-destructive metadata only
    bool initGpuKernels(cl::Context& ctx, cl::Device& dev) override;
    bool enqueueGpu(cl::CommandQueue& queue,
                    cl::Buffer& buf, cl::Buffer& aux,
                    int w, int h,
                    const QMap<QString, QVariant>& params) override;

    // ICropSource
    QRectF userCropRect()  const override;
    float  userCropAngle() const override;

    // IInteractiveEffect
    void    paintOverlay (QPainter& painter, const ViewportTransform& vt) override;
    bool    mousePress   (QMouseEvent* event, const ViewportTransform& vt) override;
    bool    mouseMove    (QMouseEvent* event, const ViewportTransform& vt) override;
    bool    mouseRelease (QMouseEvent* event, const ViewportTransform& vt) override;
    QCursor cursorFor    (QPointF screenPx,   const ViewportTransform& vt) override;

    // Accessors for testing
    SubTool subTool()      const { return m_subTool; }
    int     quarterTurns() const { return m_quarterTurns; }

private:
    // ── State ──────────────────────────────────────────────────────────────
    QRectF  m_crop{0.0, 0.0, 1.0, 1.0}; // normalised [0..1]
    float   m_angleDeg   = 0.0f;         // fine rotation, range ±45°
    int     m_quarterTurns = 0;           // 0..3, each = 90° CCW
    SubTool m_subTool    = SubTool::Handles;

    // ── UI ─────────────────────────────────────────────────────────────────
    QWidget*    m_controlsWidget = nullptr;
    ParamSlider* m_angleSlider   = nullptr;

    // ── Drag state ─────────────────────────────────────────────────────────
    enum class DragKind {
        None,
        Corner,    // 0=TL 1=TR 2=BR 3=BL
        EdgeH,     // 0=top 1=bottom
        EdgeV,     // 0=left 1=right
        Move,
        Rotation
    };
    DragKind m_dragKind  = DragKind::None;
    int      m_dragIndex = 0;
    QPointF  m_dragStart;   // screen coords where drag began
    QRectF   m_dragCropStart;
    float    m_dragAngleStart = 0.0f;

    // ── Helpers ────────────────────────────────────────────────────────────
    // Crop rect corners/edges in SOURCE pixel coords
    struct Handles {
        QPointF tl, tr, br, bl;      // corners
        QPointF tm, bm, lm, rm;      // edge midpoints
        QPointF rotGrip;             // rotation grip (above top edge)
    };
    Handles buildHandles(const ViewportTransform& vt) const;

    // Returns handle index if screen point is within hit radius, else -1.
    // Returns handle encoding: 0-3=corner(TL/TR/BR/BL), 4-7=edge(T/B/L/R),
    // 8=rotation grip, -1=miss.
    int hitHandle(QPointF screenPx, const Handles& h) const;

    // Is screenPx inside the projected crop rect (not on a handle)?
    bool insideCrop(QPointF screenPx, const ViewportTransform& vt) const;

    static constexpr float HIT_RADIUS    = 8.0f;    // screen pixels
    static constexpr float MIN_CROP_SIZE = 0.05f;   // normalised units
    static constexpr float ROT_GRIP_OFFSET = 30.0f; // screen pixels above top edge

    // TODO (StraightenLine): store two line endpoints here when that feature is
    // implemented:
    //   QPointF m_lineP1, m_lineP2;  // screen coords, set in mousePress/mouseMove
    //   bool    m_lineSet = false;
    // On mouseRelease, compute atan2(p2-p1), derive the horizon angle, and update
    // m_angleDeg accordingly.
};

#endif // CROPROTATEEFFECT_H
