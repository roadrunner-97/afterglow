#ifndef LINEARGRADIENTTOOL_H
#define LINEARGRADIENTTOOL_H

#include "IInteractiveEffect.h"
#include "LinearGradientMask.h"

#include <QObject>
#include <optional>

class LinearGradientTool : public QObject, public IInteractiveEffect {
    Q_OBJECT

public:
    explicit LinearGradientTool(QObject *parent = nullptr);

    bool                      hasMask() const;
    const LinearGradientMask *mask() const;
    bool                      isCreating() const;
    bool                      isOverlayVisible() const;

    void beginCreation();
    void setMask(const LinearGradientMask &mask);
    void clearMask();
    void setInverted(bool inverted);
    void setOverlayVisible(bool visible);

    void    paintOverlay(QPainter &painter, const ViewportTransform &vt) override;
    bool    mousePress(QMouseEvent *event, const ViewportTransform &vt) override;
    bool    mouseMove(QMouseEvent *event, const ViewportTransform &vt) override;
    bool    mouseRelease(QMouseEvent *event, const ViewportTransform &vt) override;
    bool    keyPress(QKeyEvent *event) override;
    QCursor cursorFor(QPointF screenPx, const ViewportTransform &vt) override;

signals:
    void maskChanged();
    void gestureFinished();
    void creationModeChanged(bool creating);

private:
    enum class Drag { None, Create, Move, Start, End };

    struct Handles {
        QPointF start;
        QPointF center;
        QPointF end;
    };

    static constexpr double HIT_RADIUS = 10.0;

    QPointF screenToNormalized(QPointF screen, const ViewportTransform &vt) const;
    QPointF normalizedToScreen(QPointF normalized, const ViewportTransform &vt) const;
    Handles handles(const ViewportTransform &vt) const;
    Drag    hitTest(QPointF screen, const ViewportTransform &vt) const;
    void    setFromScreenEndpoints(QPointF start, QPointF end, const ViewportTransform &vt);
    void    cancelGesture();

    std::optional<LinearGradientMask> m_mask;
    std::optional<LinearGradientMask> m_beforeGesture;
    Drag                              m_drag           = Drag::None;
    bool                              m_creationMode   = false;
    bool                              m_overlayVisible = true;
    QPointF                           m_anchor;
    QPointF                           m_dragStart;
};

#endif // LINEARGRADIENTTOOL_H
