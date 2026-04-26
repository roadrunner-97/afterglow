#include "CropRotateEffect.h"
#include "ParamSlider.h"

#include <QFont>
#include <QFontMetrics>
#include <QMouseEvent>
#include <QPainter>
#include <QPainterPath>
#include <QHBoxLayout>
#include <QPushButton>
#include <QVBoxLayout>

// OpenCL headers are required by IGpuEffect.h; the actual GPU functions are no-ops.
#define CL_HPP_TARGET_OPENCL_VERSION 200
#define CL_HPP_MINIMUM_OPENCL_VERSION 110
#define CL_HPP_ENABLE_EXCEPTIONS
#include <CL/opencl.hpp>

#include <cmath>
#include <algorithm>

// ============================================================================
// Construction / meta
// ============================================================================

CropRotateEffect::CropRotateEffect() = default;
CropRotateEffect::~CropRotateEffect() = default;

QString CropRotateEffect::getName()        const { return "Crop & Rotate"; }
QString CropRotateEffect::getDescription() const {
    return "Non-destructive crop and rotation";
}
QString CropRotateEffect::getVersion()     const { return "1.0"; }
bool    CropRotateEffect::initialize()           { return true; }

// ============================================================================
// processImage — passthrough (crop/rotate is metadata only)
// ============================================================================

QImage CropRotateEffect::processImage(const QImage& image,
                                       const QMap<QString, QVariant>& /*parameters*/) {
    return image;
}

// ============================================================================
// createControlsWidget
// ============================================================================

QWidget* CropRotateEffect::createControlsWidget() {
    if (m_controlsWidget) return m_controlsWidget;

    m_controlsWidget = new QWidget();
    auto* layout = new QVBoxLayout(m_controlsWidget);
    layout->setContentsMargins(0, 0, 0, 0);
    layout->setSpacing(8);

    // ── Rotation slider ─────────────────────────────────────────────────────
    m_angleSlider = new ParamSlider("Rotation", -45.0, 45.0, 0.1, 1);
    connect(m_angleSlider, &ParamSlider::valueChanged, this, [this](double v) {
        m_angleDeg = static_cast<float>(v);
        emit liveParametersChanged();
    });
    connect(m_angleSlider, &ParamSlider::editingFinished, this, [this]() {
        emit parametersChanged();
    });
    layout->addWidget(m_angleSlider);

    // ── Button row: 90° turns ───────────────────────────────────────────────
    auto* row1 = new QHBoxLayout();
    auto* btn90ccw = new QPushButton("Rotate 90° CCW");
    auto* btn90cw  = new QPushButton("Rotate 90° CW");
    row1->addWidget(btn90ccw);
    row1->addWidget(btn90cw);
    layout->addLayout(row1);

    connect(btn90ccw, &QPushButton::clicked, this, [this]() {
        m_quarterTurns = (m_quarterTurns + 1) % 4;
        emit parametersChanged();
    });
    connect(btn90cw, &QPushButton::clicked, this, [this]() {
        m_quarterTurns = (m_quarterTurns + 3) % 4;   // +3 mod 4 = -1 mod 4
        emit parametersChanged();
    });

    // ── Reset Crop ───────────────────────────────────────────────────────────
    auto* btnReset = new QPushButton("Reset Crop");
    connect(btnReset, &QPushButton::clicked, this, [this]() {
        m_crop        = QRectF(0.0, 0.0, 1.0, 1.0);
        m_angleDeg    = 0.0f;
        m_quarterTurns = 0;
        if (m_angleSlider) {
            m_angleSlider->blockSignals(true);
            m_angleSlider->setValue(0.0);
            m_angleSlider->blockSignals(false);
        }
        emit parametersChanged();
    });
    layout->addWidget(btnReset);

    // ── Straighten by Line ───────────────────────────────────────────────────
    m_straightenButton = new QPushButton("Straighten by Line");
    m_straightenButton->setToolTip(
        "Drag along a line in the image that should be horizontal or vertical");
    connect(m_straightenButton, &QPushButton::clicked, this, [this]() {
        if (m_subTool == SubTool::StraightenLine) {
            m_subTool = SubTool::Handles;
            m_lineDrawing = false;
            m_straightenButton->setText("Straighten by Line");
        } else {
            m_subTool = SubTool::StraightenLine;
            m_lineDrawing = false;
            m_straightenButton->setText("Cancel Straighten");
        }
    });
    layout->addWidget(m_straightenButton);

    layout->addStretch();
    return m_controlsWidget;
}

// ============================================================================
// getParameters
// ============================================================================

QMap<QString, QVariant> CropRotateEffect::getParameters() const {
    QMap<QString, QVariant> p;
    p["angle"]        = static_cast<double>(m_angleDeg);
    p["quarterTurns"] = m_quarterTurns;
    p["cropX0"]       = m_crop.x();
    p["cropY0"]       = m_crop.y();
    p["cropX1"]       = m_crop.x() + m_crop.width();
    p["cropY1"]       = m_crop.y() + m_crop.height();
    return p;
}

// ============================================================================
// IGpuEffect — no-op (crop/rotate is pure metadata, no kernel work)
// ============================================================================

// GCOVR_EXCL_START
bool CropRotateEffect::initGpuKernels(cl::Context& /*ctx*/, cl::Device& /*dev*/) {
    return true;
}

bool CropRotateEffect::enqueueGpu(cl::CommandQueue& /*queue*/,
                                   cl::Buffer& /*buf*/, cl::Buffer& /*aux*/,
                                   int /*w*/, int /*h*/,
                                   const QMap<QString, QVariant>& /*params*/) {
    return true;
}
// GCOVR_EXCL_STOP

// ============================================================================
// ICropSource
// ============================================================================

QRectF CropRotateEffect::userCropRect()  const { return m_crop; }
float  CropRotateEffect::userCropAngle() const {
    return static_cast<float>(m_quarterTurns) * 90.0f + m_angleDeg;
}

// ============================================================================
// IInteractiveEffect helpers
// ============================================================================

// Rotation in screen coords.  userCropAngle() is CCW-positive; screen Y is
// down so a screen-CCW rotation uses R(θ) = [[cos,  sin], [-sin, cos]].
static QPointF rotateAround(QPointF p, QPointF centre, float angleRad) {
    const float cosA = std::cos(angleRad);
    const float sinA = std::sin(angleRad);
    const float dx = static_cast<float>(p.x() - centre.x());
    const float dy = static_cast<float>(p.y() - centre.y());
    return { centre.x() + cosA * dx + sinA * dy,
             centre.y() - sinA * dx + cosA * dy };
}

// Inverse of rotateAround — un-rotates a point back into the rect's axis-
// aligned local frame.
static QPointF unrotateAround(QPointF p, QPointF centre, float angleRad) {
    return rotateAround(p, centre, -angleRad);
}

CropRotateEffect::Handles
CropRotateEffect::buildHandles(const ViewportTransform& vt) const {
    const float iw = static_cast<float>(vt.imageSize.width());
    const float ih = static_cast<float>(vt.imageSize.height());

    // Crop rect corners in source pixels (axis-aligned)
    const float x0 = static_cast<float>(m_crop.x())                    * iw;
    const float y0 = static_cast<float>(m_crop.y())                    * ih;
    const float x1 = static_cast<float>(m_crop.x() + m_crop.width())   * iw;
    const float y1 = static_cast<float>(m_crop.y() + m_crop.height())  * ih;

    const QPointF centreSrc((x0 + x1) * 0.5, (y0 + y1) * 0.5);
    const float angleRad = userCropAngle() * static_cast<float>(M_PI) / 180.0f;

    auto rs = [&](float x, float y) {
        return vt.sourceToScreen(rotateAround({x, y}, centreSrc, angleRad));
    };

    Handles h;
    h.tl = rs(x0, y0);
    h.tr = rs(x1, y0);
    h.br = rs(x1, y1);
    h.bl = rs(x0, y1);
    h.tm = rs((x0 + x1) * 0.5f, y0);
    h.bm = rs((x0 + x1) * 0.5f, y1);
    h.lm = rs(x0, (y0 + y1) * 0.5f);
    h.rm = rs(x1, (y0 + y1) * 0.5f);

    // Rotation grip: offset outward from the (rotated) top-edge midpoint,
    // along the direction from crop centre to that midpoint in screen space.
    const QPointF centreScreen = vt.sourceToScreen(centreSrc);
    QPointF toTop = h.tm - centreScreen;
    const double len = std::hypot(toTop.x(), toTop.y());
    if (len > 1e-3)
        h.rotGrip = h.tm + (toTop / len) * ROT_GRIP_OFFSET;
    else
        h.rotGrip = QPointF(h.tm.x(), h.tm.y() - ROT_GRIP_OFFSET); // GCOVR_EXCL_LINE

    return h;
}

static float screenDist2(QPointF a, QPointF b) {
    float dx = static_cast<float>(a.x() - b.x());
    float dy = static_cast<float>(a.y() - b.y());
    return dx * dx + dy * dy;
}

int CropRotateEffect::hitHandle(QPointF screenPx, const Handles& h) const {
    const float r2 = HIT_RADIUS * HIT_RADIUS;
    // Corners: 0=TL 1=TR 2=BR 3=BL
    if (screenDist2(screenPx, h.tl) <= r2) return 0;
    if (screenDist2(screenPx, h.tr) <= r2) return 1;
    if (screenDist2(screenPx, h.br) <= r2) return 2;
    if (screenDist2(screenPx, h.bl) <= r2) return 3;
    // Edge midpoints: 4=top 5=bottom 6=left 7=right
    if (screenDist2(screenPx, h.tm) <= r2) return 4;
    if (screenDist2(screenPx, h.bm) <= r2) return 5;
    if (screenDist2(screenPx, h.lm) <= r2) return 6;
    if (screenDist2(screenPx, h.rm) <= r2) return 7;
    // Rotation grip: 8
    if (screenDist2(screenPx, h.rotGrip) <= r2) return 8;
    return -1;
}

bool CropRotateEffect::insideCrop(QPointF screenPx, const ViewportTransform& vt) const {
    const QPointF src = vt.screenToSource(screenPx);
    const float iw = static_cast<float>(vt.imageSize.width());
    const float ih = static_cast<float>(vt.imageSize.height());
    if (iw <= 0.0f || ih <= 0.0f) return false;

    // Un-rotate the source point into the crop's axis-aligned frame before
    // testing containment.
    const QPointF centreSrc(m_crop.center().x() * iw, m_crop.center().y() * ih);
    const float angleRad = userCropAngle() * static_cast<float>(M_PI) / 180.0f;
    const QPointF local = unrotateAround(src, centreSrc, angleRad);

    return m_crop.contains(QPointF(local.x() / iw, local.y() / ih));
}

// ============================================================================
// paintOverlay
// ============================================================================

void CropRotateEffect::paintOverlay(QPainter& painter, const ViewportTransform& vt) {
    if (vt.imageSize.isEmpty()) return;

    painter.save();
    painter.setRenderHint(QPainter::Antialiasing, false);

    const Handles h = buildHandles(vt);

    // ── Dimming outside the crop rect ────────────────────────────────────────
    {
        const QRectF viewport(QPointF(0, 0), QSizeF(vt.viewportSize));
        QPolygonF cropPoly;
        cropPoly << h.tl << h.tr << h.br << h.bl << h.tl;

        // Use a clip inversion: fill whole viewport, then erase crop area.
        painter.save();
        // Start with a full-viewport clip, subtract the crop polygon.
        QPainterPath fullPath;
        fullPath.addRect(viewport);
        QPainterPath cropPath;
        cropPath.addPolygon(cropPoly);
        QPainterPath outerPath = fullPath.subtracted(cropPath);
        painter.setClipPath(outerPath);
        painter.fillRect(viewport, QColor(0, 0, 0, 128));
        painter.restore();
    }

    // ── Rule of thirds grid ───────────────────────────────────────────────────
    {
        painter.save();
        QPen gridPen(QColor(255, 255, 255, 60), 1.0);
        painter.setPen(gridPen);

        // Horizontal thirds
        for (int i = 1; i <= 2; ++i) {
            float t = static_cast<float>(i) / 3.0f;
            QPointF lft = h.tl + (h.bl - h.tl) * static_cast<double>(t);
            QPointF rgt = h.tr + (h.br - h.tr) * static_cast<double>(t);
            painter.drawLine(lft, rgt);
        }
        // Vertical thirds
        for (int i = 1; i <= 2; ++i) {
            float t = static_cast<float>(i) / 3.0f;
            QPointF top = h.tl + (h.tr - h.tl) * static_cast<double>(t);
            QPointF bot = h.bl + (h.br - h.bl) * static_cast<double>(t);
            painter.drawLine(top, bot);
        }
        painter.restore();
    }

    // ── Crop rect outline ────────────────────────────────────────────────────
    {
        painter.save();
        QPen outlinePen(Qt::white, 2.0);
        painter.setPen(outlinePen);
        painter.setBrush(Qt::NoBrush);
        QPolygonF poly;
        poly << h.tl << h.tr << h.br << h.bl;
        painter.drawPolygon(poly);
        painter.restore();
    }

    // ── Handle squares ───────────────────────────────────────────────────────
    // Each handle has three visual states:
    //   normal — white fill, dark border
    //   hover  — light-blue fill, dark border (mouse over but not pressed)
    //   active — solid blue fill, dark border (currently being dragged)
    {
        painter.save();
        const QPointF corners[4] = { h.tl, h.tr, h.br, h.bl };
        const QPointF edges[4]   = { h.tm, h.bm, h.lm, h.rm };
        const double hs = 5.0;

        auto activeHandleIndex = [&]() -> int {
            switch (m_dragKind) {
                case DragKind::Corner: return m_dragIndex;          // 0..3
                case DragKind::EdgeH:  return 4 + m_dragIndex;      // 4..5
                case DragKind::EdgeV:  return 6 + m_dragIndex;      // 6..7
                default: return -1;
            }
        };
        const int activeIdx = activeHandleIndex();

        auto drawHandle = [&](QPointF c, int idx) {
            QRectF r(c.x() - hs, c.y() - hs, hs * 2.0, hs * 2.0);
            painter.setPen(QPen(QColor(50, 50, 50), 1.5));
            QColor fill = Qt::white;
            if (idx == activeIdx)             fill = QColor( 80, 160, 255);
            else if (idx == m_hoverHandle)    fill = QColor(180, 215, 255);
            painter.setBrush(fill);
            painter.drawRect(r);
        };

        for (int i = 0; i < 4; ++i) drawHandle(corners[i], i);
        for (int i = 0; i < 4; ++i) drawHandle(edges[i],   4 + i);
        painter.restore();
    }

    // ── Rotation grip ────────────────────────────────────────────────────────
    // Visual: outer ring + curved-arrow glyph.  Same three states as handles,
    // and while rotating we also draw a small "+12.3°" readout next to the
    // grip so the user can see what they're committing to.
    {
        painter.save();
        painter.setRenderHint(QPainter::Antialiasing, true);

        const bool gripActive = (m_dragKind == DragKind::Rotation);
        const bool gripHover  = (m_hoverHandle == 8);

        QColor fill   = Qt::white;
        QColor accent = QColor(50, 50, 50);
        if (gripActive)      { fill = QColor( 80, 160, 255); accent = Qt::white; }
        else if (gripHover)  { fill = QColor(180, 215, 255); accent = QColor(20, 60, 120); }

        // Tether line from top edge to grip — dotted normally, solid + accent
        // colour when the grip is hot.  This is the main discoverability cue:
        // a visible "stalk" sticking out of the crop frame says "grab me".
        painter.setPen(QPen((gripActive || gripHover) ? accent : Qt::white,
                            (gripActive || gripHover) ? 1.5    : 1.0,
                            (gripActive || gripHover) ? Qt::SolidLine : Qt::DotLine));
        painter.drawLine(h.tm, h.rotGrip);

        // Filled disc
        painter.setPen(QPen(QColor(50, 50, 50), 1.5));
        painter.setBrush(fill);
        painter.drawEllipse(h.rotGrip, ROT_GRIP_RADIUS, ROT_GRIP_RADIUS);

        // Curved-arrow glyph inside the disc — 270° arc with an arrowhead at
        // one end, communicates "rotation handle" at a glance.
        const double r = ROT_GRIP_RADIUS - 3.5;
        QRectF arcRect(h.rotGrip.x() - r, h.rotGrip.y() - r, r * 2, r * 2);
        painter.setPen(QPen(accent, 1.6));
        painter.setBrush(Qt::NoBrush);
        // Qt arc angles: 0° at 3 o'clock, CCW positive, 1/16 of a degree units.
        const int startDeg16 = 45 * 16;
        const int spanDeg16  = 270 * 16;
        painter.drawArc(arcRect, startDeg16, spanDeg16);

        // Arrowhead at the arc's CCW end (315°).  Compute end point and a
        // tangent direction, then draw a small triangle.
        const double endRad = (45 - 270) * M_PI / 180.0;  // -225°  == 135°
        const QPointF endPt(h.rotGrip.x() + r * std::cos(endRad),
                            h.rotGrip.y() - r * std::sin(endRad));
        const double tx = -std::sin(endRad);
        const double ty = -std::cos(endRad);
        const double px = -ty, py = tx;  // perpendicular
        const double t = 3.0, w = 2.2;
        QPolygonF arrow;
        arrow << endPt
              << QPointF(endPt.x() + tx * t + px * w, endPt.y() + ty * t + py * w)
              << QPointF(endPt.x() + tx * t - px * w, endPt.y() + ty * t - py * w);
        painter.setBrush(accent);
        painter.setPen(Qt::NoPen);
        painter.drawPolygon(arrow);

        // Live degree readout while rotating
        if (gripActive) {
            QFont f = painter.font();
            f.setPointSizeF(10.5);
            f.setBold(true);
            painter.setFont(f);
            const QString text = QString("%1%2°")
                .arg(m_angleDeg >= 0 ? "+" : "")
                .arg(static_cast<double>(m_angleDeg), 0, 'f', 1);
            QFontMetrics fm(f);
            const QRect tb = fm.boundingRect(text);
            // Place the readout to the side of the grip that has more room
            // (left if grip is past the right half of the viewport, else right).
            const double pad = 6.0;
            const bool toLeft = h.rotGrip.x() > vt.viewportSize.width() * 0.66;
            const double bx = toLeft
                ? h.rotGrip.x() - ROT_GRIP_RADIUS - pad - tb.width()  - 6
                : h.rotGrip.x() + ROT_GRIP_RADIUS + pad;
            const double by = h.rotGrip.y() - tb.height() * 0.5 - 3;
            QRectF bg(bx - 4, by, tb.width() + 8, tb.height() + 6);
            painter.setPen(Qt::NoPen);
            painter.setBrush(QColor(0, 0, 0, 180));
            painter.drawRoundedRect(bg, 3, 3);
            painter.setPen(Qt::white);
            painter.drawText(bg, Qt::AlignCenter, text);
        }

        painter.restore();
    }

    // ── Straighten-by-line overlay ───────────────────────────────────────────
    if (m_subTool == SubTool::StraightenLine) {
        painter.save();
        painter.setRenderHint(QPainter::Antialiasing, true);

        // Hint text strip at the top of the viewport
        const QString hint = m_lineDrawing
            ? "Release on the other end of the line"
            : "Drag along a line that should be horizontal or vertical (Esc / button to cancel)";
        QFont f = painter.font();
        f.setPointSizeF(10.0);
        painter.setFont(f);
        QFontMetrics fm(f);
        const QRect tb = fm.boundingRect(hint);
        QRectF strip((vt.viewportSize.width() - tb.width()) * 0.5 - 10, 8,
                     tb.width() + 20, tb.height() + 8);
        painter.setPen(Qt::NoPen);
        painter.setBrush(QColor(0, 0, 0, 180));
        painter.drawRoundedRect(strip, 4, 4);
        painter.setPen(Qt::white);
        painter.drawText(strip, Qt::AlignCenter, hint);

        if (m_lineDrawing) {
            // Bright line + small disc at each endpoint
            painter.setPen(QPen(QColor(255, 200, 60), 2.0));
            painter.drawLine(m_lineP1, m_lineP2);
            painter.setBrush(QColor(255, 200, 60));
            painter.setPen(QPen(QColor(40, 40, 40), 1.0));
            painter.drawEllipse(m_lineP1, 3.5, 3.5);
            painter.drawEllipse(m_lineP2, 3.5, 3.5);
        }
        painter.restore();
    }

    painter.restore();
}

// ============================================================================
// mousePress
// ============================================================================

bool CropRotateEffect::mousePress(QMouseEvent* event, const ViewportTransform& vt) {
    if (event->button() != Qt::LeftButton) return false;

    // ── StraightenLine sub-tool ──────────────────────────────────────────────
    // Click-and-drag: press = P1, drag updates P2, release computes the angle.
    if (m_subTool == SubTool::StraightenLine) {
        m_lineP1 = QPointF(event->pos());
        m_lineP2 = m_lineP1;
        m_lineDrawing = true;
        return true;
    }

    const Handles handles = buildHandles(vt);
    int hit = hitHandle(QPointF(event->pos()), handles);

    m_dragStart     = QPointF(event->pos());
    m_dragCropStart = m_crop;
    m_dragAngleStart = m_angleDeg;

    if (hit >= 0 && hit <= 3) {
        m_dragKind  = DragKind::Corner;
        m_dragIndex = hit;
        return true;
    }
    if (hit == 4 || hit == 5) {
        m_dragKind  = DragKind::EdgeH;
        m_dragIndex = hit - 4;   // 0=top 1=bottom
        return true;
    }
    if (hit == 6 || hit == 7) {
        m_dragKind  = DragKind::EdgeV;
        m_dragIndex = hit - 6;   // 0=left 1=right
        return true;
    }
    if (hit == 8) {
        m_dragKind = DragKind::Rotation;
        return true;
    }

    if (insideCrop(QPointF(event->pos()), vt)) {
        m_dragKind = DragKind::Move;
        return true;
    }

    return false;
}

// ============================================================================
// mouseMove
// ============================================================================

bool CropRotateEffect::mouseMove(QMouseEvent* event, const ViewportTransform& vt) {
    if (m_subTool == SubTool::StraightenLine && m_lineDrawing) {
        m_lineP2 = QPointF(event->pos());
        return true;
    }
    if (m_dragKind == DragKind::None) return false;

    const QPointF curScreen = QPointF(event->pos());

    if (m_dragKind == DragKind::Rotation) {
        // Angle = atan2 from crop centre to mouse position
        const float iw = static_cast<float>(vt.imageSize.width());
        const float ih = static_cast<float>(vt.imageSize.height());
        const float cx = static_cast<float>(m_dragCropStart.center().x()) * iw;
        const float cy = static_cast<float>(m_dragCropStart.center().y()) * ih;
        const QPointF centreScreen = vt.sourceToScreen({cx, cy});

        float dx = static_cast<float>(curScreen.x() - centreScreen.x());
        float dy = static_cast<float>(curScreen.y() - centreScreen.y());
        float angleDrag = std::atan2(dy, dx) * 180.0f / static_cast<float>(M_PI);
        // Normalise to ±45
        float startDx = static_cast<float>(m_dragStart.x() - centreScreen.x());
        float startDy = static_cast<float>(m_dragStart.y() - centreScreen.y());
        float angleStart = std::atan2(startDy, startDx) * 180.0f / static_cast<float>(M_PI);
        float delta = angleDrag - angleStart;
        // Clamp
        float newAngle = std::max(-45.0f, std::min(45.0f, m_dragAngleStart + delta));
        m_angleDeg = newAngle;
        // GCOVR_EXCL_START  (UI-gated — only when controls widget has been built)
        if (m_angleSlider) {
            m_angleSlider->blockSignals(true);
            m_angleSlider->setValue(static_cast<double>(m_angleDeg));
            m_angleSlider->blockSignals(false);
        }
        // GCOVR_EXCL_STOP
        emit liveParametersChanged();
        return true;
    }

    // Convert delta in screen space to normalised crop coords.  When the
    // crop is rotated, un-rotate the delta so corner/edge drags resize along
    // the crop's own tilted axes rather than the screen axes.
    const float iw = static_cast<float>(vt.imageSize.width());
    const float ih = static_cast<float>(vt.imageSize.height());
    if (iw <= 0.0f || ih <= 0.0f) return false;

    const QPointF srcStart = vt.screenToSource(m_dragStart);
    const QPointF srcCur   = vt.screenToSource(curScreen);
    const float angleRad = userCropAngle() * static_cast<float>(M_PI) / 180.0f;
    const QPointF rawDelta = srcCur - srcStart;
    const QPointF localDelta = unrotateAround(rawDelta, {0, 0}, angleRad);
    const float dnx = static_cast<float>(localDelta.x()) / iw;
    const float dny = static_cast<float>(localDelta.y()) / ih;

    float x0 = static_cast<float>(m_dragCropStart.x());
    float y0 = static_cast<float>(m_dragCropStart.y());
    float x1 = x0 + static_cast<float>(m_dragCropStart.width());
    float y1 = y0 + static_cast<float>(m_dragCropStart.height());

    auto clamp01 = [](float v) { return std::max(0.0f, std::min(1.0f, v)); };

    if (m_dragKind == DragKind::Move) {
        float w = x1 - x0;
        float h = y1 - y0;
        float nx0 = clamp01(x0 + dnx);
        float ny0 = clamp01(y0 + dny);
        // Keep size; clamp the opposite edge
        if (nx0 + w > 1.0f) nx0 = 1.0f - w;
        if (ny0 + h > 1.0f) ny0 = 1.0f - h;
        m_crop = QRectF(static_cast<double>(nx0),
                        static_cast<double>(ny0),
                        static_cast<double>(w),
                        static_cast<double>(h));
    } else if (m_dragKind == DragKind::Corner) {
        // 0=TL 1=TR 2=BR 3=BL
        switch (m_dragIndex) {
        case 0: // TL — moves x0, y0
            x0 = clamp01(x0 + dnx);
            y0 = clamp01(y0 + dny);
            if (x1 - x0 < MIN_CROP_SIZE) x0 = x1 - MIN_CROP_SIZE;
            if (y1 - y0 < MIN_CROP_SIZE) y0 = y1 - MIN_CROP_SIZE;
            break;
        case 1: // TR — moves x1, y0
            x1 = clamp01(x1 + dnx);
            y0 = clamp01(y0 + dny);
            if (x1 - x0 < MIN_CROP_SIZE) x1 = x0 + MIN_CROP_SIZE;
            if (y1 - y0 < MIN_CROP_SIZE) y0 = y1 - MIN_CROP_SIZE;
            break;
        case 2: // BR — moves x1, y1
            x1 = clamp01(x1 + dnx);
            y1 = clamp01(y1 + dny);
            if (x1 - x0 < MIN_CROP_SIZE) x1 = x0 + MIN_CROP_SIZE;
            if (y1 - y0 < MIN_CROP_SIZE) y1 = y0 + MIN_CROP_SIZE;
            break;
        case 3: // BL — moves x0, y1
            x0 = clamp01(x0 + dnx);
            y1 = clamp01(y1 + dny);
            if (x1 - x0 < MIN_CROP_SIZE) x0 = x1 - MIN_CROP_SIZE;
            if (y1 - y0 < MIN_CROP_SIZE) y1 = y0 + MIN_CROP_SIZE;
            break;
        default: break; // GCOVR_EXCL_LINE
        }
        m_crop = QRectF(static_cast<double>(x0),
                        static_cast<double>(y0),
                        static_cast<double>(x1 - x0),
                        static_cast<double>(y1 - y0));
    } else if (m_dragKind == DragKind::EdgeH) {
        // 0=top 1=bottom
        if (m_dragIndex == 0) {
            y0 = clamp01(y0 + dny);
            if (y1 - y0 < MIN_CROP_SIZE) y0 = y1 - MIN_CROP_SIZE;
        } else {
            y1 = clamp01(y1 + dny);
            if (y1 - y0 < MIN_CROP_SIZE) y1 = y0 + MIN_CROP_SIZE;
        }
        m_crop = QRectF(static_cast<double>(x0),
                        static_cast<double>(y0),
                        static_cast<double>(x1 - x0),
                        static_cast<double>(y1 - y0));
    } else if (m_dragKind == DragKind::EdgeV) {
        // 0=left 1=right
        if (m_dragIndex == 0) {
            x0 = clamp01(x0 + dnx);
            if (x1 - x0 < MIN_CROP_SIZE) x0 = x1 - MIN_CROP_SIZE;
        } else {
            x1 = clamp01(x1 + dnx);
            if (x1 - x0 < MIN_CROP_SIZE) x1 = x0 + MIN_CROP_SIZE;
        }
        m_crop = QRectF(static_cast<double>(x0),
                        static_cast<double>(y0),
                        static_cast<double>(x1 - x0),
                        static_cast<double>(y1 - y0));
    }

    emit liveParametersChanged();
    return true;
}

// ============================================================================
// mouseRelease
// ============================================================================

bool CropRotateEffect::mouseRelease(QMouseEvent* event, const ViewportTransform& vt) {
    (void)vt;
    if (event->button() != Qt::LeftButton) return false;

    // ── StraightenLine: turn the drawn line into a rotation ────────────────
    if (m_subTool == SubTool::StraightenLine && m_lineDrawing) {
        m_lineP2 = QPointF(event->pos());
        m_lineDrawing = false;

        const double dx = m_lineP2.x() - m_lineP1.x();
        const double dy = m_lineP2.y() - m_lineP1.y();
        // Reject sub-pixel "lines" (accidental clicks); just exit the tool.
        if (std::hypot(dx, dy) >= 4.0) {
            // Line angle in screen-CCW (m_angleDeg's convention): screen Y
            // is down, so screen-CCW = -atan2(dy, dx).  Snap to the nearest
            // 90° axis; the rotation needed to align the line with that axis
            // is (axis - lineAngle).
            const double lineDeg = -std::atan2(dy, dx) * 180.0 / M_PI;
            const double quarter = std::round(lineDeg / 90.0) * 90.0;
            const float clamped  = std::clamp(static_cast<float>(quarter - lineDeg),
                                              -45.0f, 45.0f);
            m_angleDeg = clamped;
            // GCOVR_EXCL_START  (UI-gated — only when controls widget exists)
            if (m_angleSlider) {
                m_angleSlider->blockSignals(true);
                m_angleSlider->setValue(static_cast<double>(m_angleDeg));
                m_angleSlider->blockSignals(false);
            }
            // GCOVR_EXCL_STOP
        }

        m_subTool = SubTool::Handles;
        // GCOVR_EXCL_START  (UI-gated)
        if (m_straightenButton) m_straightenButton->setText("Straighten by Line");
        // GCOVR_EXCL_STOP
        emit parametersChanged();
        return true;
    }

    if (m_dragKind == DragKind::None) return false;
    m_dragKind = DragKind::None;
    emit parametersChanged();
    return true;
}

// ============================================================================
// cursorFor
// ============================================================================

QCursor CropRotateEffect::cursorFor(QPointF screenPx, const ViewportTransform& vt) {
    if (m_subTool == SubTool::StraightenLine) {
        m_hoverHandle = -1;
        return QCursor(Qt::CrossCursor);
    }

    // While actively rotating, override the hit-test so the cursor stays
    // ClosedHand even if the mouse drifts off the grip during the drag.
    if (m_dragKind == DragKind::Rotation) {
        m_hoverHandle = 8;
        return QCursor(Qt::ClosedHandCursor);
    }

    const Handles handles = buildHandles(vt);
    int hit = hitHandle(screenPx, handles);
    m_hoverHandle = hit;

    if (hit == 8) return QCursor(Qt::OpenHandCursor);  // rotation grip

    if (hit == 0 || hit == 2) return QCursor(Qt::SizeFDiagCursor);  // TL, BR
    if (hit == 1 || hit == 3) return QCursor(Qt::SizeBDiagCursor);  // TR, BL
    if (hit == 4 || hit == 5) return QCursor(Qt::SizeVerCursor);    // top/bottom
    if (hit == 6 || hit == 7) return QCursor(Qt::SizeHorCursor);    // left/right

    if (insideCrop(screenPx, vt)) return QCursor(Qt::SizeAllCursor);

    return QCursor(Qt::ArrowCursor);
}
