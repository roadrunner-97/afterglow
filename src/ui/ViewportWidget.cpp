#include "ViewportWidget.h"
#include "IInteractiveEffect.h"
#include <QOpenGLShaderProgram>
#include <QOpenGLBuffer>
#include <QOpenGLVertexArrayObject>
#include <QVector2D>
#include <QKeyEvent>
#include <QWheelEvent>
#include <QMouseEvent>
#include <QPainter>
#include <QDebug>
#include <algorithm>
#include <cmath>

// Fullscreen quad: 4 × (x, y, u, v) in NDC
static const float QUAD_VERTS[] = {
    -1.f, -1.f,  0.f, 1.f,
     1.f, -1.f,  1.f, 1.f,
    -1.f,  1.f,  0.f, 0.f,
     1.f,  1.f,  1.f, 0.f,
};

// The quad VBO is fullscreen NDC.  uRectNdc remaps it to the image's sub-rect
// of the widget (so the surrounding viewport stays at the GL clear colour),
// then the rotation is applied around the pivot in pixel-correct space.
// uRectNdc = (cx, cy, halfW, halfH) of the image rect in NDC.
static const char* VERT_SRC =
    "#version 330 core\n"
    "layout(location=0) in vec2 aPos;\n"
    "layout(location=1) in vec2 aUv;\n"
    "uniform float uAngleRad;\n"
    "uniform vec2  uPivotNdc;\n"
    "uniform vec2  uViewport;\n"
    "uniform vec4  uRectNdc;\n"
    "out vec2 vUv;\n"
    "void main() {\n"
    "    vec2 imgNdc = uRectNdc.xy + aPos * uRectNdc.zw;\n"
    "    vec2 delta = imgNdc - uPivotNdc;\n"
    "    vec2 px = delta * uViewport * 0.5;\n"
    "    float c = cos(uAngleRad);\n"
    "    float s = sin(uAngleRad);\n"
    "    vec2 rotPx = vec2(c * px.x - s * px.y, s * px.x + c * px.y);\n"
    "    vec2 rotNdc = rotPx * 2.0 / uViewport;\n"
    "    gl_Position = vec4(rotNdc + uPivotNdc, 0.0, 1.0);\n"
    "    vUv = aUv;\n"
    "}\n";

static const char* FRAG_SRC =
    "#version 330 core\n"
    "in vec2 vUv;\n"
    "uniform sampler2D uTex;\n"
    "out vec4 fragColor;\n"
    "void main() { fragColor = texture(uTex, vUv); }\n";

ViewportWidget::ViewportWidget(QWidget* parent)
    : QOpenGLWidget(parent)
{
    setMouseTracking(true);
    setFocusPolicy(Qt::StrongFocus);
}

ViewportWidget::~ViewportWidget() {
    makeCurrent();
    if (m_glTexture) glDeleteTextures(1, &m_glTexture);
    m_vbo.destroy();
    m_vao.destroy();
    delete m_shader;
    doneCurrent();
}

// ── GL lifecycle ─────────────────────────────────────────────────────────────

void ViewportWidget::initializeGL() {
    initializeOpenGLFunctions();

    glClearColor(30.f/255.f, 30.f/255.f, 30.f/255.f, 1.f);

    // Fullscreen quad VAO/VBO using Qt wrappers (handle GL 3.0 function resolution)
    m_vao.create();
    m_vao.bind();

    m_vbo.create();
    m_vbo.bind();
    m_vbo.allocate(QUAD_VERTS, sizeof(QUAD_VERTS));

    // attrib 0: position (x, y)
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE,
                          4 * sizeof(float), reinterpret_cast<void*>(0));
    glEnableVertexAttribArray(0);
    // attrib 1: UV (u, v)
    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE,
                          4 * sizeof(float), reinterpret_cast<void*>(2 * sizeof(float)));
    glEnableVertexAttribArray(1);

    m_vao.release();
    m_vbo.release();

    // Shader
    m_shader = new QOpenGLShaderProgram(this);
    if (!m_shader->addShaderFromSourceCode(QOpenGLShader::Vertex, VERT_SRC) ||
        !m_shader->addShaderFromSourceCode(QOpenGLShader::Fragment, FRAG_SRC) ||
        !m_shader->link()) {
        qWarning() << "[ViewportWidget] shader error:" << m_shader->log();
    }

    // Texture is allocated lazily in setImage() — its size tracks the image,
    // not the widget, so there's nothing to do here until we have content.
    glGenTextures(1, &m_glTexture);
    glBindTexture(GL_TEXTURE_2D, m_glTexture);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glBindTexture(GL_TEXTURE_2D, 0);

    // Apply any image that arrived while we were still hidden (e.g. async
    // processor result delivered before the develop page was first shown).
    if (!m_pendingImage.isNull()) {
        QImage pending = std::move(m_pendingImage);
        QPoint offset  = m_pendingOffset;
        m_pendingImage = QImage();
        m_pendingOffset = {};
        setImage(pending, offset);
    }
}

void ViewportWidget::resizeGL(int, int) {
    // Widget resize doesn't change the texture — only setImage() does.
}

void ViewportWidget::paintGL() {
    glClear(GL_COLOR_BUFFER_BIT);
    if (!m_hasContent || !m_shader || !m_vao.isCreated() || !m_glTexture)
        return;

    m_shader->bind();
    m_shader->setUniformValue("uTex", 0);
    // Bind the rotation uniforms every frame.  uViewport must be non-zero
    // (the vertex shader divides by it).  Pivot conversion: m_imgPivotNorm
    // is in normalised image-source coords; the on-screen position depends
    // on pan and zoom, so route it through the same ViewportTransform the
    // overlay uses, then map screen pixels → NDC.
    const float Vw = static_cast<float>(std::max(1, width()));
    const float Vh = static_cast<float>(std::max(1, height()));
    QPointF pivotScreen(Vw * 0.5f, Vh * 0.5f);
    if (!m_imageSize.isEmpty()) {
        pivotScreen = currentTransform().sourceToScreen({
            m_imgPivotNorm.x() * m_imageSize.width(),
            m_imgPivotNorm.y() * m_imageSize.height() });
    }
    const float angleRad = m_imgAngleDeg * static_cast<float>(M_PI) / 180.0f;
    const QVector2D pivotNdc(2.0f * static_cast<float>(pivotScreen.x()) / Vw - 1.0f,
                             1.0f - 2.0f * static_cast<float>(pivotScreen.y()) / Vh);

    // Image rect in NDC.  The image occupies widget pixels
    // [imageOffset.x, imageOffset.x + renderedSize.w] × [...].  Top row in NDC
    // is +1, so the y axis is flipped relative to widget pixels.
    const float x0 = static_cast<float>(m_imageOffset.x());
    const float y0 = static_cast<float>(m_imageOffset.y());
    const float x1 = x0 + static_cast<float>(m_renderedSize.width());
    const float y1 = y0 + static_cast<float>(m_renderedSize.height());
    const float ndcX0 = 2.0f * x0 / Vw - 1.0f;
    const float ndcX1 = 2.0f * x1 / Vw - 1.0f;
    const float ndcYTop    = 1.0f - 2.0f * y0 / Vh;  // smaller y → higher NDC
    const float ndcYBottom = 1.0f - 2.0f * y1 / Vh;
    const QVector4D rectNdc((ndcX0 + ndcX1) * 0.5f,
                            (ndcYTop + ndcYBottom) * 0.5f,
                            (ndcX1 - ndcX0) * 0.5f,
                            (ndcYTop - ndcYBottom) * 0.5f);

    m_shader->setUniformValue("uAngleRad", angleRad);
    m_shader->setUniformValue("uPivotNdc", pivotNdc);
    m_shader->setUniformValue("uViewport", QVector2D(Vw, Vh));
    m_shader->setUniformValue("uRectNdc", rectNdc);

    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, m_glTexture);
    m_vao.bind();
    glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
    m_vao.release();
    m_shader->release();

    if (m_active) {
        QPainter painter(this);
        m_active->paintOverlay(painter, currentTransform());
    }
}

void ViewportWidget::createOrResizeTexture(int w, int h) {
    if (m_textureSize == QSize(w, h)) return;
    glBindTexture(GL_TEXTURE_2D, m_glTexture);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, w, h,
                 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
    glBindTexture(GL_TEXTURE_2D, 0);
    m_textureSize = QSize(w, h);
}

// ── Public API ───────────────────────────────────────────────────────────────

void ViewportWidget::setImageSize(QSize size) {
    m_imageSize = size;
    setCursor(size.isEmpty() ? Qt::ArrowCursor : Qt::OpenHandCursor);
}

void ViewportWidget::setImage(QImage image, QPoint offset) {
    if (image.isNull()) return;

    // GL context may not exist yet if the widget has never been shown
    // (Loupe → Develop with the develop page hidden inside QStackedWidget).
    // Defer the upload to initializeGL().
    if (m_glTexture == 0) {
        m_pendingImage  = std::move(image);
        m_pendingOffset = offset;
        return;
    }

    // Upload CPU image to GL texture on the GL thread (we're on the main thread here).
    makeCurrent();
    QImage rgba = image.convertToFormat(QImage::Format_RGBA8888);
    createOrResizeTexture(rgba.width(), rgba.height());
    glBindTexture(GL_TEXTURE_2D, m_glTexture);
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0,
                    rgba.width(), rgba.height(),
                    GL_RGBA, GL_UNSIGNED_BYTE, rgba.constBits());
    glBindTexture(GL_TEXTURE_2D, 0);
    doneCurrent();

    m_imageOffset  = offset;
    m_renderedSize = rgba.size();
    m_hasContent   = true;
    update();
}

void ViewportWidget::resetView() {
    m_zoom   = 1.0f;
    m_center = {0.5, 0.5};
}

void ViewportWidget::setActiveInteractiveEffect(IInteractiveEffect* effect) {
    m_active = effect;
    update();
}

void ViewportWidget::setImageRotation(float angleDeg, QPointF pivotNorm) {
    m_imgAngleDeg  = angleDeg;
    m_imgPivotNorm = pivotNorm;
    update();
}

ViewportTransform ViewportWidget::currentTransform() const {
    return ViewportTransform{ m_imageSize, size(), m_center, m_zoom };
}

ViewportRequest ViewportWidget::viewportRequest() const {
    return ViewportRequest{size(), m_zoom, m_center};
}

// ── Pan / zoom ───────────────────────────────────────────────────────────────

void ViewportWidget::keyPressEvent(QKeyEvent* event) {
    if (m_imageSize.isEmpty()) { event->ignore(); return; }

    const bool ctrl = event->modifiers() & Qt::ControlModifier;
    const int key   = event->key();

    // Ctrl+0: fit to window
    if (ctrl && key == Qt::Key_0) {
        m_zoom   = 1.0f;
        m_center = {0.5, 0.5};
        emit viewportChanged();
        event->accept();
        return;
    }

    // Ctrl+1: 100% (one image pixel per screen pixel)
    if (ctrl && key == Qt::Key_1) {
        const float W  = m_imageSize.width(), H = m_imageSize.height();
        const float Vw = width(),             Vh = height();
        const float fitScale = std::min(Vw / W, Vh / H);
        m_zoom   = std::clamp(1.0f / fitScale, 1.0f, 64.0f);
        m_center = {0.5, 0.5};
        clampCenter();
        emit viewportChanged();
        event->accept();
        return;
    }

    // +/= zoom in, - zoom out
    if (key == Qt::Key_Plus || key == Qt::Key_Equal) {
        const float newZoom = std::clamp(m_zoom * 1.15f, 1.0f, 64.0f);
        if (newZoom != m_zoom) { m_zoom = newZoom; clampCenter(); emit viewportChanged(); }
        event->accept();
        return;
    }
    if (key == Qt::Key_Minus) {
        const float newZoom = std::clamp(m_zoom / 1.15f, 1.0f, 64.0f);
        if (newZoom != m_zoom) { m_zoom = newZoom; clampCenter(); emit viewportChanged(); }
        event->accept();
        return;
    }

    event->ignore();
}

void ViewportWidget::wheelEvent(QWheelEvent* event) {
    if (m_imageSize.isEmpty()) { event->ignore(); return; }

    const float factor = (event->angleDelta().y() > 0) ? 1.15f : (1.0f / 1.15f);
    const float newZoom = std::clamp(m_zoom * factor, 1.0f, 64.0f);
    if (newZoom == m_zoom) { event->accept(); return; }

    const QPointF mousePos = event->position();
    const float Vw = width(), Vh = height();
    const float W  = m_imageSize.width(), H = m_imageSize.height();

    float fitScale    = std::min(Vw / W, Vh / H);
    float displayScale = fitScale * m_zoom;
    float regionW = Vw / displayScale, regionH = Vh / displayScale;
    float x0 = (float)m_center.x() * W - regionW * 0.5f;
    float y0 = (float)m_center.y() * H - regionH * 0.5f;

    float imgX = x0 + (mousePos.x() / Vw) * regionW;
    float imgY = y0 + (mousePos.y() / Vh) * regionH;

    float newDisplayScale = fitScale * newZoom;
    float newRegionW = Vw / newDisplayScale, newRegionH = Vh / newDisplayScale;

    float newX0 = imgX - (mousePos.x() / Vw) * newRegionW;
    float newY0 = imgY - (mousePos.y() / Vh) * newRegionH;

    m_zoom = newZoom;
    m_center.setX((newX0 + newRegionW * 0.5f) / W);
    m_center.setY((newY0 + newRegionH * 0.5f) / H);
    clampCenter();

    event->accept();
    emit viewportChanged();
}

void ViewportWidget::mousePressEvent(QMouseEvent* event) {
    if (m_active) {
        const ViewportTransform vt = currentTransform();
        if (m_active->mousePress(event, vt)) { update(); event->accept(); return; }
    }
    const bool isPan = (event->button() == Qt::LeftButton || event->button() == Qt::MiddleButton);
    if (isPan && !m_imageSize.isEmpty()) {
        m_panning      = true;
        m_lastMousePos = event->pos();
        setCursor(Qt::ClosedHandCursor);
        event->accept();
    }
}

void ViewportWidget::mouseMoveEvent(QMouseEvent* event) {
    if (m_active) {
        const ViewportTransform vt = currentTransform();
        if (event->buttons() != Qt::NoButton) {
            if (m_active->mouseMove(event, vt)) { update(); event->accept(); return; }
        } else {
            // Hover: update cursor and repaint so the overlay can reflect
            // hover state (e.g. handle highlights).  Mouse-move events are
            // batched against vsync, so repaints stay bounded.
            const QCursor c = m_active->cursorFor(event->position(), vt);
            setCursor(c.shape() != Qt::ArrowCursor ? c : (m_imageSize.isEmpty() ? Qt::ArrowCursor : Qt::OpenHandCursor));
            update();
            event->accept();
            return;
        }
    }

    if (!m_panning || m_imageSize.isEmpty()) { event->ignore(); return; }

    const QPoint delta = event->pos() - m_lastMousePos;
    m_lastMousePos = event->pos();

    if (m_zoom <= 1.0f) { event->accept(); return; }

    const float W  = m_imageSize.width(), H = m_imageSize.height();
    const float Vw = width(), Vh = height();
    float fitScale    = std::min(Vw / W, Vh / H);
    float displayScale = fitScale * m_zoom;

    m_center.rx() -= delta.x() / displayScale / W;
    m_center.ry() -= delta.y() / displayScale / H;
    clampCenter();

    event->accept();
    emit viewportChanged();
}

void ViewportWidget::mouseReleaseEvent(QMouseEvent* event) {
    if (m_active) {
        const ViewportTransform vt = currentTransform();
        if (m_active->mouseRelease(event, vt)) { update(); event->accept(); return; }
    }
    const bool isPan = (event->button() == Qt::LeftButton || event->button() == Qt::MiddleButton);
    if (isPan && m_panning) {
        m_panning = false;
        setCursor(m_imageSize.isEmpty() ? Qt::ArrowCursor : Qt::OpenHandCursor);
        event->accept();
    }
}

void ViewportWidget::clampCenter() {
    if (m_imageSize.isEmpty() || width() == 0 || height() == 0) return;

    const float W  = m_imageSize.width(), H = m_imageSize.height();
    const float Vw = width(), Vh = height();
    float fitScale = std::min(Vw / W, Vh / H);

    float halfW = Vw / (2.0f * fitScale * m_zoom * W);
    float halfH = Vh / (2.0f * fitScale * m_zoom * H);

    if (halfW >= 0.5f) { m_center.setX(0.5); }
    else               { m_center.setX(std::clamp(m_center.x(), (double)halfW, 1.0 - (double)halfW)); }

    if (halfH >= 0.5f) { m_center.setY(0.5); }
    else               { m_center.setY(std::clamp(m_center.y(), (double)halfH, 1.0 - (double)halfH)); }
}
