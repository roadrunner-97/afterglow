#ifndef VIEWPORTWIDGET_H
#define VIEWPORTWIDGET_H

#include <QOpenGLWidget>
#include <QOpenGLFunctions>
#include <QOpenGLBuffer>
#include <QOpenGLVertexArrayObject>
#include <QImage>
#include <QPointF>
#include "GpuPipeline.h"

class QOpenGLShaderProgram;
class QKeyEvent;
class IInteractiveEffect;
struct ViewportTransform;

/**
 * @brief Image display widget with scroll-wheel zoom and left-drag pan.
 *
 * Uses QOpenGLWidget for efficient display: setImage(QImage) uploads the
 * processed image to a GL texture and repaints via a fullscreen quad shader.
 */
class ViewportWidget : public QOpenGLWidget, protected QOpenGLFunctions {
    Q_OBJECT

public:
    explicit ViewportWidget(QWidget *parent = nullptr);
    ~ViewportWidget() override;

    // Called by PhotoEditorApp when a new image is opened (before first render).
    void setImageSize(QSize size);

    // Upload a CPU QImage to the GL texture and repaint.  `offset` places
    // the image at that top-left position within the widget; the surrounding
    // viewport is the GL clear colour (i.e. proper letterbox).
    void setImage(QImage image, QPoint offset = {});

    // Reset to fit-zoom, centred.
    void resetView();

    // Build the ViewportRequest to hand to ImageProcessor.
    ViewportRequest viewportRequest() const;

    // Set the currently active interactive effect (nullptr to clear).
    // Triggers an immediate repaint so the overlay appears/disappears.
    void setActiveInteractiveEffect(IInteractiveEffect *effect);
    // Activates a tool while retaining the current tool as a passive overlay.
    // Used by local masks so crop geometry remains visible but does not
    // compete for pointer input.
    void setActiveInteractiveEffectWithBackground(IInteractiveEffect *effect);
    void clearActiveInteractiveEffect(IInteractiveEffect *effect);

    // Rotation applied to the displayed image (and the overlay, which uses
    // the same pivot so handles stay anchored to the image content).  Pivot
    // is in normalised source coords (0..1).  Angle is CCW-positive in
    // screen-space degrees.
    void setImageRotation(float angleDeg, QPointF pivotNorm);

signals:
    void viewportChanged();

protected:
    void initializeGL() override;
    void resizeGL(int w, int h) override;
    void paintGL() override;

    void keyPressEvent(QKeyEvent *event) override;
    void wheelEvent(QWheelEvent *event) override;
    void mousePressEvent(QMouseEvent *event) override;
    void mouseMoveEvent(QMouseEvent *event) override;
    void mouseReleaseEvent(QMouseEvent *event) override;

private:
    void              createOrResizeTexture(int w, int h);
    void              clampCenter();
    ViewportTransform currentTransform() const;

    // GL resources
    GLuint                   m_glTexture = 0;
    QSize                    m_textureSize; // current texture allocation
    QOpenGLVertexArrayObject m_vao;
    QOpenGLBuffer            m_vbo{QOpenGLBuffer::VertexBuffer};
    QOpenGLShaderProgram    *m_shader     = nullptr;
    bool                     m_hasContent = false;

    // Position of the image within the widget (top-left, widget pixels).
    // Drives the quad-rect uniform so the image lands at the correct sub-rect
    // and the GL clear shows through as letterbox.
    QPoint m_imageOffset;
    QSize  m_renderedSize;

    // setImage() can be called before the widget has ever been shown — for
    // example when the user goes Loupe → Develop and the develop page's GL
    // context hasn't been created yet.  Stash the image here and apply it
    // from initializeGL() once the context is up.
    QImage m_pendingImage;
    QPoint m_pendingOffset;

    // Pan/zoom state
    QSize   m_imageSize;
    float   m_zoom   = 1.0f;
    QPointF m_center = {0.5, 0.5};
    QPoint  m_lastMousePos;
    bool    m_panning = false;

    // Optional overlay / event consumer
    IInteractiveEffect *m_active            = nullptr;
    IInteractiveEffect *m_backgroundOverlay = nullptr;

    // Image rotation (degrees, CCW-positive on screen) applied in the GL
    // quad's vertex shader.  Pivot is normalised source coords.
    float   m_imgAngleDeg = 0.0f;
    QPointF m_imgPivotNorm{0.5, 0.5};
};

#endif // VIEWPORTWIDGET_H
