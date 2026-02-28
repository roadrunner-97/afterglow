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

/**
 * @brief Image display widget with scroll-wheel zoom and left-drag pan.
 *
 * Uses QOpenGLWidget for efficient display: setImage(QImage) uploads the
 * processed image to a GL texture and repaints via a fullscreen quad shader.
 */
class ViewportWidget : public QOpenGLWidget, protected QOpenGLFunctions {
    Q_OBJECT

public:
    explicit ViewportWidget(QWidget* parent = nullptr);
    ~ViewportWidget() override;

    // Called by PhotoEditorApp when a new image is opened (before first render).
    void setImageSize(QSize size);

    // Fallback: upload a CPU QImage to the GL texture and repaint.
    void setImage(QImage image);

    // Reset to fit-zoom, centred.
    void resetView();

    // Build the ViewportRequest to hand to ImageProcessor.
    ViewportRequest viewportRequest() const;

    // The last image uploaded via setImage() (used by save).
    QImage currentImage() const { return m_lastImage; }

signals:
    void viewportChanged();

protected:
    void initializeGL() override;
    void resizeGL(int w, int h) override;
    void paintGL() override;

    void wheelEvent(QWheelEvent* event) override;
    void mousePressEvent(QMouseEvent* event) override;
    void mouseMoveEvent(QMouseEvent* event) override;
    void mouseReleaseEvent(QMouseEvent* event) override;

private:
    void createOrResizeTexture(int w, int h);
    void clampCenter();

    // GL resources
    GLuint                    m_glTexture = 0;
    QOpenGLVertexArrayObject  m_vao;
    QOpenGLBuffer             m_vbo{QOpenGLBuffer::VertexBuffer};
    QOpenGLShaderProgram*     m_shader    = nullptr;
    bool                  m_hasContent = false;

    // CPU image — kept for File→Save
    QImage m_lastImage;

    // Pan/zoom state
    QSize   m_imageSize;
    float   m_zoom   = 1.0f;
    QPointF m_center = {0.5, 0.5};
    QPoint  m_lastMousePos;
    bool    m_panning = false;
};

#endif // VIEWPORTWIDGET_H
