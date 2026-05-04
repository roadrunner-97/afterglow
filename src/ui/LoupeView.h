#ifndef LOUPEVIEW_H
#define LOUPEVIEW_H

#include <QWidget>
#include <QImage>
#include <QPointF>

class LoupeView : public QWidget {
    Q_OBJECT
public:
    explicit LoupeView(QWidget* parent = nullptr);

    // Replace the displayed image. Resets to fit-zoom centred.
    void setImage(QImage image);

    // Reset to fit-zoom centred (also called automatically by setImage).
    void resetView();

signals:
    // Emitted when the user presses Enter or double-clicks the image —
    // the host typically transitions to the full Develop mode in response.
    void developRequested();

protected:
    void paintEvent(QPaintEvent* event) override;
    void resizeEvent(QResizeEvent* event) override;
    void wheelEvent(QWheelEvent* event) override;
    void mousePressEvent(QMouseEvent* event) override;
    void mouseMoveEvent(QMouseEvent* event) override;
    void mouseReleaseEvent(QMouseEvent* event) override;
    void mouseDoubleClickEvent(QMouseEvent* event) override;
    void keyPressEvent(QKeyEvent* event) override;

private:
    // Image-space → widget-space scale at the current zoom.
    float currentScale() const;
    // Re-clamp m_centre so the image cannot be panned entirely out of view.
    void clampCentre();

    QImage  m_image;          // null until setImage()
    float   m_zoom    = 1.0f; // 1.0 = fit-to-widget; >1 zooms in
    QPointF m_centre  = {0.5f, 0.5f}; // image-space normalised
    QPoint  m_lastMousePos;
    bool    m_panning = false;
};

#endif // LOUPEVIEW_H
