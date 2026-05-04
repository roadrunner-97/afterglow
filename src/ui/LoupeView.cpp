#include "LoupeView.h"

#include <QPainter>
#include <QWheelEvent>
#include <QMouseEvent>
#include <QKeyEvent>
#include <cmath>

LoupeView::LoupeView(QWidget* parent)
    : QWidget(parent)
{
    setFocusPolicy(Qt::StrongFocus);
    setStyleSheet("background-color: #1e1e1e;");
}

void LoupeView::setImage(QImage image)
{
    m_image = image;
    resetView();
    update();
}

void LoupeView::resetView()
{
    m_zoom = 1.0f;
    m_centre = {0.5f, 0.5f};
    update();
}

float LoupeView::currentScale() const
{
    if (m_image.isNull()) {
        return 1.0f;
    }

    const float fitScaleX = static_cast<float>(width()) / m_image.width();
    const float fitScaleY = static_cast<float>(height()) / m_image.height();
    const float fitScale = std::min(fitScaleX, fitScaleY);

    return fitScale * m_zoom;
}

void LoupeView::clampCentre()
{
    if (m_image.isNull()) {
        return;
    }

    const float scale = currentScale();
    const float scaledWidth = m_image.width() * scale;
    const float scaledHeight = m_image.height() * scale;

    // Compute the range of valid centres such that the scaled image stays
    // visible. If the scaled image is smaller than the widget, allow it to be
    // centred. Otherwise, clamp to prevent panning it entirely out of view.
    const float maxCentreX = (scaledWidth >= width())
        ? (1.0f - width() / (2.0f * scale * m_image.width()))
        : 0.5f;
    const float maxCentreY = (scaledHeight >= height())
        ? (1.0f - height() / (2.0f * scale * m_image.height()))
        : 0.5f;
    const float minCentreX = 1.0f - maxCentreX;
    const float minCentreY = 1.0f - maxCentreY;

    m_centre.setX(std::clamp(m_centre.x(), static_cast<qreal>(minCentreX), static_cast<qreal>(maxCentreX)));
    m_centre.setY(std::clamp(m_centre.y(), static_cast<qreal>(minCentreY), static_cast<qreal>(maxCentreY)));
}

void LoupeView::paintEvent(QPaintEvent* /*event*/)
{
    QPainter painter(this);
    painter.fillRect(rect(), palette().window());

    if (m_image.isNull()) {
        return;
    }

    painter.setRenderHint(QPainter::SmoothPixmapTransform, true);

    const float scale = currentScale();
    const float scaledWidth = m_image.width() * scale;
    const float scaledHeight = m_image.height() * scale;

    // Compute the top-left corner of the scaled image in widget space,
    // given the centre point in normalised image space.
    const float centrePixelX = m_centre.x() * m_image.width();
    const float centrePixelY = m_centre.y() * m_image.height();
    const float centreWidgetX = width() / 2.0f;
    const float centreWidgetY = height() / 2.0f;

    const float targetX = centreWidgetX - centrePixelX * scale;
    const float targetY = centreWidgetY - centrePixelY * scale;

    const QRectF targetRect(targetX, targetY, scaledWidth, scaledHeight);
    painter.drawImage(targetRect, m_image);
}

void LoupeView::resizeEvent(QResizeEvent* event)
{
    QWidget::resizeEvent(event);
    clampCentre();
    update();
}

void LoupeView::wheelEvent(QWheelEvent* event)
{
    const float delta = event->angleDelta().y() / 1200.0f;
    m_zoom *= std::exp(delta);
    m_zoom = std::clamp(m_zoom, 1.0f, 16.0f);

    clampCentre();
    update();
    event->accept();
}

void LoupeView::mousePressEvent(QMouseEvent* event)
{
    if (event->button() == Qt::LeftButton) {
        m_panning = true;
        m_lastMousePos = event->pos();
        event->accept();
    }
}

void LoupeView::mouseMoveEvent(QMouseEvent* event)
{
    if (m_panning && !m_image.isNull()) {
        const QPoint delta = event->pos() - m_lastMousePos;
        const float scale = currentScale();

        // Delta in widget pixels maps to delta in normalised image space.
        m_centre.setX(m_centre.x() - delta.x() / (m_image.width() * scale));
        m_centre.setY(m_centre.y() - delta.y() / (m_image.height() * scale));

        clampCentre();
        m_lastMousePos = event->pos();
        update();
        event->accept();
    }
}

void LoupeView::mouseReleaseEvent(QMouseEvent* event)
{
    if (event->button() == Qt::LeftButton) {
        m_panning = false;
        event->accept();
    }
}

void LoupeView::mouseDoubleClickEvent(QMouseEvent* event)
{
    if (event->button() == Qt::LeftButton) {
        emit developRequested();
        event->accept();
    }
}

void LoupeView::keyPressEvent(QKeyEvent* event)
{
    if (event->key() == Qt::Key_Return || event->key() == Qt::Key_Enter) {
        emit developRequested();
        event->accept();
    } else if (event->key() == Qt::Key_R || event->key() == Qt::Key_F) {
        resetView();
        event->accept();
    } else if (event->key() == Qt::Key_Left) {
        emit previousRequested();
        event->accept();
    } else if (event->key() == Qt::Key_Right) {
        emit nextRequested();
        event->accept();
    } else {
        QWidget::keyPressEvent(event);
    }
}
