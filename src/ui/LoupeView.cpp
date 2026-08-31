#include "LoupeView.h"

#include <QButtonGroup>
#include <QFrame>
#include <QHBoxLayout>
#include <QKeyEvent>
#include <QLabel>
#include <QMouseEvent>
#include <QPainter>
#include <QPushButton>
#include <QResizeEvent>
#include <QScrollArea>
#include <QVBoxLayout>
#include <QWheelEvent>
#include <algorithm>
#include <cmath>

namespace {

// Fixed-width sidebar carved off the right edge of the widget.  Wide enough
// for a two-column "Camera | Sony ILCE-7M4" row at the body font without
// wrapping, narrow enough to leave the image area dominant on a 1280px
// laptop screen.
constexpr int SIDEBAR_W = 360;

} // namespace

LoupeView::LoupeView(QWidget *parent) : QWidget(parent) {
    setFocusPolicy(Qt::StrongFocus);
    buildSidebar();

    // "Proofing…" overlay — shown in the top-right of the image area while
    // the background proofer generates this photo's proof.
    m_proofingLabel = new QLabel("Proofing…", this);
    m_proofingLabel->adjustSize();
    m_proofingLabel->hide();
    m_proofingLabel->raise();
}

void LoupeView::buildSidebar() {
    m_sidebar = new QWidget(this);
    m_sidebar->setObjectName("loupeSidebar");
    m_sidebar->setStyleSheet("QLabel { font-size: 14px; }"
                             "QLabel[role=\"key\"]     { font-size: 13px; letter-spacing: 1px; }"
                             "QLabel[role=\"section\"] { font-size: 13px; letter-spacing: 1px; padding-top: 4px; }");

    auto *outer = new QVBoxLayout(m_sidebar);
    outer->setContentsMargins(14, 14, 14, 14);
    outer->setSpacing(10);

    // ── Mark buttons row ──────────────────────────────────────────────────
    auto *markHeader = new QLabel("Mark", m_sidebar);
    markHeader->setProperty("role", "section");
    outer->addWidget(markHeader);

    auto *btnRow = new QHBoxLayout();
    btnRow->setSpacing(4);

    auto makeBtn = [&](const QString &label) {
        auto *b = new QPushButton(label, m_sidebar);
        b->setCheckable(true);
        btnRow->addWidget(b, 1);
        return b;
    };
    m_btnAccept  = makeBtn("Accept");
    m_btnRefine  = makeBtn("Refine");
    m_btnDecline = makeBtn("Decline");

    // QButtonGroup gives us radio-style exclusion across the three buttons.
    // Auto-exclusivity is off so that clicking the already-checked button
    // can run our toggle handler (otherwise QButtonGroup blocks the second
    // click and we'd never see "active mark pressed → clear to None").
    auto *group = new QButtonGroup(this);
    group->setExclusive(false);
    group->addButton(m_btnAccept);
    group->addButton(m_btnRefine);
    group->addButton(m_btnDecline);

    connect(m_btnAccept, &QPushButton::clicked, this, [this]() { emitMarkToggle(GridView::Mark::Accept); });
    connect(m_btnRefine, &QPushButton::clicked, this, [this]() { emitMarkToggle(GridView::Mark::Refine); });
    connect(m_btnDecline, &QPushButton::clicked, this, [this]() { emitMarkToggle(GridView::Mark::Decline); });

    outer->addLayout(btnRow);

    // ── Image version selector ────────────────────────────────────────────
    auto *versionHeader = new QLabel("Image", m_sidebar);
    versionHeader->setProperty("role", "section");
    outer->addWidget(versionHeader);

    auto *versionRow = new QHBoxLayout();
    versionRow->setSpacing(4);
    auto makeVersionButton = [&](const QString &label) {
        auto *button = new QPushButton(label, m_sidebar);
        button->setCheckable(true);
        versionRow->addWidget(button, 1);
        return button;
    };
    m_btnCameraJpeg  = makeVersionButton("Camera JPEG");
    m_btnOriginalRaw = makeVersionButton("Original RAW");
    m_btnEditedRaw   = makeVersionButton("Edited RAW");
    m_btnOriginalRaw->setEnabled(false);
    m_btnEditedRaw->setChecked(true);

    auto *versionGroup = new QButtonGroup(this);
    versionGroup->setExclusive(true);
    versionGroup->addButton(m_btnCameraJpeg);
    versionGroup->addButton(m_btnOriginalRaw);
    versionGroup->addButton(m_btnEditedRaw);

    connect(m_btnCameraJpeg, &QPushButton::clicked, this, [this]() {
        m_selectedVersion = ImageVersion::CameraJpeg;
        updateDisplayedImage();
    });
    connect(m_btnOriginalRaw, &QPushButton::clicked, this, [this]() {
        m_selectedVersion = ImageVersion::OriginalRaw;
        updateDisplayedImage();
    });
    connect(m_btnEditedRaw, &QPushButton::clicked, this, [this]() {
        m_selectedVersion = ImageVersion::EditedRaw;
        updateDisplayedImage();
    });
    outer->addLayout(versionRow);

    auto *sep = new QFrame(m_sidebar);
    sep->setFrameShape(QFrame::HLine);
    outer->addWidget(sep);

    // ── Metadata (shared with Gallery and Develop) ────────────────────────
    auto *scroll = new QScrollArea(m_sidebar);
    scroll->setWidgetResizable(true);
    scroll->setFrameShape(QFrame::NoFrame);
    m_metadataTray = new MetadataTray(scroll);
    scroll->setWidget(m_metadataTray);
    outer->addWidget(scroll, 1);

    m_sidebar->raise();
}

void LoupeView::setProofImage(QImage proof) {
    m_proofImage = proof;
    if (m_selectedVersion == ImageVersion::EditedRaw) updateDisplayedImage();
}

void LoupeView::beginPhoto(QImage placeholder) {
    m_cameraJpegImage  = std::move(placeholder);
    m_originalRawImage = {};
    m_selectedVersion  = ImageVersion::EditedRaw;
    {
        QSignalBlocker cameraBlock(m_btnCameraJpeg), originalBlock(m_btnOriginalRaw), editedBlock(m_btnEditedRaw);
        m_btnCameraJpeg->setChecked(false);
        m_btnOriginalRaw->setChecked(false);
        m_btnOriginalRaw->setEnabled(false);
        m_btnEditedRaw->setChecked(true);
    }
    m_proofImage = {}; // clear stale proof from previous photo
    updateDisplayedImage();
}

void LoupeView::setCameraJpegImage(QImage jpeg) {
    m_cameraJpegImage = std::move(jpeg);
    if (m_selectedVersion == ImageVersion::CameraJpeg || m_proofImage.isNull() || m_showBefore) updateDisplayedImage();
}

void LoupeView::setOriginalRawImage(QImage raw) {
    m_originalRawImage = std::move(raw);
    m_btnOriginalRaw->setEnabled(!m_originalRawImage.isNull());
    if (m_selectedVersion == ImageVersion::OriginalRaw) updateDisplayedImage();
}

void LoupeView::setProofingState(bool proofing) {
    m_proofingLabel->setVisible(proofing);
}

void LoupeView::setShowBefore(bool on) {
    if (m_showBefore == on) return;
    m_showBefore = on;
    // Swap the displayed image without calling resetView() so the user's
    // current zoom/pan is preserved across the hold-and-release cycle.
    if (m_showBefore) {
        m_image = m_cameraJpegImage;
    } else if (m_selectedVersion == ImageVersion::OriginalRaw && !m_originalRawImage.isNull()) {
        m_image = m_originalRawImage;
    } else if (m_selectedVersion == ImageVersion::EditedRaw && !m_proofImage.isNull()) {
        m_image = m_proofImage;
    } else {
        m_image = m_cameraJpegImage;
    }
    update();
}

void LoupeView::updateDisplayedImage() {
    if (m_showBefore || m_selectedVersion == ImageVersion::CameraJpeg) m_image = m_cameraJpegImage;
    else if (m_selectedVersion == ImageVersion::OriginalRaw && !m_originalRawImage.isNull())
        m_image = m_originalRawImage;
    else if (m_selectedVersion == ImageVersion::EditedRaw && !m_proofImage.isNull()) m_image = m_proofImage;
    else m_image = m_cameraJpegImage;
    if (!m_image.isNull()) resetView();
    update();
}

void LoupeView::setMetadata(const MetadataTray::Info &info) {
    m_metadataTray->setInfo(info);
}

void LoupeView::setCurrentMark(GridView::Mark m) {
    m_currentMark = m;
    // QSignalBlocker on each — setChecked() would otherwise fire clicked()
    // and bounce a markChanged back through emitMarkToggle.
    QSignalBlocker ba(m_btnAccept), br(m_btnRefine), bd(m_btnDecline);
    m_btnAccept->setChecked(m == GridView::Mark::Accept);
    m_btnRefine->setChecked(m == GridView::Mark::Refine);
    m_btnDecline->setChecked(m == GridView::Mark::Decline);
}

void LoupeView::emitMarkToggle(GridView::Mark requested) {
    const GridView::Mark next = (m_currentMark == requested) ? GridView::Mark::None : requested;
    setCurrentMark(next);
    emit markChanged(next);
}

void LoupeView::resetView() {
    m_zoom   = 1.0f;
    m_centre = {0.5f, 0.5f};
    update();
}

QRect LoupeView::imageRect() const {
    const int w = std::max(0, width() - SIDEBAR_W);
    return QRect(0, 0, w, height());
}

float LoupeView::currentScale() const {
    if (m_image.isNull()) {
        return 1.0f;
    }

    const QRect r = imageRect();
    if (r.width() <= 0 || r.height() <= 0) return 1.0f;

    const float fitScaleX = static_cast<float>(r.width()) / static_cast<float>(m_image.width());
    const float fitScaleY = static_cast<float>(r.height()) / static_cast<float>(m_image.height());
    const float fitScale  = std::min(fitScaleX, fitScaleY);

    return fitScale * m_zoom;
}

void LoupeView::clampCentre() {
    if (m_image.isNull()) {
        return;
    }

    const QRect r            = imageRect();
    const float scale        = currentScale();
    const float scaledWidth  = static_cast<float>(m_image.width()) * scale;
    const float scaledHeight = static_cast<float>(m_image.height()) * scale;

    // Compute the range of valid centres such that the scaled image stays
    // visible. If the scaled image is smaller than the image area, allow it
    // to be centred. Otherwise, clamp to prevent panning it entirely out.
    const float maxCentreX =
        (scaledWidth >= static_cast<float>(r.width()))
            ? (1.0f - static_cast<float>(r.width()) / (2.0f * scale * static_cast<float>(m_image.width())))
            : 0.5f;
    const float maxCentreY =
        (scaledHeight >= static_cast<float>(r.height()))
            ? (1.0f - static_cast<float>(r.height()) / (2.0f * scale * static_cast<float>(m_image.height())))
            : 0.5f;
    const float minCentreX = 1.0f - maxCentreX;
    const float minCentreY = 1.0f - maxCentreY;

    m_centre.setX(std::clamp(m_centre.x(), static_cast<qreal>(minCentreX), static_cast<qreal>(maxCentreX)));
    m_centre.setY(std::clamp(m_centre.y(), static_cast<qreal>(minCentreY), static_cast<qreal>(maxCentreY)));
}

void LoupeView::paintEvent(QPaintEvent * /*event*/) {
    QPainter    painter(this);
    const QRect r = imageRect();
    painter.fillRect(r, palette().window());

    if (m_image.isNull()) {
        return;
    }

    painter.setClipRect(r);
    painter.setRenderHint(QPainter::SmoothPixmapTransform, true);

    const float scale        = currentScale();
    const float scaledWidth  = static_cast<float>(m_image.width()) * scale;
    const float scaledHeight = static_cast<float>(m_image.height()) * scale;

    // Compute the top-left corner of the scaled image in widget space,
    // given the centre point in normalised image space.
    const float centrePixelX  = static_cast<float>(m_centre.x()) * static_cast<float>(m_image.width());
    const float centrePixelY  = static_cast<float>(m_centre.y()) * static_cast<float>(m_image.height());
    const float centreWidgetX = static_cast<float>(r.width()) / 2.0f;
    const float centreWidgetY = static_cast<float>(r.height()) / 2.0f;

    const float targetX = centreWidgetX - centrePixelX * scale;
    const float targetY = centreWidgetY - centrePixelY * scale;

    const QRectF targetRect(targetX, targetY, scaledWidth, scaledHeight);
    painter.drawImage(targetRect, m_image);
}

void LoupeView::resizeEvent(QResizeEvent *event) {
    QWidget::resizeEvent(event);
    if (m_sidebar) {
        const int w = std::min(SIDEBAR_W, width());
        m_sidebar->setGeometry(width() - w, 0, w, height());
    }
    if (m_proofingLabel) {
        const QRect imgR   = imageRect();
        const int   margin = 8;
        m_proofingLabel->move(imgR.right() - m_proofingLabel->width() - margin, imgR.top() + margin);
    }
    clampCentre();
    update();
}

void LoupeView::wheelEvent(QWheelEvent *event) {
    // Wheel inside the sidebar bounds belongs to the form scroll area; let
    // it propagate naturally instead of zooming the image.
    if (!imageRect().contains(event->position().toPoint())) {
        QWidget::wheelEvent(event);
        return;
    }

    const float delta  = static_cast<float>(event->angleDelta().y()) / 1200.0f;
    m_zoom            *= std::exp(delta);
    m_zoom             = std::clamp(m_zoom, 1.0f, 16.0f);

    clampCentre();
    update();
    event->accept();
}

void LoupeView::mousePressEvent(QMouseEvent *event) {
    if (event->button() == Qt::LeftButton && imageRect().contains(event->pos())) {
        m_panning      = true;
        m_lastMousePos = event->pos();
        event->accept();
    }
}

void LoupeView::mouseMoveEvent(QMouseEvent *event) {
    if (m_panning && !m_image.isNull()) {
        const QPoint delta = event->pos() - m_lastMousePos;
        const float  scale = currentScale();

        // Delta in widget pixels maps to delta in normalised image space.
        m_centre.setX(m_centre.x() - static_cast<float>(delta.x()) / (static_cast<float>(m_image.width()) * scale));
        m_centre.setY(m_centre.y() - static_cast<float>(delta.y()) / (static_cast<float>(m_image.height()) * scale));

        clampCentre();
        m_lastMousePos = event->pos();
        update();
        event->accept();
    }
}

void LoupeView::mouseReleaseEvent(QMouseEvent *event) {
    if (event->button() == Qt::LeftButton) {
        m_panning = false;
        event->accept();
    }
}

void LoupeView::mouseDoubleClickEvent(QMouseEvent *event) {
    if (event->button() == Qt::LeftButton && imageRect().contains(event->pos())) {
        emit developRequested();
        event->accept();
    }
}

void LoupeView::keyPressEvent(QKeyEvent *event) {
    switch (event->key()) {
    case Qt::Key_Return:
    case Qt::Key_Enter:
        emit developRequested();
        event->accept();
        return;
    case Qt::Key_F:
        resetView();
        event->accept();
        return;
    case Qt::Key_Left:
        emit previousRequested();
        event->accept();
        return;
    case Qt::Key_Right:
        emit nextRequested();
        event->accept();
        return;
    case Qt::Key_A:
        emitMarkToggle(GridView::Mark::Accept);
        event->accept();
        return;
    case Qt::Key_R:
        emitMarkToggle(GridView::Mark::Refine);
        event->accept();
        return;
    case Qt::Key_D:
        emitMarkToggle(GridView::Mark::Decline);
        event->accept();
        return;
    default:
        QWidget::keyPressEvent(event);
        return;
    }
}
