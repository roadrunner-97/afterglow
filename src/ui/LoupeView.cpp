#include "LoupeView.h"
#include "Theme.h"

#include <QButtonGroup>
#include <QFormLayout>
#include <QFrame>
#include <QHBoxLayout>
#include <QKeyEvent>
#include <QLabel>
#include <QLocale>
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
constexpr int SIDEBAR_W = 260;

QString formatShutter(float s) {
    if (s <= 0.0f) return "—";
    if (s >= 1.0f) return QString::number(s, 'f', 1) + " s";
    // Sub-second exposures display as their reciprocal — shooters read
    // "1/250" far faster than "0.004 s".
    const int denom = static_cast<int>(std::round(1.0f / s));
    return QString("1/%1 s").arg(denom);
}

QString formatAperture(float f) {
    if (f <= 0.0f) return "—";
    return QString("f/%1").arg(QString::number(f, 'f', f < 10.0f ? 1 : 0));
}

QString formatIso(float iso) {
    if (iso <= 0.0f) return "—";
    return "ISO " + QString::number(static_cast<int>(std::round(iso)));
}

QString formatFocal(float mm) {
    if (mm <= 0.0f) return "—";
    return QString::number(mm, 'f', mm < 100.0f ? 1 : 0) + " mm";
}

// Most cameras already include the make in the model string ("Canon EOS R5"),
// so de-duplicate the make prefix to avoid "Canon Canon EOS R5".
QString formatCamera(const ImageMetadata& m) {
    if (m.cameraMake.isEmpty() && m.cameraModel.isEmpty()) return "—";
    if (!m.cameraMake.isEmpty() &&
        m.cameraModel.startsWith(m.cameraMake, Qt::CaseInsensitive))
        return m.cameraModel;
    if (m.cameraMake.isEmpty())  return m.cameraModel;
    if (m.cameraModel.isEmpty()) return m.cameraMake;
    return m.cameraMake + " " + m.cameraModel;
}

QString formatLens(const QString& s) {
    return s.isEmpty() ? QString("—") : s;
}

QString formatDateTime(const QDateTime& dt) {
    if (!dt.isValid()) return "—";
    return QLocale::system().toString(dt, QLocale::ShortFormat);
}

QString formatTempK(float k) {
    if (k <= 0.0f) return "—";
    return QString::number(static_cast<int>(std::round(k))) + " K";
}

} // namespace

LoupeView::LoupeView(QWidget* parent)
    : QWidget(parent)
{
    setFocusPolicy(Qt::StrongFocus);
    setStyleSheet(QString("LoupeView { background-color: #1e1e1e; }"));
    buildSidebar();

    // "Proofing…" overlay — shown in the top-right of the image area while
    // the background proofer generates this photo's proof.
    m_proofingLabel = new QLabel("Proofing…", this);
    m_proofingLabel->setStyleSheet(
        "QLabel { color: #e0d8c0; background: rgba(30,30,30,160);"
        "  border-radius: 4px; padding: 3px 8px; font-size: 11px; }");
    m_proofingLabel->adjustSize();
    m_proofingLabel->hide();
    m_proofingLabel->raise();
}

void LoupeView::buildSidebar()
{
    m_sidebar = new QWidget(this);
    m_sidebar->setObjectName("loupeSidebar");
    // Single stylesheet covers all sidebar children: panel chrome, the two
    // label roles ("section" headers and "key" labels), and the one body
    // text size used by metadata values.  Centralising it here keeps the
    // type/colour rhythm consistent — every label inherits one of three
    // explicit roles instead of falling back to Qt's default body font,
    // which is what made the old version look ungoverned.
    m_sidebar->setStyleSheet(QString(
        "QWidget#loupeSidebar { background-color: %1; border-left: 1px solid %2; }"
        "QLabel { color: %3; background: transparent; font-size: 12px; }"
        "QLabel[role=\"key\"] { color: %4; font-size: 10px;"
        "  text-transform: uppercase; letter-spacing: 1px; }"
        "QLabel[role=\"section\"] { color: %4; font-size: 10px;"
        "  text-transform: uppercase; letter-spacing: 1px;"
        "  padding-top: 4px; }"
        ).arg(Theme::BG_RIGHT_PANEL, Theme::BORDER,
              Theme::TEXT_PRIMARY,  Theme::TEXT_SECONDARY));

    auto* outer = new QVBoxLayout(m_sidebar);
    outer->setContentsMargins(14, 14, 14, 14);
    outer->setSpacing(10);

    // ── Mark buttons row ──────────────────────────────────────────────────
    auto* markHeader = new QLabel("Mark", m_sidebar);
    markHeader->setProperty("role", "section");
    outer->addWidget(markHeader);

    auto* btnRow = new QHBoxLayout();
    btnRow->setSpacing(4);

    // Match the rest of the app's button language (toolbar / collapse btn):
    // flat cream surface, thin border, amber-on-cream when checked, no bold.
    // The toolbar uses `BG_MAIN` for its buttons; we follow suit so the
    // sidebar doesn't introduce a third button style.
    auto makeBtn = [&](const QString& label) {
        auto* b = new QPushButton(label, m_sidebar);
        b->setCheckable(true);
        b->setStyleSheet(QString(
            "QPushButton { color: %1; background: %2; border: 1px solid %3;"
            "  border-radius: 3px; padding: 5px 4px; font-size: 12px; }"
            "QPushButton:hover  { background: %4; }"
            "QPushButton:checked { background: %5; color: %6; border-color: %5; }"
            ).arg(Theme::TEXT_PRIMARY, Theme::BG_MAIN, Theme::BORDER,
                  Theme::COLLAPSE_HOVER, Theme::CHECKED_BG, Theme::CHECKED_TEXT));
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
    auto* group = new QButtonGroup(this);
    group->setExclusive(false);
    group->addButton(m_btnAccept);
    group->addButton(m_btnRefine);
    group->addButton(m_btnDecline);

    connect(m_btnAccept,  &QPushButton::clicked, this,
            [this]() { emitMarkToggle(GridView::Mark::Accept); });
    connect(m_btnRefine,  &QPushButton::clicked, this,
            [this]() { emitMarkToggle(GridView::Mark::Refine); });
    connect(m_btnDecline, &QPushButton::clicked, this,
            [this]() { emitMarkToggle(GridView::Mark::Decline); });

    outer->addLayout(btnRow);

    // ── Camera JPEG toggle ────────────────────────────────────────────────
    m_btnCameraJpeg = new QPushButton("Camera JPEG", m_sidebar);
    m_btnCameraJpeg->setCheckable(true);
    m_btnCameraJpeg->setToolTip(
        "Compare against the camera's embedded JPEG (as-shot, no edits applied).\n"
        "Uncheck to return to the pipeline-rendered proof.");
    m_btnCameraJpeg->setStyleSheet(QString(
        "QPushButton { color: %1; background: %2; border: 1px solid %3;"
        "  border-radius: 3px; padding: 5px 8px; font-size: 12px; }"
        "QPushButton:hover   { background: %4; }"
        "QPushButton:checked { background: %5; color: %6; border-color: %5; }"
        ).arg(Theme::TEXT_PRIMARY, Theme::BG_MAIN, Theme::BORDER,
              Theme::COLLAPSE_HOVER, Theme::CHECKED_BG, Theme::CHECKED_TEXT));
    connect(m_btnCameraJpeg, &QPushButton::toggled, this,
            [this](bool checked) {
        m_userForcedCameraJpeg = checked;
        updateDisplayedImage();
    });
    outer->addWidget(m_btnCameraJpeg);

    auto* sep = new QFrame(m_sidebar);
    sep->setFrameShape(QFrame::HLine);
    sep->setStyleSheet(QString("color: %1;").arg(Theme::BORDER_PANEL));
    outer->addWidget(sep);

    // ── Metadata table (scrollable) ───────────────────────────────────────
    auto* metaHeader = new QLabel("Metadata", m_sidebar);
    metaHeader->setProperty("role", "section");
    outer->addWidget(metaHeader);

    auto* scroll = new QScrollArea(m_sidebar);
    scroll->setWidgetResizable(true);
    scroll->setFrameShape(QFrame::NoFrame);
    scroll->setStyleSheet("QScrollArea { background: transparent; }");

    auto* table = new QWidget(scroll);
    table->setStyleSheet("background: transparent;");
    auto* form = new QFormLayout(table);
    form->setContentsMargins(0, 0, 0, 0);
    form->setHorizontalSpacing(12);
    form->setVerticalSpacing(6);
    // Right-align keys so the value column reads as a clean left-aligned
    // list — same pattern as Lightroom's metadata panel.  The eye scans the
    // values, the keys recede; without this they fight for attention.
    form->setLabelAlignment(Qt::AlignRight | Qt::AlignTop);
    form->setFormAlignment(Qt::AlignLeft | Qt::AlignTop);

    auto addRow = [&](const QString& key, QLabel*& valSlot) {
        auto* k = new QLabel(key, table);
        k->setProperty("role", "key");
        valSlot = new QLabel("—", table);
        valSlot->setWordWrap(true);
        form->addRow(k, valSlot);
    };
    addRow("Camera",   m_valCamera);
    addRow("Lens",     m_valLens);
    addRow("ISO",      m_valIso);
    addRow("Shutter",  m_valShutter);
    addRow("Aperture", m_valAperture);
    addRow("Focal",    m_valFocal);
    addRow("Captured", m_valDate);
    addRow("Color Temp", m_valTempK);

    scroll->setWidget(table);
    outer->addWidget(scroll, 1);

    m_sidebar->raise();
}

void LoupeView::setProofImage(QImage proof)
{
    m_proofImage = proof;
    // Reset the manual toggle so arriving proofs auto-display, but only if
    // the user hasn't explicitly requested Camera JPEG for this photo.
    if (!m_userForcedCameraJpeg)
        updateDisplayedImage();
}

void LoupeView::setCameraJpegImage(QImage jpeg)
{
    m_cameraJpegImage = jpeg;
    // Reset per-photo toggle state when a new photo is loaded.
    m_userForcedCameraJpeg = false;
    {
        QSignalBlocker sb(m_btnCameraJpeg);
        m_btnCameraJpeg->setChecked(false);
    }
    m_proofImage = {};   // clear stale proof from previous photo
    updateDisplayedImage();
}

void LoupeView::setProofingState(bool proofing)
{
    m_proofingLabel->setVisible(proofing);
}

void LoupeView::setShowBefore(bool on)
{
    if (m_showBefore == on) return;
    m_showBefore = on;
    // Swap the displayed image without calling resetView() so the user's
    // current zoom/pan is preserved across the hold-and-release cycle.
    const bool useProof = !m_proofImage.isNull()
                          && !m_userForcedCameraJpeg
                          && !m_showBefore;
    m_image = useProof ? m_proofImage : m_cameraJpegImage;
    update();
}

void LoupeView::updateDisplayedImage()
{
    // Show the proof if available and the user hasn't forced camera view.
    const bool useProof = !m_proofImage.isNull()
                          && !m_userForcedCameraJpeg
                          && !m_showBefore;
    m_image = useProof ? m_proofImage : m_cameraJpegImage;
    if (!m_image.isNull())
        resetView();
    update();
}

void LoupeView::setMetadata(const ImageMetadata& meta)
{
    m_valCamera->setText(formatCamera(meta));
    m_valLens->setText(formatLens(meta.lens));
    m_valIso->setText(formatIso(meta.isoSpeed));
    m_valShutter->setText(formatShutter(meta.shutterSec));
    m_valAperture->setText(formatAperture(meta.aperture));
    m_valFocal->setText(formatFocal(meta.focalLenMm));
    m_valDate->setText(formatDateTime(meta.captureTime));
    m_valTempK->setText(formatTempK(meta.colorTempK));
}

void LoupeView::setCurrentMark(GridView::Mark m)
{
    m_currentMark = m;
    // QSignalBlocker on each — setChecked() would otherwise fire clicked()
    // and bounce a markChanged back through emitMarkToggle.
    QSignalBlocker ba(m_btnAccept), br(m_btnRefine), bd(m_btnDecline);
    m_btnAccept->setChecked(m == GridView::Mark::Accept);
    m_btnRefine->setChecked(m == GridView::Mark::Refine);
    m_btnDecline->setChecked(m == GridView::Mark::Decline);
}

void LoupeView::emitMarkToggle(GridView::Mark requested)
{
    const GridView::Mark next =
        (m_currentMark == requested) ? GridView::Mark::None : requested;
    setCurrentMark(next);
    emit markChanged(next);
}

void LoupeView::resetView()
{
    m_zoom = 1.0f;
    m_centre = {0.5f, 0.5f};
    update();
}

QRect LoupeView::imageRect() const
{
    const int w = std::max(0, width() - SIDEBAR_W);
    return QRect(0, 0, w, height());
}

float LoupeView::currentScale() const
{
    if (m_image.isNull()) {
        return 1.0f;
    }

    const QRect r = imageRect();
    if (r.width() <= 0 || r.height() <= 0) return 1.0f;

    const float fitScaleX = static_cast<float>(r.width())  / m_image.width();
    const float fitScaleY = static_cast<float>(r.height()) / m_image.height();
    const float fitScale  = std::min(fitScaleX, fitScaleY);

    return fitScale * m_zoom;
}

void LoupeView::clampCentre()
{
    if (m_image.isNull()) {
        return;
    }

    const QRect r = imageRect();
    const float scale = currentScale();
    const float scaledWidth  = m_image.width()  * scale;
    const float scaledHeight = m_image.height() * scale;

    // Compute the range of valid centres such that the scaled image stays
    // visible. If the scaled image is smaller than the image area, allow it
    // to be centred. Otherwise, clamp to prevent panning it entirely out.
    const float maxCentreX = (scaledWidth >= r.width())
        ? (1.0f - r.width()  / (2.0f * scale * m_image.width()))
        : 0.5f;
    const float maxCentreY = (scaledHeight >= r.height())
        ? (1.0f - r.height() / (2.0f * scale * m_image.height()))
        : 0.5f;
    const float minCentreX = 1.0f - maxCentreX;
    const float minCentreY = 1.0f - maxCentreY;

    m_centre.setX(std::clamp(m_centre.x(), static_cast<qreal>(minCentreX), static_cast<qreal>(maxCentreX)));
    m_centre.setY(std::clamp(m_centre.y(), static_cast<qreal>(minCentreY), static_cast<qreal>(maxCentreY)));
}

void LoupeView::paintEvent(QPaintEvent* /*event*/)
{
    QPainter painter(this);
    const QRect r = imageRect();
    painter.fillRect(r, palette().window());

    if (m_image.isNull()) {
        return;
    }

    painter.setClipRect(r);
    painter.setRenderHint(QPainter::SmoothPixmapTransform, true);

    const float scale = currentScale();
    const float scaledWidth  = m_image.width()  * scale;
    const float scaledHeight = m_image.height() * scale;

    // Compute the top-left corner of the scaled image in widget space,
    // given the centre point in normalised image space.
    const float centrePixelX  = m_centre.x() * m_image.width();
    const float centrePixelY  = m_centre.y() * m_image.height();
    const float centreWidgetX = r.width()  / 2.0f;
    const float centreWidgetY = r.height() / 2.0f;

    const float targetX = centreWidgetX - centrePixelX * scale;
    const float targetY = centreWidgetY - centrePixelY * scale;

    const QRectF targetRect(targetX, targetY, scaledWidth, scaledHeight);
    painter.drawImage(targetRect, m_image);
}

void LoupeView::resizeEvent(QResizeEvent* event)
{
    QWidget::resizeEvent(event);
    if (m_sidebar) {
        const int w = std::min(SIDEBAR_W, width());
        m_sidebar->setGeometry(width() - w, 0, w, height());
    }
    if (m_proofingLabel) {
        const QRect imgR = imageRect();
        const int margin = 8;
        m_proofingLabel->move(imgR.right() - m_proofingLabel->width() - margin,
                              imgR.top() + margin);
    }
    clampCentre();
    update();
}

void LoupeView::wheelEvent(QWheelEvent* event)
{
    // Wheel inside the sidebar bounds belongs to the form scroll area; let
    // it propagate naturally instead of zooming the image.
    if (!imageRect().contains(event->position().toPoint())) {
        QWidget::wheelEvent(event);
        return;
    }

    const float delta = event->angleDelta().y() / 1200.0f;
    m_zoom *= std::exp(delta);
    m_zoom = std::clamp(m_zoom, 1.0f, 16.0f);

    clampCentre();
    update();
    event->accept();
}

void LoupeView::mousePressEvent(QMouseEvent* event)
{
    if (event->button() == Qt::LeftButton && imageRect().contains(event->pos())) {
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
        m_centre.setX(m_centre.x() - delta.x() / (m_image.width()  * scale));
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
    if (event->button() == Qt::LeftButton && imageRect().contains(event->pos())) {
        emit developRequested();
        event->accept();
    }
}

void LoupeView::keyPressEvent(QKeyEvent* event)
{
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
