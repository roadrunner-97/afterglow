#ifndef LOUPEVIEW_H
#define LOUPEVIEW_H

#include <QWidget>
#include <QImage>
#include <QPointF>
#include "GridView.h"
#include "ImageMetadata.h"

class QLabel;
class QPushButton;

class LoupeView : public QWidget {
    Q_OBJECT
public:
    explicit LoupeView(QWidget *parent = nullptr);

    // Set the pipeline-rendered proof.  Displayed automatically unless the
    // user has toggled to Camera JPEG view.  Pass a null QImage to clear.
    void setProofImage(QImage proof);

    // Start displaying a different photo. Clears the previous proof and
    // per-photo toggle state; placeholder may be a Gallery thumbnail.
    void beginPhoto(QImage placeholder);

    // Replace the placeholder with the decoded camera/full preview without
    // clearing a proof that may already have arrived asynchronously.
    void setCameraJpegImage(QImage jpeg);

    // Show or hide the "Proofing…" overlay in the top-right of the image area.
    void setProofingState(bool proofing);

    // Push EXIF / camera fields into the sidebar table.  Pass an empty
    // ImageMetadata to clear all rows back to "—".
    void setMetadata(const ImageMetadata &meta);

    // Reflect the currently-stored mark in the sidebar buttons.  None
    // leaves all three buttons unchecked.
    void setCurrentMark(GridView::Mark m);

    // Reset to fit-zoom centred (also called automatically when the image changes).
    void resetView();

    // Hold-to-show "before edits" preview.  While true the displayed image
    // falls back to the camera JPEG (no pipeline edits applied) without
    // touching the user's persistent Camera-JPEG toggle.
    void setShowBefore(bool on);

    bool isShowingProof() const {
        return !m_proofImage.isNull() && m_image.cacheKey() == m_proofImage.cacheKey();
    }
    QSize displayedImageSize() const {
        return m_image.size();
    }

signals:
    // Emitted when the user presses Enter or double-clicks the image —
    // the host typically transitions to the full Develop mode in response.
    void developRequested();

    // Emitted on Left / Right arrow keys; host advances to the prev / next
    // photo in the current folder and pushes its preview back via setImage().
    void previousRequested();
    void nextRequested();

    // Emitted when the user taps A / R / D or clicks one of the sidebar
    // mark buttons.  The host already knows the current image's path and
    // injects it on the way to the catalog write.
    void markChanged(GridView::Mark m);

protected:
    void paintEvent(QPaintEvent *event) override;
    void resizeEvent(QResizeEvent *event) override;
    void wheelEvent(QWheelEvent *event) override;
    void mousePressEvent(QMouseEvent *event) override;
    void mouseMoveEvent(QMouseEvent *event) override;
    void mouseReleaseEvent(QMouseEvent *event) override;
    void mouseDoubleClickEvent(QMouseEvent *event) override;
    void keyPressEvent(QKeyEvent *event) override;

private:
    // Image-display sub-rect: full widget minus the right-side sidebar.
    QRect imageRect() const;
    // Image-space → widget-space scale at the current zoom.
    float currentScale() const;
    // Re-clamp m_centre so the image cannot be panned entirely out of view.
    void clampCentre();
    // Build the right-side metadata + mark-buttons panel.  Called from ctor.
    void buildSidebar();
    // Toggle helper shared by buttons and keyboard: pressing the active mark
    // a second time cycles back to None, matching GridView's behaviour.
    void emitMarkToggle(GridView::Mark requested);

    // Pick which image to render based on the current toggle state.
    void updateDisplayedImage();

    QImage m_proofImage;                   // pipeline-rendered proof
    QImage m_cameraJpegImage;              // camera-embedded JPEG
    QImage m_image;                        // currently displayed (proof or JPEG)
    bool   m_userForcedCameraJpeg = false; // true after user clicks toggle
    bool   m_showBefore           = false; // backslash held — show JPEG transiently

    float   m_zoom   = 1.0f;         // 1.0 = fit-to-widget; >1 zooms in
    QPointF m_centre = {0.5f, 0.5f}; // image-space normalised
    QPoint  m_lastMousePos;
    bool    m_panning = false;

    // Sidebar widgets — all live as direct children of LoupeView and are
    // positioned manually in resizeEvent().  Holding pointers here lets
    // setMetadata() swap text without rebuilding the layout.
    QWidget     *m_sidebar       = nullptr;
    QPushButton *m_btnAccept     = nullptr;
    QPushButton *m_btnRefine     = nullptr;
    QPushButton *m_btnDecline    = nullptr;
    QPushButton *m_btnCameraJpeg = nullptr;
    QLabel      *m_proofingLabel = nullptr; // "Proofing…" overlay
    QLabel      *m_valCamera     = nullptr;
    QLabel      *m_valLens       = nullptr;
    QLabel      *m_valIso        = nullptr;
    QLabel      *m_valShutter    = nullptr;
    QLabel      *m_valAperture   = nullptr;
    QLabel      *m_valFocal      = nullptr;
    QLabel      *m_valDate       = nullptr;
    QLabel      *m_valTempK      = nullptr;

    GridView::Mark m_currentMark = GridView::Mark::None;
};

#endif // LOUPEVIEW_H
