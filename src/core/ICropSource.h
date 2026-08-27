#ifndef ICROPSOURCE_H
#define ICROPSOURCE_H

#include <QRectF>
#include <QSize>
#include <QImage>
#include <QString>

// Interface implemented by the effect that owns the user's non-destructive
// crop/rotate state.  ImageProcessor queries this once per pipeline run and
// injects _userCrop* keys into every effect's params map so geometry-aware
// effects (vignette, film grain, ...) can operate relative to the cropped
// frame without any coupling to the crop plugin.
class ICropSource {
public:
    virtual ~ICropSource() = default; // GCOVR_EXCL_LINE

    // Crop rect in normalised source coordinates (0..1 on both axes).
    virtual QRectF userCropRect() const = 0;

    // Rotation in degrees (positive = counter-clockwise), applied around
    // the crop-rect centre.
    virtual float userCropAngle() const = 0;

    // Optional sink: PhotoEditorApp pushes the loaded image's size in pixels
    // here so the crop owner can constrain its rect against the actual image
    // aspect ratio.  Default no-op for implementations that don't care.
    virtual void setSourceImageSize(QSize) {} // GCOVR_EXCL_LINE

    // Applied crop/rotate operations are retained as effect state so the host
    // can rebuild its working source after undo/redo or a sidecar reload.
    // GCOVR_EXCL_START
    virtual QString committedGeometryState() const {
        return {};
    } // GCOVR_EXCL_LINE
    virtual QImage applyCommittedGeometry(const QImage &source) const {
        return source;
    } // GCOVR_EXCL_LINE
    // GCOVR_EXCL_STOP
};

#endif // ICROPSOURCE_H
