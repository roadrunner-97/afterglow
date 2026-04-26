#ifndef ICROPSOURCE_H
#define ICROPSOURCE_H

#include <QRectF>

// Interface implemented by the effect that owns the user's non-destructive
// crop/rotate state.  ImageProcessor queries this once per pipeline run and
// injects _userCrop* keys into every effect's params map so geometry-aware
// effects (vignette, film grain, ...) can operate relative to the cropped
// frame without any coupling to the crop plugin.
class ICropSource {
public:
    virtual ~ICropSource() = default;

    // Crop rect in normalised source coordinates (0..1 on both axes).
    virtual QRectF userCropRect() const = 0;

    // Rotation in degrees (positive = counter-clockwise), applied around
    // the crop-rect centre.
    virtual float userCropAngle() const = 0;
};

#endif // ICROPSOURCE_H
