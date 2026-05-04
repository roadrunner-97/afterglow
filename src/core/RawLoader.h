#pragma once
#include <QImage>
#include <QString>
#include "ImageMetadata.h"

// Loads RAW camera files (CR2, CR3, NEF, ARW, DNG, etc.) into a
// QImage::Format_RGBX64 image (16-bit per channel) via LibRaw.
class RawLoader {
public:
    // Decode a RAW file into a 16-bit-per-channel QImage (Format_RGBX64).
    // If meta is non-null, fills it with the as-shot color temperature (K)
    // derived from LibRaw's cam_mul[] white-balance coefficients.
    // Returns a null QImage on failure.
    static QImage load(const QString& filePath, ImageMetadata* meta = nullptr);

    // Decode the camera-embedded preview (typically a full-res JPEG) without
    // demosaicing the sensor data. Much faster than load() and intended for
    // grid thumbnails / loupe previews. Returns a null QImage on failure.
    static QImage loadThumbnail(const QString& filePath);

    // Returns true when the file's extension is a known RAW format.
    static bool isRawFile(const QString& filePath);
};
