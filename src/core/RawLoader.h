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
    // If meta is non-null, fills it with EXIF camera/exposure fields read
    // from the same LibRaw open — free of cost vs. doing it separately.
    static QImage loadThumbnail(const QString& filePath, ImageMetadata* meta = nullptr);

    // Read EXIF metadata only (no image decode).  Cheap — no unpack, no
    // demosaic.  Used by the Loupe sidebar to populate the info table when
    // we don't need to (re)decode the preview.
    static bool loadMetadata(const QString& filePath, ImageMetadata* meta);

    // Returns true when the file's extension is a known RAW format.
    static bool isRawFile(const QString& filePath);
};
