#pragma once
#include <QImage>
#include <QString>
#include "ImageMetadata.h"

// Loads RAW camera files (CR2, CR3, NEF, ARW, DNG, etc.) into a
// QImage::Format_RGBX64 image (16-bit per channel).
// Requires LibRaw at build time (HAVE_LIBRAW); returns a null QImage otherwise.
class RawLoader {
public:
    // Decode a RAW file into a 16-bit-per-channel QImage (Format_RGBX64).
    // If meta is non-null, fills it with the as-shot color temperature (K)
    // derived from LibRaw's cam_mul[] white-balance coefficients.
    // Returns a null QImage on failure or when LibRaw is unavailable.
    static QImage load(const QString& filePath, ImageMetadata* meta = nullptr);

    // Returns true when the file's extension is a known RAW format.
    static bool isRawFile(const QString& filePath);
};
