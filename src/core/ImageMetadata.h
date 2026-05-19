#pragma once

#include <QDateTime>
#include <QString>
#include <cstdint>
#include <vector>

// Metadata extracted from an image file at load time.
// colorTempK == 0 means unknown / not available.
struct ImageMetadata {
    float colorTempK = 0.0f; // as-shot color temperature in Kelvin (0 = unknown)
    float tintGM     = 0.0f; // green-magenta tint, 0 = neutral (reserved for future use)

    // EXIF orientation tag of the source file (1..8). 1 = no rotation. The
    // loader applies the rotation to the QImage before returning, so the
    // pixel data is always upright; this field records what was applied.
    int orientation = 1;

    // 256-bin luminance histogram of the loaded image.  Bin index is
    // floor(L * 256) clamped to [0, 255], where L is perceptual (sRGB-encoded)
    // luminance.  Empty until computed; populated once by the app on image load
    // and is not updated when effect parameters change.
    std::vector<uint32_t> luminanceHistogram;

    // EXIF camera/exposure fields populated by RawLoader.  Strings empty and
    // numerics 0 when unknown — the Loupe sidebar treats those as "—" rows.
    QString   cameraMake;
    QString   cameraModel;
    QString   lens;
    float     isoSpeed   = 0.0f;
    float     shutterSec = 0.0f; // exposure time in seconds (0.004 = 1/250)
    float     aperture   = 0.0f; // f-number
    float     focalLenMm = 0.0f;
    QDateTime captureTime; // null when the file lacks a timestamp
};
