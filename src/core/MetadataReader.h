#pragma once

#include "ImageMetadata.h"
#include <QString>

// Reads metadata without decoding image pixels. RAW files are handled by
// LibRaw; JPEG/TIFF EXIF is handled by libexif when it is available.
class MetadataReader {
public:
    static bool read(const QString &filePath, ImageMetadata *metadata);
};
