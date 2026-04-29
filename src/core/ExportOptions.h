#ifndef EXPORTOPTIONS_H
#define EXPORTOPTIONS_H

#include <QString>

namespace ExportOptions {

enum class Format { JPEG, PNG, TIFF };

enum class OverwritePolicy {
    Overwrite,    // Replace any existing file at the destination
    Skip,         // Leave the original; chooseDestination() returns ""
    AppendSuffix, // Insert _1, _2, ... before the extension until a free name
};

struct Options {
    QString destinationDir;

    // Pattern with brace tokens — {name}, {n}, {date}.  Anything outside a
    // recognised token (literal text, unknown tokens) passes through verbatim.
    QString filenamePattern = "{name}";

    Format format = Format::JPEG;

    // 1..100; consulted only for Format::JPEG.  Other formats ignore it.
    int jpegQuality = 90;

    OverwritePolicy onConflict = OverwritePolicy::AppendSuffix;
};

inline QString extensionFor(Format f) {
    switch (f) {
        case Format::JPEG: return QStringLiteral("jpg");
        case Format::PNG:  return QStringLiteral("png");
        case Format::TIFF: return QStringLiteral("tif");
    }
    return QStringLiteral("jpg"); // GCOVR_EXCL_LINE — unreachable; satisfies -Wreturn-type
}

inline const char* qImageFormatHint(Format f) {
    switch (f) {
        case Format::JPEG: return "JPEG";
        case Format::PNG:  return "PNG";
        case Format::TIFF: return "TIFF";
    }
    return "JPEG"; // GCOVR_EXCL_LINE — unreachable; satisfies -Wreturn-type
}

// Returns -1 ("use the format's default") for formats where the quality knob
// is meaningless.  Pass straight to QImage::save().
inline int qualityFor(const Options& opts) {
    return opts.format == Format::JPEG ? opts.jpegQuality : -1;
}

} // namespace ExportOptions

#endif // EXPORTOPTIONS_H
