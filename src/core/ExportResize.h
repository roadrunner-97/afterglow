#ifndef EXPORTRESIZE_H
#define EXPORTRESIZE_H

#include <QSize>

class QImage;

namespace ExportResize {

enum class Mode {
    None,        // Pass-through; original size.
    LongEdge,    // Fit so max(w, h) == pixels (preserve aspect).
    ShortEdge,   // Fit so min(w, h) == pixels (preserve aspect).
    Width,       // Fit so w == pixels (preserve aspect).
    Height,      // Fit so h == pixels (preserve aspect).
    Percentage,  // Scale uniformly by `percent` / 100.
};

struct Params {
    Mode mode    = Mode::None;
    int  pixels  = 2048;   // LongEdge / ShortEdge / Width / Height
    int  percent = 100;    // Percentage
    bool dontEnlarge = true;
};

// Pure: compute the destination size for a source size under `p`.  Returns
// `src` unchanged when mode == None, when the result would round to 0, or
// when dontEnlarge is set and the requested size is larger than the source.
QSize targetSize(const QSize& src, const Params& p);

// Resizes `src` per `p` using Qt's smooth scaler.  Identity for Mode::None.
QImage apply(const QImage& src, const Params& p);

} // namespace ExportResize

#endif // EXPORTRESIZE_H
