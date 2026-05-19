#include "ExportResize.h"

#include <QImage>
#include <algorithm>

namespace ExportResize {

QSize targetSize(const QSize &src, const Params &p) {
    if (p.mode == Mode::None || src.isEmpty()) return src;

    const double sw    = src.width();
    const double sh    = src.height();
    double       scale = 1.0;

    switch (p.mode) {
    case Mode::None: // GCOVR_EXCL_LINE — unreachable: early return at line 9
        return src;  // GCOVR_EXCL_LINE
    case Mode::LongEdge:
        scale = double(std::max(1, p.pixels)) / std::max(sw, sh);
        break;
    case Mode::ShortEdge:
        scale = double(std::max(1, p.pixels)) / std::min(sw, sh);
        break;
    case Mode::Width:
        scale = double(std::max(1, p.pixels)) / sw;
        break;
    case Mode::Height:
        scale = double(std::max(1, p.pixels)) / sh;
        break;
    case Mode::Percentage:
        scale = std::max(1, p.percent) / 100.0;
        break;
    }

    if (p.dontEnlarge && scale >= 1.0) return src;

    const int w = std::max(1, int(std::lround(sw * scale)));
    const int h = std::max(1, int(std::lround(sh * scale)));
    return QSize(w, h);
}

QImage apply(const QImage &src, const Params &p) {
    const QSize tgt = targetSize(src.size(), p);
    if (tgt == src.size()) return src;
    return src.scaled(tgt, Qt::IgnoreAspectRatio, Qt::SmoothTransformation);
}

} // namespace ExportResize
