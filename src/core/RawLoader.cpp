#include "RawLoader.h"
#include <QFileInfo>
#include <QStringList>

#ifdef HAVE_LIBRAW
#include <libraw/libraw.h>
#endif

bool RawLoader::isRawFile(const QString& filePath) {
    static const QStringList rawExts = {
        "cr2", "cr3",               // Canon
        "nef", "nrw",               // Nikon
        "arw", "sr2", "srf",        // Sony
        "dng",                      // Adobe / universal
        "raf",                      // Fujifilm
        "orf",                      // Olympus
        "rw2",                      // Panasonic
        "pef",                      // Pentax
        "srw",                      // Samsung
        "x3f",                      // Sigma / Foveon
        "rwl",                      // Leica
        "mrw",                      // Minolta
        "3fr",                      // Hasselblad
        "kdc", "dcr",               // Kodak
        "erf",                      // Epson
    };
    return rawExts.contains(QFileInfo(filePath).suffix().toLower());
}

QImage RawLoader::load(const QString& filePath) {
#ifdef HAVE_LIBRAW
    LibRaw rawProcessor;

    // 16-bit sRGB output; use the camera's own white balance; no auto-brightness
    rawProcessor.imgdata.params.output_bps     = 16;
    rawProcessor.imgdata.params.output_color   = 1;  // sRGB
    rawProcessor.imgdata.params.use_camera_wb  = 1;
    rawProcessor.imgdata.params.no_auto_bright = 1;
    rawProcessor.imgdata.params.fbdd_noiserd   = 1;  // basic noise reduction

    if (rawProcessor.open_file(filePath.toLocal8Bit().data()) != LIBRAW_SUCCESS)
        return {};
    if (rawProcessor.unpack() != LIBRAW_SUCCESS)
        return {};
    if (rawProcessor.dcraw_process() != LIBRAW_SUCCESS)
        return {};

    int errorCode = 0;
    libraw_processed_image_t* img = rawProcessor.dcraw_make_mem_image(&errorCode);
    if (!img || errorCode != LIBRAW_SUCCESS)
        return {};

    // Sanity check: must be a bitmap with 3 colour channels at 16 bpc
    if (img->type != LIBRAW_IMAGE_BITMAP || img->bits != 16 || img->colors != 3) {
        LibRaw::dcraw_clear_mem(img);
        return {};
    }

    // img->data is uint16_t[height * width * 3], R G B interleaved
    const uint16_t* src = reinterpret_cast<const uint16_t*>(img->data);
    QImage result(img->width, img->height, QImage::Format_RGBX64);

    for (int y = 0; y < static_cast<int>(img->height); ++y) {
        QRgba64* row = reinterpret_cast<QRgba64*>(result.scanLine(y));
        for (int x = 0; x < static_cast<int>(img->width); ++x) {
            const uint16_t* px = src + 3 * (y * img->width + x);
            row[x] = QRgba64::fromRgba64(px[0], px[1], px[2], 65535);
        }
    }

    LibRaw::dcraw_clear_mem(img);
    return result;
#else
    Q_UNUSED(filePath)
    return {};
#endif
}
