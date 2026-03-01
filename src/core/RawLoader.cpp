#include "RawLoader.h"
#include <QFileInfo>
#include <QStringList>

#ifdef HAVE_LIBRAW
#include <libraw/libraw.h>
#include <cmath>
#include <algorithm>

// ---------------------------------------------------------------------------
// Color temperature estimation from LibRaw cam_mul[]
//
// cam_mul[4] = {R, G1, B, G2} as-shot WB multipliers.  Each entry is
// proportional to 1 / illuminant_channel, so:
//   cam_mul[0]/cam_mul[1]  =  G_illum / R_illum
//   cam_mul[2]/cam_mul[1]  =  G_illum / B_illum
//
// The Planckian locus (Kang et al. 2002) maps temperature T → xy chromaticity
// → linear sRGB.  We want K such that R(K)/B(K) = cam_mul[2]/cam_mul[0]
// (derived from the ratio above), and binary-search for it.
// ---------------------------------------------------------------------------

// Kang et al. (2002) Planckian locus → linear sRGB (D65 reference).
static void kangToRGB(float T, float& r, float& g, float& b) {
    T = std::max(1000.0f, std::min(15000.0f, T));

    float x, y;
    if (T <= 4000.0f) {
        x = -0.2661239e9f/(T*T*T) - 0.2343580e6f/(T*T) + 0.8776956e3f/T + 0.179910f;
        y = (T <= 2222.0f)
            ? (-1.1063814f*(x*x*x) - 1.34811020f*(x*x) + 2.18555832f*x - 0.20219683f)
            : (-0.9549476f*(x*x*x) - 1.37418593f*(x*x) + 2.09137015f*x - 0.16748867f);
    } else {
        x = -3.0258469e9f/(T*T*T) + 2.1070379e6f/(T*T) + 0.2226347e3f/T + 0.240390f;
        y = 3.0817580f*(x*x*x) - 5.8733867f*(x*x) + 3.75112997f*x - 0.37001483f;
    }

    float X = x / y;
    float Z = (1.0f - x - y) / y;
    // XYZ → linear sRGB (D65)
    r = std::max(1e-6f,  3.2406f*X - 1.5372f       - 0.4986f*Z);
    g = std::max(1e-6f, -0.9689f*X + 1.8758f        + 0.0415f*Z);
    b = std::max(1e-6f,  0.0557f*X - 0.2040f        + 1.0570f*Z);
}

// Binary search for temperature K where R(K)/B(K) == targetRB.
// R/B is monotonically decreasing with K (higher K = cooler = less red, more blue).
static float findTempFromRBRatio(float targetRB) {
    float lo = 1000.0f, hi = 15000.0f;
    for (int i = 0; i < 64; ++i) {
        float mid = (lo + hi) * 0.5f;
        float r, g, b;
        kangToRGB(mid, r, g, b);
        if (r / b > targetRB)
            lo = mid;   // too warm  → search higher K (cooler)
        else
            hi = mid;   // too cool  → search lower K (warmer)
    }
    return (lo + hi) * 0.5f;
}

// Estimate color temperature (K) from LibRaw cam_mul[4] = {R, G1, B, G2}.
// Returns 0 if the coefficients are unusable.
static float camMulToTemp(const float cam_mul[4]) {
    // Use G1 (index 1) as the green reference; fall back to G2 (index 3).
    float g = cam_mul[1] > 0.0f ? cam_mul[1] : cam_mul[3];
    float r = cam_mul[0];
    float bl = cam_mul[2];
    if (g <= 0.0f || r <= 0.0f || bl <= 0.0f) return 0.0f;

    // R(K)/B(K)  =  (G_illum/B_illum) / (G_illum/R_illum)
    //            =  cam_mul[2]/cam_mul[0]   (B WB multiplier / R WB multiplier)
    float targetRB = bl / r;   // = R(K)/B(K) on the Planckian locus
    return findTempFromRBRatio(targetRB);
}
#endif // HAVE_LIBRAW

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

QImage RawLoader::load(const QString& filePath, ImageMetadata* meta) {
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

    // cam_mul[] is populated after unpack() — read before dcraw_process() discards it.
    if (meta) {
        float tempK = camMulToTemp(rawProcessor.imgdata.color.cam_mul);
        meta->colorTempK = (tempK >= 1500.0f && tempK <= 15000.0f) ? tempK : 0.0f;
    }
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
    Q_UNUSED(meta)
    return {};
#endif
}
