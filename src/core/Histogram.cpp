#include "Histogram.h"

#include <array>
#include <cmath>

namespace {

// CPU mirror of linear_to_srgb from color_kernels.h (clamps to [0,1]).
inline float linearToSrgb(float v) {
    if (v <= 0.0f) return 0.0f;
    if (v >= 1.0f) return 1.0f;
    return v <= 0.0031308f ? v * 12.92f
                           : 1.055f * std::pow(v, 1.0f / 2.4f) - 0.055f;
}

// Precomputed sRGB 8-bit → linear-float LUT (indices 0..255).
static const std::array<float, 256>& srgb8ToLinearLut() {
    static const std::array<float, 256> lut = [] {
        std::array<float, 256> t{};
        for (int i = 0; i < 256; ++i) {
            float v = i / 255.0f;
            t[i] = (v <= 0.04045f) ? v / 12.92f
                                   : std::pow((v + 0.055f) / 1.055f, 2.4f);
        }
        return t;
    }();
    return lut;
}

// sRGB 16-bit → linear.  No LUT (65536 entries would dominate cache); the
// polynomial cost is amortised against the much smaller RAW workloads.
inline float srgb16ToLinear(uint16_t s) {
    float v = s / 65535.0f;
    return (v <= 0.04045f) ? v / 12.92f
                           : std::pow((v + 0.055f) / 1.055f, 2.4f);
}

constexpr float kRLum = 0.2126f;
constexpr float kGLum = 0.7152f;
constexpr float kBLum = 0.0722f;

void fill8bit(const QImage& img, std::vector<uint32_t>& bins) {
    const auto& lut = srgb8ToLinearLut();
    const int w = img.width();
    const int h = img.height();
    for (int y = 0; y < h; ++y) {
        const auto* row = reinterpret_cast<const QRgb*>(img.constScanLine(y));
        for (int x = 0; x < w; ++x) {
            QRgb px = row[x];
            float lin = kRLum * lut[qRed(px)]
                      + kGLum * lut[qGreen(px)]
                      + kBLum * lut[qBlue(px)];
            float L = linearToSrgb(lin);
            int bin = int(L * 256.0f);
            if (bin < 0)        bin = 0;
            else if (bin > 255) bin = 255;
            ++bins[bin];
        }
    }
}

void fill16bit(const QImage& img, std::vector<uint32_t>& bins) {
    const int w = img.width();
    const int h = img.height();
    for (int y = 0; y < h; ++y) {
        const auto* row = reinterpret_cast<const uint16_t*>(img.constScanLine(y));
        for (int x = 0; x < w; ++x) {
            const uint16_t* p = row + 4 * x;   // RGBX64: 4 × uint16 per pixel
            float lin = kRLum * srgb16ToLinear(p[0])
                      + kGLum * srgb16ToLinear(p[1])
                      + kBLum * srgb16ToLinear(p[2]);
            float L = linearToSrgb(lin);
            int bin = int(L * 256.0f);
            if (bin < 0)        bin = 0;
            else if (bin > 255) bin = 255;
            ++bins[bin];
        }
    }
}

} // namespace

std::vector<uint32_t> computeLuminanceHistogram(const QImage& image) {
    if (image.isNull()) return {};

    std::vector<uint32_t> bins(256, 0);

    if (image.format() == QImage::Format_RGBX64) {
        fill16bit(image, bins);
    } else if (image.format() == QImage::Format_RGB32) {
        fill8bit(image, bins);
    } else {
        fill8bit(image.convertToFormat(QImage::Format_RGB32), bins);
    }
    return bins;
}
