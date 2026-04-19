#pragma once

#include <QImage>
#include <cstdint>
#include <vector>

// Computes a 256-bin perceptual-luminance histogram of an image.
//
// Bin index = clamp(floor(L * 256), 0, 255), where L is sRGB-encoded
// Rec.709 luminance of the pixel (L = sRGB(0.2126·Rlin + 0.7152·Glin + 0.0722·Blin)).
// Supports QImage::Format_RGB32 (8-bit sRGB) and QImage::Format_RGBX64
// (16-bit sRGB).  Other formats are converted to Format_RGB32 first.
// Returns an empty vector for a null image.
//
// This is a single-pass CPU routine intended to run once per image load.
// Designed for reuse across effect widgets that want an input-luminance
// distribution (e.g. the Exposure tone-curve widget).
std::vector<uint32_t> computeLuminanceHistogram(const QImage& image);
