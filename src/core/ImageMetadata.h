#pragma once

#include <cstdint>
#include <vector>

// Metadata extracted from an image file at load time.
// colorTempK == 0 means unknown / not available.
struct ImageMetadata {
    float colorTempK = 0.0f;  // as-shot color temperature in Kelvin (0 = unknown)
    float tintGM     = 0.0f;  // green-magenta tint, 0 = neutral (reserved for future use)

    // 256-bin luminance histogram of the loaded image.  Bin index is
    // floor(L * 256) clamped to [0, 255], where L is perceptual (sRGB-encoded)
    // luminance.  Empty until computed; populated once by the app on image load
    // and is not updated when effect parameters change.
    std::vector<uint32_t> luminanceHistogram;
};
