#pragma once

// Metadata extracted from an image file at load time.
// colorTempK == 0 means unknown / not available.
struct ImageMetadata {
    float colorTempK = 0.0f;  // as-shot color temperature in Kelvin (0 = unknown)
    float tintGM     = 0.0f;  // green-magenta tint, 0 = neutral (reserved for future use)
};
