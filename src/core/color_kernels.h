#ifndef COLOR_KERNELS_H
#define COLOR_KERNELS_H

// Shared OpenCL color-space helpers used by every effect's linear-float
// pipeline kernel.  Intended to be concatenated into a kernel source string
// ahead of effect-specific kernels:
//
//   #include "color_kernels.h"
//   static const char* KERNEL_SRC = COLOR_KERNELS_SRC R"CL(
//       __kernel void adjustBrightnessLinear(__global float4* px, ...) { ... }
//   )CL";
//
// Convention for the linear-float pipeline: pixel buffers are cl_float4 in
// scene-linear sRGB primaries; the `.w` channel is unused (write 1.0 on out).

#define COLOR_KERNELS_SRC R"CL(

float srgb_to_linear(float v) {
    return v <= 0.04045f ? v * (1.0f / 12.92f)
                         : native_powr((v + 0.055f) * (1.0f / 1.055f), 2.4f);
}

float linear_to_srgb(float v) {
    v = clamp(v, 0.0f, 1.0f);
    return v <= 0.0031308f ? v * 12.92f
                           : 1.055f * native_powr(v, 1.0f / 2.4f) - 0.055f;
}

float4 srgb_to_linear4(float4 c) {
    return (float4)(srgb_to_linear(c.x),
                    srgb_to_linear(c.y),
                    srgb_to_linear(c.z),
                    c.w);
}

float4 linear_to_srgb4(float4 c) {
    return (float4)(linear_to_srgb(c.x),
                    linear_to_srgb(c.y),
                    linear_to_srgb(c.z),
                    c.w);
}

// Rec. 709 luma, evaluated on linear RGB.  The result is scene-linear
// luminance — for perceptual zone selection, pass through linear_to_srgb()
// to get a gamma-encoded L in [0, 1].
float linear_luma(float4 c) {
    return 0.2126f * c.x + 0.7152f * c.y + 0.0722f * c.z;
}

)CL"

#endif // COLOR_KERNELS_H
