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
//
// NB: this is *not* a raw string.  GCC < 14 has a longstanding bug where a
// multi-line raw string literal inside a #define replacement list isn't
// tokenised correctly, so we fall back to one narrow-string-per-line.  Each
// line ends with an explicit \n so the OpenCL JIT compiler can still report
// useful line numbers.

#define COLOR_KERNELS_SRC                                                       \
    "\n"                                                                        \
    "float srgb_to_linear(float v) {\n"                                         \
    "    return v <= 0.04045f ? v * (1.0f / 12.92f)\n"                          \
    "                         : native_powr((v + 0.055f) * (1.0f / 1.055f), 2.4f);\n" \
    "}\n"                                                                       \
    "\n"                                                                        \
    "float linear_to_srgb(float v) {\n"                                         \
    "    v = clamp(v, 0.0f, 1.0f);\n"                                           \
    "    return v <= 0.0031308f ? v * 12.92f\n"                                 \
    "                           : 1.055f * native_powr(v, 1.0f / 2.4f) - 0.055f;\n" \
    "}\n"                                                                       \
    "\n"                                                                        \
    "float4 srgb_to_linear4(float4 c) {\n"                                      \
    "    return (float4)(srgb_to_linear(c.x),\n"                                \
    "                    srgb_to_linear(c.y),\n"                                \
    "                    srgb_to_linear(c.z),\n"                                \
    "                    c.w);\n"                                               \
    "}\n"                                                                       \
    "\n"                                                                        \
    "float4 linear_to_srgb4(float4 c) {\n"                                      \
    "    return (float4)(linear_to_srgb(c.x),\n"                                \
    "                    linear_to_srgb(c.y),\n"                                \
    "                    linear_to_srgb(c.z),\n"                                \
    "                    c.w);\n"                                               \
    "}\n"                                                                       \
    "\n"                                                                        \
    "// Rec. 709 luma, evaluated on linear RGB.  The result is scene-linear\n"  \
    "// luminance — for perceptual zone selection, pass through linear_to_srgb()\n" \
    "// to get a gamma-encoded L in [0, 1].\n"                                  \
    "float linear_luma(float4 c) {\n"                                           \
    "    return 0.2126f * c.x + 0.7152f * c.y + 0.0722f * c.z;\n"               \
    "}\n"                                                                       \
    "\n"

#endif // COLOR_KERNELS_H
