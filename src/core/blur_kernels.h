#ifndef BLUR_KERNELS_H
#define BLUR_KERNELS_H

// Shared separable Gaussian blur OpenCL kernel source.
//
// Defines four kernels:
//   blurH     — horizontal pass, 8-bit  (QImage::Format_RGB32, stride = bytesPerLine/4)
//   blurV     — vertical pass,   8-bit
//   blurH16   — horizontal pass, 16-bit (QImage::Format_RGBX64, stride = bytesPerLine/8)
//   blurV16   — vertical pass,   16-bit
//
// All kernels take: (in, out, stride, width, height, radius)
// Gaussian sigma = radius/3  (edge weight ≈ exp(-4.5) ≈ 0.01 — negligible).
// Uses native_exp() for performance; quality difference is imperceptible at
// typical blur radii (1–30 pixels).
//
// Usage (string-literal concatenation with effect-specific kernels):
//
//   #include "blur_kernels.h"
//   static const char* GPU_KERNEL_SOURCE = SHARED_BLUR_KERNELS R"CL(
//       // effect-specific kernels here
//   )CL";
//
// Consumed by: UnsharpEffect, DenoiseEffect.
//
// Also provides a float4-linear variant for the shared float4 preview pipeline:
//   blurHLinear / blurVLinear  — both take __global float4* in/out, tightly
//   packed (stride == width).  Pixels are scene-linear sRGB primaries; .w is
//   unused and written as 1.0.  Signature: (in, out, w, h, radius, isGaussian).
// Consumed by: BlurEffect, UnsharpEffect, ClarityEffect, DenoiseEffect
// (pipeline enqueueGpu path only).

#define SHARED_BLUR_KERNELS_F4                                                                                         \
    "\n"                                                                                                               \
    "// float4 linear path — tightly packed (stride == w); .w written as 1.0.\n"                                       \
    "__kernel void blurHLinear(__global const float4* in, __global float4* out,\n"                                     \
    "                           int w, int h, int radius, int isGaussian)\n"                                           \
    "{\n"                                                                                                              \
    "    int x = get_global_id(0), y = get_global_id(1);\n"                                                            \
    "    if (x >= w || y >= h) return;\n"                                                                              \
    "\n"                                                                                                               \
    "    float4 sum = in[y * w + x];\n"                                                                                \
    "    float wsum = 1.0f;\n"                                                                                         \
    "    if (isGaussian) {\n"                                                                                          \
    "        float sigma = max((float)radius / 3.0f, 0.5f);\n"                                                         \
    "        float invSigma2 = 1.0f / (sigma * sigma);\n"                                                              \
    "        float ratio = native_exp(-0.5f * invSigma2);\n"                                                           \
    "        float ratioStep = native_exp(-invSigma2);\n"                                                              \
    "        float ww = 1.0f;\n"                                                                                       \
    "        for (int d = 1; d <= radius; ++d) {\n"                                                                    \
    "            ww *= ratio; ratio *= ratioStep;\n"                                                                   \
    "            float4 a = in[y * w + clamp(x - d, 0, w - 1)];\n"                                                     \
    "            float4 b = in[y * w + clamp(x + d, 0, w - 1)];\n"                                                     \
    "            sum += ww * (a + b); wsum += 2.0f * ww;\n"                                                            \
    "        }\n"                                                                                                      \
    "    } else {\n"                                                                                                   \
    "        for (int d = 1; d <= radius; ++d) {\n"                                                                    \
    "            sum += in[y * w + clamp(x - d, 0, w - 1)] + in[y * w + clamp(x + d, 0, w - 1)];\n"                    \
    "            wsum += 2.0f;\n"                                                                                      \
    "        }\n"                                                                                                      \
    "    }\n"                                                                                                          \
    "    float inv = 1.0f / wsum;\n"                                                                                   \
    "    out[y * w + x] = (float4)(sum.x * inv, sum.y * inv, sum.z * inv, 1.0f);\n"                                    \
    "}\n"                                                                                                              \
    "\n"                                                                                                               \
    "__kernel void blurVLinear(__global const float4* in, __global float4* out,\n"                                     \
    "                           int w, int h, int radius, int isGaussian)\n"                                           \
    "{\n"                                                                                                              \
    "    int x = get_global_id(0), y = get_global_id(1);\n"                                                            \
    "    if (x >= w || y >= h) return;\n"                                                                              \
    "\n"                                                                                                               \
    "    float4 sum = in[y * w + x];\n"                                                                                \
    "    float wsum = 1.0f;\n"                                                                                         \
    "    if (isGaussian) {\n"                                                                                          \
    "        float sigma = max((float)radius / 3.0f, 0.5f);\n"                                                         \
    "        float invSigma2 = 1.0f / (sigma * sigma);\n"                                                              \
    "        float ratio = native_exp(-0.5f * invSigma2);\n"                                                           \
    "        float ratioStep = native_exp(-invSigma2);\n"                                                              \
    "        float ww = 1.0f;\n"                                                                                       \
    "        for (int d = 1; d <= radius; ++d) {\n"                                                                    \
    "            ww *= ratio; ratio *= ratioStep;\n"                                                                   \
    "            float4 a = in[clamp(y - d, 0, h - 1) * w + x];\n"                                                     \
    "            float4 b = in[clamp(y + d, 0, h - 1) * w + x];\n"                                                     \
    "            sum += ww * (a + b); wsum += 2.0f * ww;\n"                                                            \
    "        }\n"                                                                                                      \
    "    } else {\n"                                                                                                   \
    "        for (int d = 1; d <= radius; ++d) {\n"                                                                    \
    "            sum += in[clamp(y - d, 0, h - 1) * w + x] + in[clamp(y + d, 0, h - 1) * w + x];\n"                    \
    "            wsum += 2.0f;\n"                                                                                      \
    "        }\n"                                                                                                      \
    "    }\n"                                                                                                          \
    "    float inv = 1.0f / wsum;\n"                                                                                   \
    "    out[y * w + x] = (float4)(sum.x * inv, sum.y * inv, sum.z * inv, 1.0f);\n"                                    \
    "}\n"                                                                                                              \
    "\n"

#define SHARED_BLUR_KERNELS                                                                                            \
    "\n"                                                                                                               \
    "// 8-bit path ── QImage::Format_RGB32 (0xFFRRGGBB), stride = bytesPerLine/4 ──\n"                                 \
    "\n"                                                                                                               \
    "__kernel void blurH(__global const uint* in, __global uint* out,\n"                                               \
    "                    int stride, int width, int height, int radius)\n"                                             \
    "{\n"                                                                                                              \
    "    int x = get_global_id(0), y = get_global_id(1);\n"                                                            \
    "    if (x >= width || y >= height) return;\n"                                                                     \
    "\n"                                                                                                               \
    "    float sigma = max((float)radius / 3.0f, 0.5f);\n"                                                             \
    "    float r = 0, g = 0, b = 0, wsum = 0;\n"                                                                       \
    "    for (int dx = -radius; dx <= radius; ++dx) {\n"                                                               \
    "        int sx = clamp(x + dx, 0, width - 1);\n"                                                                  \
    "        uint p = in[y * stride + sx];\n"                                                                          \
    "        float w = native_exp(-0.5f * (float)(dx * dx) / (sigma * sigma));\n"                                      \
    "        r += w * ((p >> 16) & 0xFFu);\n"                                                                          \
    "        g += w * ((p >>  8) & 0xFFu);\n"                                                                          \
    "        b += w * ( p        & 0xFFu);\n"                                                                          \
    "        wsum += w;\n"                                                                                             \
    "    }\n"                                                                                                          \
    "    out[y * stride + x] = 0xFF000000u\n"                                                                          \
    "        | ((uint)(r / wsum + 0.5f) << 16)\n"                                                                      \
    "        | ((uint)(g / wsum + 0.5f) <<  8)\n"                                                                      \
    "        |  (uint)(b / wsum + 0.5f);\n"                                                                            \
    "}\n"                                                                                                              \
    "\n"                                                                                                               \
    "__kernel void blurV(__global const uint* in, __global uint* out,\n"                                               \
    "                    int stride, int width, int height, int radius)\n"                                             \
    "{\n"                                                                                                              \
    "    int x = get_global_id(0), y = get_global_id(1);\n"                                                            \
    "    if (x >= width || y >= height) return;\n"                                                                     \
    "\n"                                                                                                               \
    "    float sigma = max((float)radius / 3.0f, 0.5f);\n"                                                             \
    "    float r = 0, g = 0, b = 0, wsum = 0;\n"                                                                       \
    "    for (int dy = -radius; dy <= radius; ++dy) {\n"                                                               \
    "        int sy = clamp(y + dy, 0, height - 1);\n"                                                                 \
    "        uint p = in[sy * stride + x];\n"                                                                          \
    "        float w = native_exp(-0.5f * (float)(dy * dy) / (sigma * sigma));\n"                                      \
    "        r += w * ((p >> 16) & 0xFFu);\n"                                                                          \
    "        g += w * ((p >>  8) & 0xFFu);\n"                                                                          \
    "        b += w * ( p        & 0xFFu);\n"                                                                          \
    "        wsum += w;\n"                                                                                             \
    "    }\n"                                                                                                          \
    "    out[y * stride + x] = 0xFF000000u\n"                                                                          \
    "        | ((uint)(r / wsum + 0.5f) << 16)\n"                                                                      \
    "        | ((uint)(g / wsum + 0.5f) <<  8)\n"                                                                      \
    "        |  (uint)(b / wsum + 0.5f);\n"                                                                            \
    "}\n"                                                                                                              \
    "\n"                                                                                                               \
    "// 16-bit path ── QImage::Format_RGBX64 (ushort4, .s0=R .s1=G .s2=B .s3=unused) ──\n"                             \
    "// stride = bytesPerLine/8\n"                                                                                     \
    "\n"                                                                                                               \
    "__kernel void blurH16(__global const ushort4* in, __global ushort4* out,\n"                                       \
    "                      int stride, int width, int height, int radius)\n"                                           \
    "{\n"                                                                                                              \
    "    int x = get_global_id(0), y = get_global_id(1);\n"                                                            \
    "    if (x >= width || y >= height) return;\n"                                                                     \
    "\n"                                                                                                               \
    "    float sigma = max((float)radius / 3.0f, 0.5f);\n"                                                             \
    "    float r = 0, g = 0, b = 0, wsum = 0;\n"                                                                       \
    "    for (int dx = -radius; dx <= radius; ++dx) {\n"                                                               \
    "        int sx = clamp(x + dx, 0, width - 1);\n"                                                                  \
    "        ushort4 p = in[y * stride + sx];\n"                                                                       \
    "        float w = native_exp(-0.5f * (float)(dx * dx) / (sigma * sigma));\n"                                      \
    "        r += w * p.s0; g += w * p.s1; b += w * p.s2;\n"                                                           \
    "        wsum += w;\n"                                                                                             \
    "    }\n"                                                                                                          \
    "    ushort4 o;\n"                                                                                                 \
    "    o.s0 = (ushort)(r / wsum + 0.5f);\n"                                                                          \
    "    o.s1 = (ushort)(g / wsum + 0.5f);\n"                                                                          \
    "    o.s2 = (ushort)(b / wsum + 0.5f);\n"                                                                          \
    "    o.s3 = 65535;\n"                                                                                              \
    "    out[y * stride + x] = o;\n"                                                                                   \
    "}\n"                                                                                                              \
    "\n"                                                                                                               \
    "__kernel void blurV16(__global const ushort4* in, __global ushort4* out,\n"                                       \
    "                      int stride, int width, int height, int radius)\n"                                           \
    "{\n"                                                                                                              \
    "    int x = get_global_id(0), y = get_global_id(1);\n"                                                            \
    "    if (x >= width || y >= height) return;\n"                                                                     \
    "\n"                                                                                                               \
    "    float sigma = max((float)radius / 3.0f, 0.5f);\n"                                                             \
    "    float r = 0, g = 0, b = 0, wsum = 0;\n"                                                                       \
    "    for (int dy = -radius; dy <= radius; ++dy) {\n"                                                               \
    "        int sy = clamp(y + dy, 0, height - 1);\n"                                                                 \
    "        ushort4 p = in[sy * stride + x];\n"                                                                       \
    "        float w = native_exp(-0.5f * (float)(dy * dy) / (sigma * sigma));\n"                                      \
    "        r += w * p.s0; g += w * p.s1; b += w * p.s2;\n"                                                           \
    "        wsum += w;\n"                                                                                             \
    "    }\n"                                                                                                          \
    "    ushort4 o;\n"                                                                                                 \
    "    o.s0 = (ushort)(r / wsum + 0.5f);\n"                                                                          \
    "    o.s1 = (ushort)(g / wsum + 0.5f);\n"                                                                          \
    "    o.s2 = (ushort)(b / wsum + 0.5f);\n"                                                                          \
    "    o.s3 = 65535;\n"                                                                                              \
    "    out[y * stride + x] = o;\n"                                                                                   \
    "}\n"                                                                                                              \
    "\n"

#endif // BLUR_KERNELS_H
