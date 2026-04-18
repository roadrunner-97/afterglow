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

#define SHARED_BLUR_KERNELS_F4 R"CL(

// float4 linear path — tightly packed (stride == w); .w written as 1.0.
__kernel void blurHLinear(__global const float4* in, __global float4* out,
                           int w, int h, int radius, int isGaussian)
{
    int x = get_global_id(0), y = get_global_id(1);
    if (x >= w || y >= h) return;

    float sigma = max((float)radius / 3.0f, 0.5f);
    float4 sum = (float4)(0.0f, 0.0f, 0.0f, 0.0f);
    float wsum = 0.0f;
    for (int dx = -radius; dx <= radius; ++dx) {
        int sx = clamp(x + dx, 0, w - 1);
        float4 p = in[y * w + sx];
        float ww = isGaussian ? native_exp(-0.5f * (float)(dx * dx) / (sigma * sigma)) : 1.0f;
        sum.x += ww * p.x;
        sum.y += ww * p.y;
        sum.z += ww * p.z;
        wsum += ww;
    }
    float inv = 1.0f / wsum;
    out[y * w + x] = (float4)(sum.x * inv, sum.y * inv, sum.z * inv, 1.0f);
}

__kernel void blurVLinear(__global const float4* in, __global float4* out,
                           int w, int h, int radius, int isGaussian)
{
    int x = get_global_id(0), y = get_global_id(1);
    if (x >= w || y >= h) return;

    float sigma = max((float)radius / 3.0f, 0.5f);
    float4 sum = (float4)(0.0f, 0.0f, 0.0f, 0.0f);
    float wsum = 0.0f;
    for (int dy = -radius; dy <= radius; ++dy) {
        int sy = clamp(y + dy, 0, h - 1);
        float4 p = in[sy * w + x];
        float ww = isGaussian ? native_exp(-0.5f * (float)(dy * dy) / (sigma * sigma)) : 1.0f;
        sum.x += ww * p.x;
        sum.y += ww * p.y;
        sum.z += ww * p.z;
        wsum += ww;
    }
    float inv = 1.0f / wsum;
    out[y * w + x] = (float4)(sum.x * inv, sum.y * inv, sum.z * inv, 1.0f);
}

)CL"

#define SHARED_BLUR_KERNELS R"CL(

// 8-bit path ── QImage::Format_RGB32 (0xFFRRGGBB), stride = bytesPerLine/4 ──

__kernel void blurH(__global const uint* in, __global uint* out,
                    int stride, int width, int height, int radius)
{
    int x = get_global_id(0), y = get_global_id(1);
    if (x >= width || y >= height) return;

    float sigma = max((float)radius / 3.0f, 0.5f);
    float r = 0, g = 0, b = 0, wsum = 0;
    for (int dx = -radius; dx <= radius; ++dx) {
        int sx = clamp(x + dx, 0, width - 1);
        uint p = in[y * stride + sx];
        float w = native_exp(-0.5f * (float)(dx * dx) / (sigma * sigma));
        r += w * ((p >> 16) & 0xFFu);
        g += w * ((p >>  8) & 0xFFu);
        b += w * ( p        & 0xFFu);
        wsum += w;
    }
    out[y * stride + x] = 0xFF000000u
        | ((uint)(r / wsum + 0.5f) << 16)
        | ((uint)(g / wsum + 0.5f) <<  8)
        |  (uint)(b / wsum + 0.5f);
}

__kernel void blurV(__global const uint* in, __global uint* out,
                    int stride, int width, int height, int radius)
{
    int x = get_global_id(0), y = get_global_id(1);
    if (x >= width || y >= height) return;

    float sigma = max((float)radius / 3.0f, 0.5f);
    float r = 0, g = 0, b = 0, wsum = 0;
    for (int dy = -radius; dy <= radius; ++dy) {
        int sy = clamp(y + dy, 0, height - 1);
        uint p = in[sy * stride + x];
        float w = native_exp(-0.5f * (float)(dy * dy) / (sigma * sigma));
        r += w * ((p >> 16) & 0xFFu);
        g += w * ((p >>  8) & 0xFFu);
        b += w * ( p        & 0xFFu);
        wsum += w;
    }
    out[y * stride + x] = 0xFF000000u
        | ((uint)(r / wsum + 0.5f) << 16)
        | ((uint)(g / wsum + 0.5f) <<  8)
        |  (uint)(b / wsum + 0.5f);
}

// 16-bit path ── QImage::Format_RGBX64 (ushort4, .s0=R .s1=G .s2=B .s3=unused) ──
// stride = bytesPerLine/8

__kernel void blurH16(__global const ushort4* in, __global ushort4* out,
                      int stride, int width, int height, int radius)
{
    int x = get_global_id(0), y = get_global_id(1);
    if (x >= width || y >= height) return;

    float sigma = max((float)radius / 3.0f, 0.5f);
    float r = 0, g = 0, b = 0, wsum = 0;
    for (int dx = -radius; dx <= radius; ++dx) {
        int sx = clamp(x + dx, 0, width - 1);
        ushort4 p = in[y * stride + sx];
        float w = native_exp(-0.5f * (float)(dx * dx) / (sigma * sigma));
        r += w * p.s0; g += w * p.s1; b += w * p.s2;
        wsum += w;
    }
    ushort4 o;
    o.s0 = (ushort)(r / wsum + 0.5f);
    o.s1 = (ushort)(g / wsum + 0.5f);
    o.s2 = (ushort)(b / wsum + 0.5f);
    o.s3 = 65535;
    out[y * stride + x] = o;
}

__kernel void blurV16(__global const ushort4* in, __global ushort4* out,
                      int stride, int width, int height, int radius)
{
    int x = get_global_id(0), y = get_global_id(1);
    if (x >= width || y >= height) return;

    float sigma = max((float)radius / 3.0f, 0.5f);
    float r = 0, g = 0, b = 0, wsum = 0;
    for (int dy = -radius; dy <= radius; ++dy) {
        int sy = clamp(y + dy, 0, height - 1);
        ushort4 p = in[sy * stride + x];
        float w = native_exp(-0.5f * (float)(dy * dy) / (sigma * sigma));
        r += w * p.s0; g += w * p.s1; b += w * p.s2;
        wsum += w;
    }
    ushort4 o;
    o.s0 = (ushort)(r / wsum + 0.5f);
    o.s1 = (ushort)(g / wsum + 0.5f);
    o.s2 = (ushort)(b / wsum + 0.5f);
    o.s3 = 65535;
    out[y * stride + x] = o;
}

)CL"

#endif // BLUR_KERNELS_H
